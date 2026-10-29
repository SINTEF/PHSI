"""
Measured observations, input interpolation, and training-only preprocessing.

See docs/attribution.md for the method distinctions
and docs/source-provenance.json for attributions to external sources.
"""

from dataclasses import dataclass

import numpy as np
import torch


class InputSignal:
    """
    A known scalar forcing signal over the observation times.

    Sampled values are joined by straight lines or held until the next sample.
    Signal functions should accept time arrays; if that raises TypeError or
    ValueError, they are called once per time instead.
    """

    def __init__(self, t, u, interpolation="linear", forced=True):
        self.t = np.array(t, dtype=float, copy=True)
        if (
            self.t.ndim != 1
            or len(self.t) < 2
            or not np.isfinite(self.t).all()
            or not (np.diff(self.t) > 0).all()
        ):
            raise ValueError(
                "Input times must be finite and strictly increasing, with at least two samples."
            )
        if interpolation not in ("linear", "previous"):
            raise ValueError("Input interpolation must be linear or previous.")
        if u is None and forced:
            raise ValueError("A forced model requires a known input.")
        self.interpolation, self.forced = interpolation, bool(forced)
        self._function = u if callable(u) else None
        self._values = None
        if self._function is None:
            values = np.zeros(len(self.t)) if u is None else np.asarray(u, dtype=float)
            if values.shape == (len(self.t), 1):
                values = values[:, 0]
            if values.shape != self.t.shape or not np.isfinite(values).all():
                raise ValueError(
                    "Sampled input must contain one finite scalar per observation time."
                )
            self._values = values.copy()
        # Validate callable outputs at the supplied observation times as well.
        values = self(self.t)
        if not forced and np.any(values != 0):
            raise ValueError("An unforced model cannot receive a nonzero input.")

    def _query(self, t):
        tensor = torch.is_tensor(t)
        if tensor:
            dtype = t.dtype if t.is_floating_point() else torch.float64
            q = t.to(dtype=dtype)
            eps = torch.finfo(dtype).eps
            valid = torch.isfinite(q).all()
        else:
            q = np.asarray(t, dtype=float)
            eps = np.finfo(float).eps
            valid = np.isfinite(q).all()
        tolerance = 64 * eps * max(1.0, abs(self.t[0]), abs(self.t[-1]))
        if (
            not valid
            or bool((q < self.t[0] - tolerance).any())
            or bool((q > self.t[-1] + tolerance).any())
        ):
            raise ValueError(
                "Input query is nonfinite or outside its observed time interval."
            )
        return (
            q.clamp(float(self.t[0]), float(self.t[-1]))
            if tensor
            else np.clip(q, self.t[0], self.t[-1])
        ), tensor

    def __call__(self, t):
        q, tensor = self._query(t)
        if self._function is not None:
            # Callables need not know about Torch; sampled inputs retain query gradients.
            argument = q.detach().cpu().numpy() if tensor else q
            try:
                out = self._function(argument)
            except (TypeError, ValueError):
                out = np.array(
                    [self._function(float(v)) for v in argument.reshape(-1)]
                ).reshape(argument.shape)
            if torch.is_tensor(out):
                out = out.detach().cpu().numpy()
            out = np.asarray(out, dtype=float)
            if out.shape != argument.shape or not np.isfinite(out).all():
                raise ValueError(
                    "An input callable must return finite scalars with the queried time shape."
                )
            if not self.forced and np.any(out != 0):
                raise ValueError("An unforced model cannot receive a nonzero input.")
            return (
                torch.as_tensor(out, dtype=q.dtype, device=q.device) if tensor else out
            )
        if tensor:
            times = torch.as_tensor(self.t, dtype=q.dtype, device=q.device)
            values = torch.as_tensor(self._values, dtype=q.dtype, device=q.device)
            i = torch.searchsorted(times, q.contiguous(), right=True) - 1
            i = i.clamp(0, len(times) - 2)
            if self.interpolation == "previous":
                return values[
                    torch.searchsorted(times, q.contiguous(), right=True)
                    .sub(1)
                    .clamp(0, len(times) - 1)
                ]
            weight = (q - times[i]) / (times[i + 1] - times[i])
            return values[i] + weight * (values[i + 1] - values[i])
        if self.interpolation == "previous":
            return self._values[
                np.searchsorted(self.t, q, side="right").clip(1, len(self.t)) - 1
            ]
        return np.interp(q, self.t, self._values)

    @property
    def is_held(self):
        """
        Whether sampled values are held constant until the next sample.
        """
        return self._function is None and self.interpolation == "previous"

    def subset(self, indices):
        """
        Limit the signal to the selected observation times.
        """
        return InputSignal(
            self.t[indices],
            self._function if self._function is not None else self._values[indices],
            self.interpolation,
            self.forced,
        )


@dataclass
class Trajectory:
    x: np.ndarray
    t: np.ndarray
    signal: InputSignal
    derivative: np.ndarray | None = None

    def __post_init__(self):
        self.x, self.t = (
            np.array(self.x, dtype=float, copy=True),
            np.array(self.t, dtype=float, copy=True),
        )
        if (
            self.x.ndim != 2
            or len(self.x) != len(self.t)
            or self.x.shape[1] < 2
            or self.x.shape[1] % 2
        ):
            raise ValueError("States must have shape (samples, even state dimension).")
        if (
            self.t.ndim != 1
            or len(self.t) < 2
            or not np.isfinite(self.t).all()
            or not np.isfinite(self.x).all()
        ):
            raise ValueError(
                "States and times must be finite with at least two observations."
            )
        dt = np.diff(self.t)
        if not (dt > 0).all() or not np.allclose(dt, dt[0], rtol=1e-7, atol=1e-12):
            raise ValueError(
                "Each trajectory requires a uniform, strictly increasing time grid."
            )
        self.signal(self.t)
        if self.derivative is not None:
            self.derivative = np.array(self.derivative, dtype=float, copy=True)
            if (
                self.derivative.shape != self.x.shape
                or not np.isfinite(self.derivative).all()
            ):
                raise ValueError(
                    "Derivatives must be finite and have the state array shape."
                )

    @property
    def u(self):
        return self.signal(self.t)


@dataclass
class Observations:
    train: list[Trajectory]
    training: list[Trajectory]
    validation: list[Trajectory]
    sigma: np.ndarray
    raw_sigma: np.ndarray
    n_state: int
    split_info: dict


def _trajectories(X, t, u, forced, interpolation):
    if isinstance(X, np.ndarray) and X.ndim == 2:
        states = [X]
    elif isinstance(X, np.ndarray) and X.ndim == 3:
        states = list(X)
    elif isinstance(X, (list, tuple)) and len(X):
        states = list(X)
    else:
        raise ValueError(
            "Supply a state array or a nonempty list of trajectory arrays."
        )
    n = len(states)
    # A sequence of scalars denotes a shared time grid, not one time per trajectory.
    shared_time = (
        np.ndim(t) == 1 if isinstance(t, np.ndarray) else all(np.isscalar(v) for v in t)
    )
    times = [t] * n if shared_time else list(t)
    if len(times) != n:
        raise ValueError("Supply one time grid per trajectory, or a shared grid.")
    if u is None or callable(u):
        inputs = [u] * n
    elif n == 1:
        # Also accept the list-of-trajectories representation for a single trajectory.
        inputs = (
            list(u)
            if isinstance(u, (list, tuple))
            and len(u) == 1
            and (callable(u[0]) or np.ndim(u[0]) > 0)
            else [u]
        )
    else:
        inputs = list(u)
        if len(inputs) != n:
            raise ValueError("Supply one known input per trajectory.")
    result = [
        Trajectory(x, ti, InputSignal(ti, ui, interpolation, forced))
        for x, ti, ui in zip(states, times, inputs)
    ]
    if len({tr.x.shape[1] for tr in result}) != 1:
        raise ValueError("All trajectories must have the same state dimension.")
    return result


def _slice(tr, indices):
    return Trajectory(tr.x[indices], tr.t[indices], tr.signal.subset(indices))


def prepare_data(
    X,
    t,
    u=None,
    *,
    validation=None,
    forced=False,
    interpolation="linear",
    preprocessing=None,
    method="integration_scheme",
    window_steps=10,
):
    """
    Split observations into training and validation before scaling or smoothing.

    Without explicit validation data, use the first 80 percent of each trajectory
    for training and the remainder for validation. Interpolation and smoothing
    stay within each split. sigma is each coordinate's population standard
    deviation across unsmoothed training observations, with a minimum of 1e-3.
    """
    if method not in (
        "integration_scheme",
        "simulation_error",
        "derivative",
        "midpoint",
    ):
        raise ValueError("Unknown fitting method.")
    if (
        isinstance(window_steps, bool)
        or not isinstance(window_steps, (int, np.integer))
        or window_steps < 1
    ):
        raise ValueError("window_steps must be a positive integer.")
    supplied = _trajectories(X, t, u, forced, interpolation)
    if validation is None:
        train, valid, counts = [], [], []
        for tr in supplied:
            count = 4 * len(tr.x) // 5
            if count < 2 or len(tr.x) - count < 2:
                raise ValueError(
                    "Each split needs at least two observations per trajectory."
                )
            train.append(_slice(tr, slice(None, count)))
            valid.append(_slice(tr, slice(count, None)))
            counts.append(
                {"supplied": len(tr.x), "train": count, "validation": len(tr.x) - count}
            )
        info = {"strategy": "chronological_80_20", "trajectories": counts}
    else:
        if not isinstance(validation, (tuple, list)) or len(validation) != 3:
            raise ValueError("validation must be (states, times, known input).")
        train, valid = supplied, _trajectories(*validation, forced, interpolation)
        info = {
            "strategy": "explicit_validation",
            "train_lengths": [len(z.x) for z in train],
            "validation_lengths": [len(z.x) for z in valid],
        }
    dimension = train[0].x.shape[1]
    if any(z.x.shape[1] != dimension for z in valid):
        raise ValueError("Validation must have the training state dimension.")
    raw_sigma = np.concatenate([z.x for z in train]).std(axis=0)
    if not np.isfinite(raw_sigma).all():
        raise ValueError("Raw training standard deviations must be finite.")
    sigma = np.maximum(raw_sigma, 1e-3)
    if preprocessing is None:
        training = list(train)
        info["preprocessing"] = None
    else:
        from scipy.signal import savgol_filter

        cfg = {"window": 11, "order": 3} if preprocessing == "savgol" else preprocessing
        if (
            not isinstance(cfg, dict)
            or cfg.get("kind", "savgol") != "savgol"
            or set(cfg) - {"kind", "window", "order"}
        ):
            raise ValueError(
                "Preprocessing must be savgol or a window/order dictionary."
            )
        window, order = cfg.get("window", 11), cfg.get("order", 3)
        if (
            isinstance(window, bool)
            or isinstance(order, bool)
            or not isinstance(window, (int, np.integer))
            or not isinstance(order, (int, np.integer))
            or window % 2 != 1
            or not 1 <= order < window
        ):
            raise ValueError(
                "Use an odd smoothing window larger than its positive polynomial order."
            )
        trim, training = window // 2, []
        for tr in train:
            if window >= len(tr.t) or len(tr.t) - 2 * trim < 2:
                raise ValueError(
                    "The full smoothing window and two retained samples must fit in training."
                )
            selection = slice(trim, -trim)
            filtered = savgol_filter(tr.x, window, order, axis=0, mode="interp")
            derivative = savgol_filter(
                tr.x,
                window,
                order,
                deriv=1,
                delta=tr.t[1] - tr.t[0],
                axis=0,
                mode="interp",
            )
            training.append(
                Trajectory(
                    filtered[selection],
                    tr.t[selection],
                    tr.signal.subset(selection),
                    derivative[selection],
                )
            )
        info["preprocessing"] = {
            "kind": "savgol",
            "window": int(window),
            "order": int(order),
            "trim": int(trim),
        }
    minimum = (
        window_steps + 1
        if method == "simulation_error"
        else 3
        if method == "derivative" and preprocessing is None
        else 2
    )
    if any(len(z.x) < minimum for z in training):
        raise ValueError(
            f"{method} requires at least {minimum} retained training observations per trajectory."
        )
    info["processed_train_lengths"] = [len(z.x) for z in training]
    return Observations(train, training, valid, sigma, raw_sigma, dimension, info)
