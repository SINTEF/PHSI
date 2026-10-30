"""
Fit analytic coefficients with bounds and remove small terms.

Inspired by Myhr, Eidnes and Kutz, sparse identification of port-Hamiltonian
systems (https://arxiv.org/abs/2609.20399; author code not yet public).
See docs/attribution.md and docs/source-provenance.json for attributions to
external sources.
"""

import copy
import time

import numpy as np
import torch
from scipy.optimize import OptimizeResult, lsq_linear

from .errors import SupportNonconvergence


def _array(value):
    return (
        value.detach().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value, dtype=float)
    )


def _library(library, kind, x, field):
    array_method = getattr(library, "numpy_" + kind, None)
    if array_method is not None:
        return np.asarray(array_method(x), dtype=float)
    return _array(
        getattr(library, kind)(
            torch.as_tensor(x, dtype=field.sigma.dtype, device=field.sigma.device)
        )
    )


def _damping_scales(field):
    shape = tuple(field.damping.raw.shape)
    scales = _array(field.damping_scales)
    if scales.shape != shape or not np.isfinite(scales).all() or (scales < 0).any():
        raise ValueError(
            "Analytic damping scales must be finite, nonnegative and match its coefficient shape."
        )
    return scales


def build_design(field, x, rates, u):
    """
    Build the scaled regression matrix, targets and count of unrestricted coefficients.

    Columns represent energy, learned gain (if any), then damping with nonnegative
    coefficients. Observed position rates keep the fit linear in damping.
    Divide each row by its coordinate's training standard deviation.
    """
    if (
        field.energy.kind != "analytic"
        or field.gain.kind not in ("analytic", "known", "none")
        or field.damping.kind not in ("constant", "analytic")
    ):
        raise ValueError(
            "Regression requires fixed energy, gain and damping libraries or known/absent gain."
        )
    x, rates, u = (
        np.asarray(x, dtype=float),
        np.asarray(rates, dtype=float),
        np.asarray(u, dtype=float),
    )
    dimension, n = field.n_state, field.n_dof
    if (
        x.ndim != 2
        or x.shape[1] != dimension
        or rates.shape != x.shape
        or u.shape != x.shape[:1]
        or len(x) == 0
    ):
        raise ValueError(
            "Regression requires state/rate arrays (samples, states) and input (samples,)."
        )
    if not all(np.isfinite(v).all() for v in (x, rates, u)):
        raise ValueError("Regression observations, rates and input must be finite.")
    sigma, scales = _array(field.sigma), _array(field.energy.scale)
    if (
        sigma.shape != (dimension,)
        or not np.all(np.isfinite(sigma) & (sigma > 0))
        or not np.all(np.isfinite(scales) & (scales > 0))
    ):
        raise ValueError("State and energy scales must be finite and positive.")
    gradients = _library(field.energy.library, "gradients", x, field)
    K = len(scales)
    if gradients.shape != (len(x), K, dimension):
        raise ValueError("Energy gradients have an incompatible shape.")
    G = gradients / scales[None, :, None]
    learned_gain = field.gain.kind == "analytic"
    L = len(field.gain.scale) if learned_gain else 0
    p = torch.as_tensor(x[:, n:], dtype=field.sigma.dtype, device=field.sigma.device)
    psi = _array(field.damping.features(p))
    F = field.damping.raw.shape[-1] if field.damping.kind == "analytic" else 1
    if psi.shape != (len(x), n, F) or not np.isfinite(psi).all() or (psi < 0).any():
        raise ValueError(
            "Damping features must be finite, nonnegative, with shape (samples, pairs, features)."
        )
    normalization = (
        np.where(_damping_scales(field) > 0, _damping_scales(field), 1.0)
        if field.damping.kind == "analytic"
        else np.ones((n, 1))
    )
    A = np.zeros((len(x), dimension, K + L + n * F))
    A[:, :n, :K] = G[:, :, n:].transpose(0, 2, 1)
    A[:, n:, :K] = -G[:, :, :n].transpose(0, 2, 1)
    if learned_gain:
        gain_scales = _array(field.gain.scale)
        if not np.all(np.isfinite(gain_scales) & (gain_scales > 0)):
            raise ValueError("Gain scales must be finite and positive.")
        argument = x[:, field.input_argument : field.input_argument + 1]
        gain = _library(field.gain.library, "values", argument, field)
        if gain.shape != (len(x), L):
            raise ValueError("Gain features have an incompatible shape.")
        A[:, field.input_index, K : K + L] = gain / gain_scales[None, :] * u[:, None]
    for i in range(n):
        A[:, n + i, K + L + i * F : K + L + (i + 1) * F] = (
            -psi[:, i, :] * rates[:, i : i + 1] / normalization[i]
        )
    b = rates.copy()
    if field.gain.kind == "known":
        b[:, field.input_index] -= u
    A, b = (
        (A / sigma[None, :, None]).reshape(len(x) * dimension, -1),
        (b / sigma).reshape(-1),
    )
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("Regression design and target must be finite after scaling.")
    return A, b, K + L


def bounded_lstsq(A, b, n_signed):
    """
    Fit coefficients with either sign for energy/gain and nonnegative damping.

    Returned coefficients correct roundoff at the damping bound; the accompanying
    result retains the original solver coefficients and diagnostics.
    """
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    if (
        A.ndim != 2
        or b.shape != (A.shape[0],)
        or A.shape[0] == 0
        or not np.isfinite(A).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError(
            "A finite nonempty-row matrix and matching finite target are required."
        )
    if not isinstance(n_signed, (int, np.integer)) or not 0 <= n_signed <= A.shape[1]:
        raise ValueError("The signed column count must fit the design.")
    if A.shape[1] == 0:
        return np.empty(0), OptimizeResult(
            success=True,
            status=3,
            message="Empty design.",
            nit=0,
            cost=0.5 * float(b @ b),
        )
    lower = np.r_[np.full(n_signed, -np.inf), np.zeros(A.shape[1] - n_signed)]
    result = lsq_linear(
        A,
        b,
        bounds=(lower, np.full(A.shape[1], np.inf)),
        method="bvls",
        tol=1e-12,
        max_iter=5000,
    )
    coefficients = result.x.copy()
    if result.success:
        damping = coefficients[n_signed:]
        # BVLS can leave cancellation roundoff at an active zero bound.
        if np.any(damping < -64 * np.finfo(float).eps):
            raise RuntimeError(
                "Bounded least squares returned negative damping beyond roundoff."
            )
        damping[damping < 0] = 0.0
    return coefficients, result


def _samples(trajectory, method):
    x, t = trajectory.x, trajectory.t
    if method == "midpoint":
        if len(x) < 2:
            raise ValueError(
                "Midpoint regression requires two observations per trajectory."
            )
        return (
            0.5 * (x[1:] + x[:-1]),
            np.diff(x, axis=0) / np.diff(t)[:, None],
            trajectory.signal(0.5 * (t[1:] + t[:-1])),
        )
    if trajectory.derivative is not None:
        return x, trajectory.derivative, trajectory.signal(t)
    if len(x) < 3:
        raise ValueError(
            "Derivative regression requires three raw observations per trajectory."
        )
    return (
        x[1:-1],
        (x[2:] - x[:-2]) / (t[2:] - t[:-2])[:, None],
        trajectory.signal(t[1:-1]),
    )


def _apply(field, coefficient, active):
    K = len(field.energy.coef)
    L = len(field.gain.coef) if field.gain.kind == "analytic" else 0
    with torch.no_grad():
        field.energy.coef.copy_(
            torch.as_tensor(
                coefficient[:K],
                dtype=field.energy.coef.dtype,
                device=field.energy.coef.device,
            )
        )
        field.energy.mask.copy_(
            torch.as_tensor(active[:K], device=field.energy.mask.device)
        )
        if L:
            field.gain.coef.copy_(
                torch.as_tensor(
                    coefficient[K : K + L],
                    dtype=field.gain.coef.dtype,
                    device=field.gain.coef.device,
                )
            )
            field.gain.mask.copy_(
                torch.as_tensor(active[K : K + L], device=field.gain.mask.device)
            )
        physical = coefficient[K + L :].reshape(field.damping.raw.shape)
        if field.damping.kind == "analytic":
            scales = _damping_scales(field)
            physical = physical / np.where(scales > 0, scales, 1.0)
            field.damping.mask.copy_(
                torch.as_tensor(
                    active[K + L :].reshape(physical.shape),
                    device=field.damping.mask.device,
                )
            )
        field.damping.set_physical(physical)


def _clock(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(
            "Aggregate regression clock exhausted before completing refit."
        )


def fit_regression(
    field, trajectories, thresholds, *, method="derivative", max_iter=20, deadline=None
):
    """
    Fit a model copy and return details of each refit.

    Missing or zero thresholds keep all terms, including zero coefficients.
    Positive thresholds remove small terms from all requested components together;
    removed terms never return. Raise SupportNonconvergence if terms are still
    being removed at the iteration limit, without another fit.
    """
    if method not in ("derivative", "midpoint"):
        raise ValueError("Regression supports derivative or midpoint observations.")
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, int)
        or not 1 <= max_iter <= 20
    ):
        raise ValueError(
            "The regression support-refit cap must be between one and twenty."
        )
    if not isinstance(thresholds, dict) or set(thresholds) - {
        "energy",
        "gain",
        "damping",
    }:
        raise ValueError(
            "Thresholds must be a dictionary with energy/gain/damping keys."
        )
    thresholds = {key: float(value) for key, value in thresholds.items()}
    if any(not np.isfinite(value) or value < 0 for value in thresholds.values()):
        raise ValueError("Thresholds must be finite and nonnegative.")
    if deadline is not None and not np.isfinite(deadline):
        raise ValueError("deadline must be an absolute finite monotonic-clock time.")
    _clock(deadline)
    fitted = copy.deepcopy(field)
    for component in (fitted.energy, fitted.gain, fitted.damping):
        if hasattr(component, "mask"):
            component.mask.fill_(1.0)
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("At least one training trajectory is required.")
    designs = [build_design(fitted, *_samples(tr, method)) for tr in trajectories]
    A, b, n_signed = (
        np.concatenate([v[0] for v in designs]),
        np.concatenate([v[1] for v in designs]),
        designs[0][2],
    )
    K = len(fitted.energy.coef)
    L = len(fitted.gain.coef) if fitted.gain.kind == "analytic" else 0
    blocks = {
        "energy": slice(0, K),
        "gain": slice(K, K + L),
        "damping": slice(K + L, A.shape[1]),
    }
    active = np.ones(A.shape[1], dtype=bool)
    history = []
    for iteration in range(1, max_iter + 1):
        _clock(deadline)
        columns = np.flatnonzero(active)
        solution, result = bounded_lstsq(A[:, columns], b, int(active[:n_signed].sum()))
        _clock(deadline)
        if (
            not result.success
            or solution.shape != columns.shape
            or not np.isfinite(solution).all()
        ):
            raise RuntimeError(
                "Bounded least squares did not return a finite completed solution."
            )
        coefficient = np.zeros(A.shape[1])
        coefficient[columns] = solution
        proposed = active.copy()
        for name, block in blocks.items():
            threshold = thresholds.get(name, 0.0)
            if threshold == 0 or (
                name == "damping" and fitted.damping.kind != "analytic"
            ):
                continue
            proposed[block] &= np.abs(coefficient[block]) >= threshold
            if name == "damping":
                proposed[block] &= _damping_scales(fitted).reshape(-1) > 0
        changed = not np.array_equal(proposed, active)
        objective = float(np.mean((A @ coefficient - b) ** 2))
        history.append(
            {
                "iteration": iteration,
                "objective": objective,
                "support_changed": changed,
                "active": {
                    name: active[block].tolist() for name, block in blocks.items()
                },
                "next_active": {
                    name: proposed[block].tolist() for name, block in blocks.items()
                },
                "solver_iterations": int(getattr(result, "nit", 0) or 0),
            }
        )
        if not changed:
            _apply(fitted, coefficient, active)
            return fitted, history
        if iteration == max_iter:
            _apply(fitted, coefficient, active)
            error = SupportNonconvergence(
                f"Component supports still changed at refit {max_iter}; no extra solve was performed."
            )
            error.history, error.partial_field = history, fitted
            raise error
        active = proposed
