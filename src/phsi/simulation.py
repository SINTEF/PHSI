"""
Simulate at requested times and preserve partial results if integration fails.
"""

import time

import numpy as np
from scipy.integrate import DOP853

from .data import InputSignal
from .errors import SimulationError


def simulate_field(
    field,
    x0,
    t,
    u,
    *,
    input_interpolation="linear",
    rtol=1e-9,
    atol=1e-11,
    deadline=None,
    max_state=1e6,
    max_evaluations=100000,
):
    """
    Return all requested states or raise SimulationError with partial predictions.

    The Dormand–Prince eighth-order solver evaluates the applied signal within
    each step and interpolates between steps to return states at requested times.
    """
    times, initial = np.asarray(t, dtype=float), np.asarray(x0, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or not (np.diff(times) > 0).all()
    ):
        raise ValueError(
            "Simulation times must be finite and strictly increasing, with at least two samples."
        )
    if initial.shape != (field.n_state,) or not np.isfinite(initial).all():
        raise ValueError("Simulation initial state must be a finite state vector.")
    if (
        not np.isfinite(rtol)
        or rtol <= 0
        or not np.all(np.isfinite(atol))
        or not np.all(np.asarray(atol) > 0)
    ):
        raise ValueError("Simulation tolerances must be positive and finite.")
    if not np.isfinite(max_state) or max_state <= 0:
        raise ValueError("max_state must be positive and finite.")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, (int, np.integer))
        or max_evaluations < 1
    ):
        raise ValueError("max_evaluations must be a positive integer.")
    if deadline is not None and not np.isfinite(deadline):
        raise ValueError("deadline must be a finite absolute monotonic-clock time.")
    signal = (
        u
        if isinstance(u, InputSignal)
        else InputSignal(
            times, u, input_interpolation, forced=field.input_kind != "none"
        )
    )
    signal_values = signal(times)
    if field.input_kind == "none" and np.any(signal_values != 0):
        raise ValueError("An unforced model cannot receive a nonzero input.")
    observations, sampled_times = [initial.copy()], [float(times[0])]
    diagnostics = {
        "evaluations": 0,
        "solver": "DOP853",
        "rtol": float(rtol),
        "atol": np.asarray(atol).tolist(),
    }

    def guard(state):
        if not np.isfinite(state).all():
            raise RuntimeError("Solver state became nonfinite.")
        if np.max(np.abs(state)) > max_state:
            raise RuntimeError(f"Solver state exceeded max_state={max_state:g}.")

    held_value = None

    def rhs(now, state):
        diagnostics["rhs_time"] = float(now)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Simulation deadline reached.")
        if diagnostics["evaluations"] >= max_evaluations:
            raise RuntimeError("Simulation reached max_evaluations.")
        diagnostics["evaluations"] += 1
        guard(state)
        rate = np.asarray(
            field.numpy_rhs(state, held_value if signal.is_held else signal(now)),
            dtype=float,
        )
        if rate.shape != initial.shape or not np.isfinite(rate).all():
            raise RuntimeError("The fitted field returned an invalid derivative.")
        return rate

    modes = [(module, module.training) for module in field.modules()]
    try:
        field.eval()
        guard(initial)
        if signal.is_held:
            knots = signal.t[(signal.t > times[0]) & (signal.t < times[-1])]
            boundaries = np.r_[times[0], knots, times[-1]]
        else:
            boundaries = np.array([times[0], times[-1]])
        index, state = 1, initial
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            # Freeze the known left-interval input even at its final stage.
            # Every jump starts a new solver so no narrow held pulse is skipped.
            if signal.is_held:
                held_value = signal(left)
            solver = DOP853(rhs, left, state, right, rtol=rtol, atol=atol)
            while solver.status == "running":
                message = solver.step()
                if solver.status == "failed":
                    raise RuntimeError(
                        message or "Solver failed to complete an integration step."
                    )
                guard(solver.y)
                dense = solver.dense_output()
                while index < len(times) and times[index] <= solver.t:
                    value = np.asarray(dense(times[index]), dtype=float)
                    guard(value)
                    observations.append(value.copy())
                    sampled_times.append(float(times[index]))
                    index += 1
            state = solver.y.copy()
        if index != len(times):
            raise RuntimeError(
                "Solver terminated before all requested observations were available."
            )
        return np.stack(observations)
    except Exception as error:
        diagnostics["cause"] = str(error)
        diagnostics["cause_type"] = type(error).__name__
        raise SimulationError(
            f"Simulation failed: {error}",
            t=np.asarray(sampled_times),
            x=np.stack(observations),
            diagnostics=diagnostics,
        ) from error
    finally:
        for module, was_training in modes:
            module.training = was_training
