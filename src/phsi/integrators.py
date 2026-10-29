"""
Differentiable Runge-Kutta formulas adapted from pseudo-Hamiltonian neural networks.

Source code: https://github.com/SINTEF/pseudo-hamiltonian-neural-networks.
Gradients pass through every integration stage. Applied signals are
evaluated at each stage. See docs/attribution.md and docs/source-provenance.json
for attributions to external sources.
"""

import math

import numpy as np
import torch

SQ3_6 = math.sqrt(3.0) / 6.0


def _state_step(step, x):
    """
    Reshape step sizes to apply one per trajectory across its state coordinates.
    """
    step = torch.as_tensor(step, dtype=x.dtype, device=x.device)
    return step[..., None] if step.ndim == x.ndim - 1 and step.ndim > 0 else step


def srk4_derivative(f, x1, x2, t1, dt):
    """
    Estimate derivatives from consecutive observations (symmetric fourth order).
    """
    dt = torch.as_tensor(dt, dtype=x1.dtype, device=x1.device)
    t1 = torch.as_tensor(t1, dtype=x1.dtype, device=x1.device)
    step = _state_step(dt, x1)
    midpoint = 0.5 * (x1 + x2)
    z1 = (0.5 + SQ3_6) * x1 + (0.5 - SQ3_6) * x2
    z2 = (0.5 - SQ3_6) * x1 + (0.5 + SQ3_6) * x2
    tm, tp = t1 + (0.5 - SQ3_6) * dt, t1 + (0.5 + SQ3_6) * dt
    z3 = midpoint - SQ3_6 * step * f(z2, tp)
    z4 = midpoint + SQ3_6 * step * f(z1, tm)
    return 0.5 * (f(z3, tm) + f(z4, tp))


def held_stage_times(t, h):
    """
    Choose evaluation times just inside a constant-input interval.

    Small offsets avoid sampling the wrong side of an input jump due to rounding.
    Use only for samples held until the next sample, not signal functions.
    """
    if torch.is_tensor(t) or torch.is_tensor(h):
        template = t if torch.is_tensor(t) else h
        t = torch.as_tensor(t, dtype=template.dtype, device=template.device)
        h = torch.as_tensor(h, dtype=template.dtype, device=template.device)
        scale = torch.maximum(t.abs(), (t + h).abs()).clamp(min=1.0)
        margin = torch.minimum(h.abs() / 4, 8 * torch.finfo(template.dtype).eps * scale)
        direction = h.sign()
    else:
        t, h = np.asarray(t, dtype=float), np.asarray(h, dtype=float)
        scale = np.maximum(1.0, np.maximum(np.abs(t), np.abs(t + h)))
        margin = np.minimum(np.abs(h) / 4, 8 * np.finfo(float).eps * scale)
        direction = np.sign(h)
    return t + direction * margin, t + h - direction * margin


def rk4_step(f, x, t, h, *, endpoint_left=False):
    """
    Advance one fourth-order Runge-Kutta step; endpoint_left is for held samples only.
    """
    h = torch.as_tensor(h, dtype=x.dtype, device=x.device)
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    step = _state_step(h, x)
    first, last = held_stage_times(t, h) if endpoint_left else (t, t + h)
    k1 = f(x, first)
    k2 = f(x + 0.5 * step * k1, t + 0.5 * h)
    k3 = f(x + 0.5 * step * k2, t + 0.5 * h)
    k4 = f(x + step * k3, last)
    return x + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(
    f, x0, t0, h, n_steps, n_sub=2, detach=False, guard=1e6, *, endpoint_left=False
):
    """
    Simulate trajectories; return shape (batch, observation times, states).

    With detach=True, stop tracking gradients between observation steps. Nonfinite
    states or values exceeding guard leave remaining states as NaN (not a number).
    """
    if (
        isinstance(n_steps, bool)
        or not isinstance(n_steps, int)
        or n_steps < 0
        or isinstance(n_sub, bool)
        or not isinstance(n_sub, int)
        or n_sub < 1
    ):
        raise ValueError(
            "n_steps must be nonnegative and n_sub must be positive integers."
        )
    if x0.ndim != 2 or not torch.isfinite(x0).all():
        raise ValueError("Initial states must be finite with shape (batch, states).")
    hs = torch.as_tensor(h, dtype=x0.dtype, device=x0.device) / n_sub
    t = torch.as_tensor(t0, dtype=x0.dtype, device=x0.device)
    if (
        not torch.isfinite(hs).all()
        or not (hs > 0).all()
        or not torch.isfinite(t).all()
    ):
        raise ValueError("Time and positive step sizes must be finite.")
    if not math.isfinite(guard) or guard <= 0:
        raise ValueError("guard must be finite and positive.")
    x = x0.detach() if detach else x0
    xs = [x]
    origin = t
    observation_step = torch.as_tensor(h, dtype=x0.dtype, device=x0.device)
    for observation in range(n_steps):
        for substep in range(n_sub):
            if endpoint_left:
                t = origin + observation * observation_step + substep * hs
            x = rk4_step(f, x, t, hs, endpoint_left=endpoint_left)
            if not endpoint_left:
                t = t + hs
        if detach:
            x = x.detach()
            if not torch.isfinite(x).all() or x.abs().max() > guard:
                xs += [torch.full_like(x, float("nan"))] * (n_steps - len(xs) + 1)
                break
        xs.append(x)
    return torch.stack(xs, dim=1)
