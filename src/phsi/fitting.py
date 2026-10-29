"""
Fit models and select which analytic terms to retain.

Integration-scheme fitting builds on pseudo-Hamiltonian neural networks
(https://github.com/SINTEF/pseudo-hamiltonian-neural-networks) and
pseudo-Hamiltonian system identification
(https://github.com/SINTEF/PHSI/tree/legacy_phsi). Trajectory fitting is informed by
neural Hamiltonian ordinary differential equations
(https://github.com/sunnivameltzer/nhode).
Regression is adapted from sparse identification of port-Hamiltonian systems
(author code not yet public). See docs/attribution.md and
docs/source-provenance.json for attributions to external sources.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .errors import FitError, SimulationError, SupportNonconvergence
from .integrators import held_stage_times, rollout, srk4_derivative
from .persistence import save_field
from .regression import fit_regression
from .simulation import simulate_field

THRESHOLDS = (0.0, 0.0001, 0.001, 0.01, 0.1)
COMPLETE = {"completed", "failed_fit", "failed_validation"}


def _tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


def _check_time(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Fitting allowance exhausted.")


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def _finite_score(score):
    score = float(score)
    if not math.isfinite(score) or score < 0:
        raise FloatingPointError("Validation score must be finite and nonnegative.")
    return score


def _write_json(path, value):
    """
    Save a record without exposing a partially written file.
    """
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(_json_finite(value), handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def analytic_components(field):
    result = {}
    if field.energy.kind == "analytic":
        result["energy"] = field.energy
    if field.damping.kind == "analytic":
        result["damping"] = field.damping
    if field.gain.kind == "analytic":
        result["gain"] = field.gain
    return result


def threshold_grid(field, prune=True):
    """
    List threshold combinations in search order; ties are resolved separately.
    """
    components = list(analytic_components(field))
    if not components:
        return []
    values = THRESHOLDS if prune else (0.0,)
    indices = list(itertools.product(range(len(values)), repeat=len(components)))
    indices.sort(key=lambda pair: (max(pair), sum(pair), pair))
    return [dict(zip(components, (values[i] for i in pair))) for pair in indices]


def set_damping_scales(field, observations):
    """
    Measure damping scales from unsmoothed training data; keep zero scales.
    """
    if field.damping.kind != "analytic":
        return
    n = field.n_dof
    sigma = observations.raw_sigma[n:]
    if not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise ValueError(
            "Analytic damping requires positive raw momentum standard deviations."
        )
    contributions = []
    for trajectory in observations.train:
        if len(trajectory.t) < 3:
            raise ValueError(
                "Damping scaling requires at least three raw training observations."
            )
        velocity = (trajectory.x[2:, :n] - trajectory.x[:-2, :n]) / (
            trajectory.t[2:] - trajectory.t[:-2]
        )[:, None]
        features = (
            field.damping.features(_tensor(trajectory.x[1:-1, n:])).detach().numpy()
        )
        contributions.append(features * velocity[..., None] / sigma[None, :, None])
    scale = np.sqrt(np.mean(np.concatenate(contributions) ** 2, axis=0))
    if not np.all(np.isfinite(scale)):
        raise ValueError("Nonfinite observed-data damping contribution scale.")
    field.damping_scales.copy_(_tensor(scale))


def apply_thresholds(field, thresholds):
    with torch.no_grad():
        for name, module in analytic_components(field).items():
            threshold = thresholds[name]
            if name == "damping":
                magnitude = (
                    module.physical_coefficients() * field.damping_scales.numpy()
                )
            else:
                magnitude = np.abs(module.normalized_coefficients())
            mask = (
                np.ones_like(magnitude, dtype=bool)
                if threshold == 0
                else magnitude >= threshold
            )
            module.mask.copy_(_tensor(mask))
    return support_record(field)


def support_record(field):
    masks = {
        name: module.mask.detach().cpu().bool().tolist()
        for name, module in analytic_components(field).items()
    }
    return dict(
        masks=masks,
        retained={name: int(np.asarray(mask).sum()) for name, mask in masks.items()},
        empty_energy_damping_unidentified=bool(
            "energy" in masks and not np.asarray(masks["energy"]).any()
        ),
    )


def _validation(field, observations, method, substeps, deadline=None):
    modes = [(module, module.training) for module in field.modules()]
    field.eval()
    try:
        return _validation_eval(field, observations, method, substeps, deadline)
    finally:
        for module, training in modes:
            module.training = training


def _validation_eval(field, observations, method, substeps, deadline=None):
    total, count = 0.0, 0
    for trajectory in observations.validation:
        _check_time(deadline)
        if method in ("derivative", "midpoint"):
            prediction = simulate_field(
                field,
                trajectory.x[0],
                trajectory.t,
                trajectory.signal,
                deadline=deadline,
                max_state=1e3,
                max_evaluations=150000,
            )
        else:
            x = trajectory.x[0].copy()
            prediction = [x.copy()]

            def f(state, moment):
                value = field.numpy_rhs(state, trajectory.signal(moment))
                if not np.isfinite(value).all():
                    raise FloatingPointError("Nonfinite validation vector field.")
                return value

            for left, right in zip(trajectory.t[:-1], trajectory.t[1:]):
                _check_time(deadline)
                h = (right - left) / substeps
                for substep in range(substeps):
                    moment = left + substep * h
                    first, last = (
                        held_stage_times(moment, h)
                        if trajectory.signal.is_held
                        else (moment, moment + h)
                    )
                    k1 = f(x, first)
                    k2 = f(x + h * k1 / 2, moment + h / 2)
                    k3 = f(x + h * k2 / 2, moment + h / 2)
                    k4 = f(x + h * k3, last)
                    x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1e6:
                        raise FloatingPointError("Validation trajectory diverged.")
                prediction.append(x.copy())
            prediction = np.asarray(prediction)
        residual = (prediction - trajectory.x) / observations.sigma
        if not np.isfinite(residual).all():
            raise FloatingPointError(
                "Validation must complete with finite predictions."
            )
        with np.errstate(over="raise", invalid="raise"):
            total += float(np.sum(residual**2))
        count += residual.size
    _check_time(deadline)
    return _finite_score(math.sqrt(total / count))


def _training_loss(
    field,
    trajectories,
    rng,
    method,
    sigma,
    batch_size,
    window_batch,
    window_steps,
    n_substeps,
):
    lengths = np.array(
        [
            len(tr.x) - (window_steps if method == "simulation_error" else 1)
            for tr in trajectories
        ]
    )
    size = window_batch if method == "simulation_error" else batch_size
    offsets = np.r_[0, np.cumsum(lengths)]
    sampled = rng.integers(0, offsets[-1], size=size)
    which = np.searchsorted(offsets[1:], sampled, side="right")
    total = None
    for index, trajectory in enumerate(trajectories):
        selection = sampled[which == index] - offsets[index]
        if not len(selection):
            continue
        h = float(trajectory.t[1] - trajectory.t[0])
        times = _tensor(trajectory.t[selection])

        def f(x, t):
            return field(x, trajectory.signal(t), create_graph=True)

        if method == "integration_scheme":
            first, second = (
                _tensor(trajectory.x[selection]),
                _tensor(trajectory.x[selection + 1]),
            )
            derivative = srk4_derivative(f, first, second, times, h)
            residual = ((second - first) / h - derivative) / sigma
        else:
            windows = _tensor(
                trajectory.x[selection[:, None] + np.arange(window_steps + 1)]
            )
            prediction = rollout(
                f,
                windows[:, 0],
                times,
                h,
                window_steps,
                n_substeps,
                endpoint_left=trajectory.signal.is_held,
            )
            residual = (prediction[:, 1:] - windows[:, 1:]) / sigma
        part = residual.square().mean() * (len(selection) / size)
        total = part if total is None else total + part
    return total


def _adam_stage(
    field,
    observations,
    *,
    name,
    thresholds,
    steps,
    method,
    settings,
    started,
    allowance,
    directory,
    terminal_validation,
    progress=False,
):
    deadline = None if allowance is None else started + allowance
    work_deadline = None if deadline is None else deadline - 5.0
    history, best_state, best_score = [], None, float("inf")
    record = dict(
        name=name,
        thresholds=thresholds,
        status="running",
        stop_reason=None,
        selection_eligible=False,
        updates=0,
        nonfinite_attempts=0,
        history=history,
        **support_record(field),
    )
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng([settings["seed"], 11])
    field.train()
    optimizer = torch.optim.Adam(field.parameters(), lr=settings["learning_rate"])
    last_loss = None
    streak = 0
    bar = tqdm(
        total=steps,
        desc="Full-library fitting" if name == "dense" else f"Refit {name}",
        unit="update",
        disable=not progress,
        position=0 if name == "dense" else 1,
        leave=name == "dense",
    )

    def show_status(status, *, refresh=False):
        if last_loss is not None:
            status += f", loss={last_loss:.4g}"
        bar.set_postfix_str(status, refresh=refresh)

    def validate():
        nonlocal best_state, best_score
        _check_time(work_deadline)
        check = dict(
            update=record["updates"],
            loss=last_loss,
            score=None,
            validation_finished_s=None,
            saved_s=None,
            selection_eligible=False,
        )
        history.append(check)
        show_status("Validating", refresh=True)
        try:
            score = _finite_score(
                _validation(
                    field,
                    observations,
                    method,
                    settings["validation_substeps"],
                    deadline,
                )
            )
            check.update(score=score, validation_finished_s=time.monotonic() - started)
            if score < best_score:
                state = copy.deepcopy(field.state_dict())
                if directory is not None:
                    checkpoint = directory / f"checkpoint-{len(history) - 1:06d}.pt"
                    digest = save_field(
                        field,
                        checkpoint,
                        fit_info=dict(stage=name, validation=check.copy()),
                    )
                    check.update(checkpoint=str(checkpoint), sha256=digest)
                check["saved_s"] = time.monotonic() - started
                check["selection_eligible"] = (
                    deadline is None or time.monotonic() <= deadline
                )
                if directory is not None:
                    _write_json(directory / "history.json", history)
                    if deadline is not None and time.monotonic() > deadline:
                        check["selection_eligible"] = False
                        _write_json(directory / "history.json", history)
                if check["selection_eligible"]:
                    best_state, best_score = state, score
                    record.update(
                        selection_eligible=True,
                        score=score,
                        best_update=record["updates"],
                        checkpoint=check.get("checkpoint"),
                    )
        except (SimulationError, FloatingPointError) as error:
            check["failure"] = f"{type(error).__name__}: {error}"
        finally:
            if directory is not None:
                _write_json(directory / "history.json", history)
            show_status("Fitting")

    try:
        validate()
        while record["updates"] < steps:
            _check_time(work_deadline)
            optimizer.zero_grad(set_to_none=True)
            loss = _training_loss(
                field,
                observations.training,
                rng,
                method,
                field.sigma,
                settings["batch_size"],
                settings["window_batch"],
                settings["window_steps"],
                settings["n_substeps"],
            )
            if not bool(torch.isfinite(loss)):
                record["nonfinite_attempts"] += 1
                streak += 1
                if streak >= settings["nonfinite_limit"]:
                    raise FloatingPointError(
                        "Consecutive nonfinite training-loss limit reached."
                    )
                continue
            streak = 0
            loss.backward()
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in field.parameters()
            ):
                raise FloatingPointError("Nonfinite parameter gradient.")
            optimizer.step()
            record["updates"] += 1
            last_loss = float(loss.detach())
            show_status("Fitting")
            bar.update(1)
            if record["updates"] % settings["validation_every"] == 0:
                validate()
        if terminal_validation and history[-1]["update"] != record["updates"]:
            validate()
        record.update(
            status="completed" if best_state is not None else "failed_validation",
            stop_reason="update_limit",
        )
    except TimeoutError:
        record.update(
            status="completed" if best_state is not None else "timed_out",
            stop_reason="time_limit",
        )
    except (FloatingPointError, RuntimeError, ValueError) as error:
        record.update(
            status="failed_fit",
            stop_reason="numerical_failure",
            failure=f"{type(error).__name__}: {error}",
        )
    finally:
        show_status(record["stop_reason"] or record["status"])
        bar.close()
    record.update(elapsed_s=time.monotonic() - started, final_loss=last_loss)
    if best_state is not None:
        field.load_state_dict(best_state, strict=True)
    if directory is not None:
        _write_json(directory / "receipt.json", record)
    return field, record


def select_record(records, baseline=None):
    """
    Choose by validation score only after the candidate search finishes.
    """
    if any(record["status"] not in COMPLETE for record in records):
        return None
    eligible = [
        record
        for record in records
        if record.get("selection_eligible")
        and math.isfinite(record.get("score", float("inf")))
    ]
    if (
        baseline is not None
        and baseline.get("selection_eligible")
        and math.isfinite(baseline.get("score", float("inf")))
    ):
        eligible.append(baseline)

    def priority(record):
        thresholds = record.get("thresholds", {})
        order = tuple(
            thresholds[name]
            for name in ("energy", "damping", "gain")
            if name in thresholds
        )
        rank = (
            0
            if record is baseline
            else (1 if all(value == 0 for value in order) else 2)
        )
        return record["score"], rank, order

    return min(eligible, key=priority) if eligible else None


def fit_field(
    field,
    observations,
    *,
    method="integration_scheme",
    max_steps=1000,
    refit_steps=200,
    learning_rate=0.001,
    validation_every=100,
    batch_size=256,
    window_batch=32,
    window_steps=10,
    n_substeps=2,
    validation_substeps=4,
    seed=0,
    prune=True,
    time_limit=None,
    refit_time_limit=None,
    output_dir=None,
    nonfinite_limit=50,
    terminal_validation=True,
    refit_terminal_validation=True,
    started=None,
    progress=False,
):
    """
    Fit models, try removing small analytic terms, and select using validation data.
    """
    started = time.monotonic() if started is None else started
    methods = ("integration_scheme", "simulation_error", "derivative", "midpoint")
    if method not in methods:
        raise ValueError(f"method must be one of {methods}")
    if method in ("derivative", "midpoint") and any(
        module.kind == "nn" for module in (field.energy, field.damping, field.gain)
    ):
        raise ValueError("Joint regression supports analytic components only.")
    settings = dict(
        method=method,
        max_steps=max_steps,
        refit_steps=refit_steps,
        learning_rate=learning_rate,
        validation_every=validation_every,
        batch_size=batch_size,
        window_batch=window_batch,
        window_steps=window_steps,
        n_substeps=n_substeps,
        validation_substeps=validation_substeps,
        seed=seed,
        nonfinite_limit=nonfinite_limit,
        prune=bool(prune),
        time_limit=time_limit,
        refit_time_limit=refit_time_limit,
        terminal_validation=terminal_validation,
        refit_terminal_validation=refit_terminal_validation,
    )
    for name in (
        "max_steps",
        "refit_steps",
        "validation_every",
        "batch_size",
        "window_batch",
        "window_steps",
        "n_substeps",
        "validation_substeps",
        "nonfinite_limit",
    ):
        value = settings[name]
        if not isinstance(value, (int, np.integer)) or value < (
            0 if name in ("max_steps", "refit_steps") else 1
        ):
            raise ValueError(
                f"{name} must be an integer in its allowed positive range."
            )
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    for value in (time_limit, refit_time_limit):
        if value is not None and (not np.isfinite(value) or value <= 5):
            raise ValueError(
                "Timed allowances must exceed the five-second finalization reserve."
            )
    if (
        any(value is not None for value in (time_limit, refit_time_limit))
        and output_dir is None
    ):
        raise ValueError("Timed fitting requires output_dir for durable checkpoints.")
    if method in ("derivative", "midpoint") and refit_time_limit is not None:
        raise ValueError(
            "Regression uses time_limit for the entire grid, not a per-pair refit limit."
        )
    directory = None if output_dir is None else Path(output_dir)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    info = dict(
        status="running",
        method=method,
        settings=settings,
        split=observations.split_info,
        sigma=observations.sigma.tolist(),
        raw_sigma=observations.raw_sigma.tolist(),
        candidates=[],
        selected=None,
        grid_complete=False,
    )
    states = {}
    try:
        set_damping_scales(field, observations)
        if method in ("integration_scheme", "simulation_error"):
            dense, baseline = _adam_stage(
                field,
                observations,
                name="dense",
                thresholds={},
                steps=max_steps,
                method=method,
                settings=settings,
                started=started,
                allowance=time_limit,
                directory=None if directory is None else directory / "dense",
                terminal_validation=terminal_validation,
                progress=progress,
            )
            info["dense"] = baseline
            states["dense"] = copy.deepcopy(dense.state_dict())
            if not baseline["selection_eligible"]:
                raise FitError(
                    "Full-library fitting produced no eligible validation checkpoint.",
                    fit_info=info,
                    partial_model=dense,
                )
            grid = threshold_grid(dense, prune=True) if prune else []
            if grid:
                candidates_bar = tqdm(
                    total=len(grid),
                    desc="Pruning candidates",
                    unit="candidate",
                    disable=not progress,
                    position=0,
                )
                try:
                    for index, thresholds in enumerate(grid):
                        candidate_started = time.monotonic()
                        candidates_bar.set_postfix_str(
                            f"Refitting {index + 1}/{len(grid)}"
                        )
                        candidate = copy.deepcopy(dense)
                        apply_thresholds(candidate, thresholds)
                        name = f"candidate-{index:02d}"
                        candidate, record = _adam_stage(
                            candidate,
                            observations,
                            name=name,
                            thresholds=thresholds,
                            steps=refit_steps,
                            method=method,
                            settings=settings,
                            started=candidate_started,
                            allowance=refit_time_limit,
                            directory=None if directory is None else directory / name,
                            terminal_validation=refit_terminal_validation,
                            progress=progress,
                        )
                        info["candidates"].append(record)
                        if (
                            all(value == 0 for value in thresholds.values())
                            and record["history"]
                        ):
                            initial = record["history"][0].get("score")
                            info["compatibility"] = dict(
                                dense_score=baseline["score"],
                                zero_initial_score=initial,
                                exact_match=initial == baseline["score"],
                            )
                            if initial != baseline["score"]:
                                record.update(
                                    status="timed_out"
                                    if record["status"] == "timed_out"
                                    else "compatibility_hold",
                                    selection_eligible=False,
                                    failure="Unchanged zero-threshold validation differs from dense checkpoint.",
                                )
                        if record["selection_eligible"]:
                            states[name] = copy.deepcopy(candidate.state_dict())
                        if record["status"] in COMPLETE:
                            candidates_bar.update(1)
                        candidates_bar.set_postfix_str(record["status"], refresh=False)
                finally:
                    candidates_bar.close()
            selected = select_record(info["candidates"], baseline)
            result = dense
        else:
            baseline = None
            deadline = None if time_limit is None else started + time_limit
            work_deadline = None if deadline is None else deadline - 5
            grid = threshold_grid(field, prune=prune)
            info["candidates"] = [
                dict(
                    name=f"candidate-{index:02d}",
                    thresholds=thresholds,
                    status="not_run",
                    selection_eligible=False,
                    history=[],
                )
                for index, thresholds in enumerate(grid)
            ]
            result = copy.deepcopy(field)
            candidates_bar = tqdm(
                total=len(grid),
                desc="Regression candidates",
                unit="candidate",
                disable=not progress,
                position=0,
            )
            try:
                for record in info["candidates"]:
                    try:
                        candidates_bar.set_postfix_str(f"Fitting {record['name']}")
                        _check_time(work_deadline)
                        candidate, history = fit_regression(
                            field,
                            observations.training,
                            record["thresholds"],
                            method=method,
                            deadline=work_deadline,
                        )
                        record.update(history=history, **support_record(candidate))
                        candidates_bar.set_postfix_str(f"Validating {record['name']}")
                        score = _finite_score(
                            _validation(
                                candidate,
                                observations,
                                method,
                                validation_substeps,
                                deadline,
                            )
                        )
                        record.update(
                            score=score,
                            validation_finished_s=time.monotonic() - started,
                        )
                        if directory is not None:
                            folder = directory / record["name"]
                            folder.mkdir()
                            checkpoint = folder / "model.pt"
                            digest = save_field(
                                candidate,
                                checkpoint,
                                fit_info=dict(stage=record["name"], score=score),
                            )
                            record.update(checkpoint=str(checkpoint), sha256=digest)
                        record["saved_s"] = time.monotonic() - started
                        record["selection_eligible"] = (
                            deadline is None or time.monotonic() <= deadline
                        )
                        record["status"] = (
                            "completed" if record["selection_eligible"] else "timed_out"
                        )
                        if directory is not None:
                            _write_json(
                                directory / "candidates.json", info["candidates"]
                            )
                            if deadline is not None and time.monotonic() > deadline:
                                record.update(
                                    selection_eligible=False, status="timed_out"
                                )
                        if record["selection_eligible"]:
                            states[record["name"]] = copy.deepcopy(
                                candidate.state_dict()
                            )
                    except TimeoutError as error:
                        record.update(status="timed_out", failure=str(error))
                        break
                    except SupportNonconvergence as error:
                        record.update(
                            status="failed_fit",
                            failure=str(error),
                            history=getattr(error, "history", []),
                        )
                    except SimulationError as error:
                        timed_out = (
                            error.diagnostics.get("cause_type") == "TimeoutError"
                        )
                        record.update(
                            status="timed_out" if timed_out else "failed_validation",
                            selection_eligible=False,
                            failure=f"{type(error).__name__}: {error}",
                            solver_diagnostics=error.diagnostics,
                        )
                        if timed_out:
                            break
                    except FloatingPointError as error:
                        record.update(
                            status="failed_validation",
                            selection_eligible=False,
                            failure=f"{type(error).__name__}: {error}",
                        )
                    except RuntimeError as error:
                        record.update(
                            status="failed_fit",
                            failure=f"{type(error).__name__}: {error}",
                        )
                    finally:
                        if record["status"] in COMPLETE:
                            candidates_bar.update(1)
                        candidates_bar.set_postfix_str(record["status"], refresh=False)
            finally:
                candidates_bar.close()
            selected = select_record(info["candidates"])
        info["grid_complete"] = all(
            row["status"] in COMPLETE for row in info["candidates"]
        )
        if selected is None:
            info["status"] = "incomplete" if not info["grid_complete"] else "failed"
            raise FitError(
                "No completed, eligible model selection; inspect fit_info.",
                fit_info=info,
                partial_model=result,
            )
        result.load_state_dict(states[selected["name"]], strict=True)
        info.update(
            status="completed",
            selected={
                key: selected[key]
                for key in ("name", "score", "thresholds", "checkpoint")
                if key in selected
            },
            elapsed_s=time.monotonic() - started,
            **support_record(result),
        )
        if directory is not None:
            _write_json(directory / "fit.json", info)
        if (
            method in ("derivative", "midpoint")
            and deadline is not None
            and time.monotonic() > deadline
        ):
            info.update(
                status="incomplete",
                selected=None,
                stop_reason="finalization_time_limit",
            )
            raise FitError(
                "Regression finalization exceeded its whole-procedure allowance.",
                fit_info=info,
                partial_model=result,
            )
        return result, info
    except Exception as error:
        if info["status"] == "running":
            info["status"] = "incomplete"
        info.update(
            elapsed_s=time.monotonic() - started,
            failure=f"{type(error).__name__}: {error}",
        )
        if directory is not None:
            _write_json(directory / "fit.json", info)
        if isinstance(error, FitError):
            raise
        raise FitError(
            "Fitting stopped; partial evidence is available in fit_info.",
            fit_info=info,
            partial_model=field,
        ) from error
