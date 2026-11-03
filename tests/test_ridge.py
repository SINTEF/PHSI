"""
Small correctness checks for mean-loss ridge and validation-only selection.
"""

import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import phsi
from phsi.data import InputSignal, Trajectory, prepare_data
from phsi.errors import FitError
from phsi.fields import build_field
from phsi.fitting import fit_field, select_record, threshold_grid
from phsi.libraries import DampingLibrary, Polynomial
from phsi.persistence import read_field
from phsi.regression import bounded_lstsq, build_design, fit_regression

DEFAULT_STRENGTHS = [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-2]


def fixture(components=1):
    t = np.linspace(0.0, 1.0, 12)
    x = np.column_stack((np.cos(t + 0.2), -np.sin(t + 0.2)))
    u = np.sin(t)
    data = prepare_data(x, t, u, validation=(x, t, u), forced=True)
    field = build_field(
        Polynomial(2),
        DampingLibrary("diagnostic", widths=[0.5]) if components == 3 else "constant",
        Polynomial(1) if components >= 2 else "known",
        sigma=data.sigma,
        x_train=x,
        u_train=u,
        seed=7,
    )
    return field, data


def fitted_copy(candidate, *args, **kwargs):
    return copy.deepcopy(candidate), []


def test_ridge_matches_known_mean_loss_solution_and_active_damping_bound():
    A = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]])
    b = A @ np.array([1.0, -0.2, 0.3])
    strength = 0.1
    expected = np.linalg.solve(A.T @ A + len(b) * strength * np.eye(3), A.T @ b)
    actual, result = bounded_lstsq(A, b, 3, ridge=strength)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
    assert result.success
    actual, _ = bounded_lstsq(np.diag([2.0, 4.0, 8.0]), [2, 2, -1], 1, ridge=0.1)
    np.testing.assert_allclose(actual, [4 / 4.3, 8 / 16.3, 0], atol=1e-12)


def test_ridge_is_invariant_when_every_data_row_is_duplicated():
    A = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]])
    b = A @ np.array([1.0, -0.2, 0.3])
    first, _ = bounded_lstsq(A, b, 2, ridge=0.01)
    repeated, _ = bounded_lstsq(np.tile(A, (3, 1)), np.tile(b, 3), 2, ridge=0.01)
    np.testing.assert_allclose(first, repeated, rtol=1e-11, atol=1e-13)


def test_zero_ridge_preserves_unaugmented_solver_empty_support_and_roundoff():
    A, b = np.eye(3), np.array([-2.0, -1.0, 3.0])
    original, _ = bounded_lstsq(A, b, 1)
    explicit, _ = bounded_lstsq(A, b, 1, ridge=0.0)
    np.testing.assert_array_equal(explicit, original)
    tiny_negative = SimpleNamespace(x=np.array([-2.0, -1e-20, 0.5]), success=True)
    with patch("phsi.regression.lsq_linear", return_value=tiny_negative) as solve:
        actual, raw = bounded_lstsq(A, b, 1, ridge=0.0)
    np.testing.assert_array_equal(solve.call_args.args[0], A)
    np.testing.assert_array_equal(solve.call_args.args[1], b)
    np.testing.assert_array_equal(actual, [-2, 0, 0.5])
    assert raw.x[1] == -1e-20
    for strength in (0.0, 0.1):
        actual, result = bounded_lstsq(np.empty((3, 0)), b, 0, ridge=strength)
        assert actual.shape == (0,)
        assert result.success


@pytest.mark.parametrize(
    "damping", ["constant", DampingLibrary("diagnostic", widths=[0.5])]
)
def test_ridge_penalizes_existing_coefficient_coordinates_including_damping(damping):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(30, 2))
    field = build_field(
        Polynomial(2),
        damping,
        "none",
        sigma=np.array([2.0, 3.0]),
        x_train=x,
        u_train=np.zeros(len(x)),
    )
    if field.damping.kind == "analytic":
        field.damping_scales.copy_(torch.tensor([[2.0, 3.0, 4.0]]))
    rates = rng.normal(size=x.shape)
    rates[:, 1] = -0.7 * rates[:, 0] + 0.1 * rates[:, 1]
    t = np.arange(len(x)) * 0.1
    trajectory = Trajectory(x, t, InputSignal(t, np.zeros(len(t))), rates)
    A, b, signed = build_design(field, x, rates, np.zeros(len(x)))
    strength = 0.2
    expected, _ = bounded_lstsq(A, b, signed, ridge=strength)
    assert np.any(expected[signed:] > 0)
    fitted, history = fit_regression(field, [trajectory], {}, ridge=strength)
    actual = [fitted.energy.normalized_coefficients()]
    damping_coefficients = fitted.damping.physical_coefficients()
    if fitted.damping.kind == "analytic":
        damping_coefficients = damping_coefficients * fitted.damping_scales.numpy()
    actual.append(np.asarray(damping_coefficients).ravel())
    np.testing.assert_allclose(np.concatenate(actual), expected, atol=1e-12)
    assert history[0]["objective"] == pytest.approx(np.mean((A @ expected - b) ** 2))
    if "ridge_penalty" in history[0]:
        assert history[0]["ridge_penalty"] == pytest.approx(
            strength * (expected @ expected)
        )
        assert history[0]["penalized_objective"] == pytest.approx(
            history[0]["objective"] + history[0]["ridge_penalty"]
        )


def test_every_pruning_refit_keeps_ridge_and_original_combined_scalar_row_count():
    field, data = fixture()
    columns = len(field.energy.coef) + 1
    # Two trajectories with differing row counts; these are scalar equations.
    designs = [
        (np.eye(columns), np.r_[2.0, np.full(columns - 2, 0.001), 0.5], columns - 1),
        (np.zeros((2, columns)), np.zeros(2), columns - 1),
    ]
    calls = []
    actual_solver = bounded_lstsq

    def solve(A, b, signed, ridge=0.0):
        calls.append((A.copy(), b.copy(), ridge))
        return actual_solver(A, b, signed, ridge=ridge)

    with (
        patch("phsi.regression.build_design", side_effect=designs),
        patch("phsi.regression.bounded_lstsq", side_effect=solve),
    ):
        fitted, history = fit_regression(
            field, [data.training[0], data.training[0]], {"energy": 0.01}, ridge=0.1
        )
    assert len(calls) == len(history) == 2
    assert [(len(b), ridge) for _, b, ridge in calls] == [(columns + 2, 0.1)] * 2
    assert [A.shape[1] for A, _, _ in calls] == [columns, 2]
    expected = 2.0 / (1 + (columns + 2) * 0.1)
    assert fitted.energy.normalized_coefficients()[0] == pytest.approx(expected)
    assert fitted.damping.physical_coefficients()[0] == pytest.approx(expected / 4)
    assert np.count_nonzero(fitted.energy.mask.numpy()) == 1


@pytest.mark.parametrize("components,count", [(1, 30), (2, 150), (3, 750)])
def test_default_search_is_independent_strength_by_component_threshold_product(
    components, count
):
    field, data = fixture(components)
    snapshots, strengths = [], []

    def regression(candidate, *args, ridge, **kwargs):
        snapshots.append(copy.deepcopy(candidate.state_dict()))
        strengths.append(ridge)
        fitted = copy.deepcopy(candidate)
        fitted.energy.mask.zero_()
        return fitted, []

    with (
        patch("phsi.fitting.fit_regression", side_effect=regression),
        patch("phsi.fitting._validation", return_value=1.0),
    ):
        _, info = fit_field(field, data, method="midpoint")
    assert len(info["candidates"]) == count
    assert info["ridge_strengths"] == DEFAULT_STRENGTHS
    expected = {
        (strength, tuple(sorted(threshold.items())))
        for strength in DEFAULT_STRENGTHS
        for threshold in threshold_grid(field)
    }
    assert {
        (row["ridge"], tuple(sorted(row["thresholds"].items())))
        for row in info["candidates"]
    } == expected
    assert strengths == [row["ridge"] for row in info["candidates"]]
    for snapshot in snapshots:
        for key, value in snapshot.items():
            torch.testing.assert_close(value, field.state_dict()[key], rtol=0, atol=0)
    assert info["selected"]["ridge"] == 0.0
    assert all(value == 0 for value in info["selected"]["thresholds"].values())
    assert "dense" not in info
    assert info["grid_complete"]


@pytest.mark.parametrize(
    "ridge,prune,count,strengths",
    [
        (None, False, 6, DEFAULT_STRENGTHS),
        (0.1, True, 5, [0.1]),
        ([0.1, 0.0, 0.1, 1e-6], True, 15, [0.0, 1e-6, 0.1]),
    ],
)
def test_fixed_custom_and_unpruned_searches(ridge, prune, count, strengths):
    field, data = fixture()
    with (
        patch("phsi.fitting.fit_regression", side_effect=fitted_copy),
        patch("phsi.fitting._validation", return_value=1.0),
    ):
        _, info = fit_field(field, data, method="derivative", ridge=ridge, prune=prune)
    assert info["ridge_strengths"] == strengths
    assert len(info["candidates"]) == count
    assert info["selected"]["ridge"] == min(strengths)


@pytest.mark.parametrize(
    "ridge", [True, False, -0.1, np.nan, np.inf, [], [0, -1], [True], [[0.1]], "0.1"]
)
def test_public_rejects_invalid_strengths(ridge):
    _, data = fixture()
    tr = data.train[0]
    with pytest.raises(ValueError, match="ridge"):
        phsi.model(Polynomial(2), forcing_gain="none").fit(
            tr.x,
            tr.t,
            validation=(tr.x, tr.t, None),
            method="midpoint",
            ridge=ridge,
            progress=False,
        )


@pytest.mark.parametrize("method", ["integration_scheme", "simulation_error"])
def test_public_adam_methods_reject_explicit_ridge_but_keep_omitted_behavior(method):
    _, data = fixture()
    tr = data.train[0]
    with pytest.raises(ValueError, match="ridge"):
        phsi.model(Polynomial(2), forcing_gain="none").fit(
            tr.x, tr.t, method=method, ridge=0.0, progress=False
        )
    with patch("phsi.fitting._validation", return_value=1.0):
        model = phsi.model(Polynomial(2), forcing_gain="none").fit(
            tr.x,
            tr.t,
            validation=(tr.x, tr.t, None),
            method=method,
            prune=False,
            max_steps=0,
            progress=False,
        )
    assert model.fit_info_["selected"]["name"] == "dense"
    assert model.fit_info_["candidates"] == []


def test_exact_ties_prefer_strength_then_existing_threshold_order():
    def row(name, ridge, thresholds):
        return dict(
            name=name,
            ridge=ridge,
            thresholds=thresholds,
            score=1.0,
            selection_eligible=True,
            status="completed",
        )

    rows = [
        row("dense-positive", 0.1, {"energy": 0.0, "gain": 0.0}),
        row("sparse-zero", 0.0, {"energy": 0.1, "gain": 0.1}),
        row("earlier-zero", 0.0, {"energy": 0.0, "gain": 0.1}),
    ]
    assert select_record(rows)["name"] == "earlier-zero"
    rows.append(row("dense-zero", 0.0, {"energy": 0.0, "gain": 0.0}))
    assert select_record(rows)["name"] == "dense-zero"
    rows[0]["score"] = np.nextafter(1.0, 0.0)
    assert select_record(rows)["name"] == "dense-positive"


@pytest.mark.parametrize(
    "failure", [RuntimeError("fit failed"), FloatingPointError("validation failed")]
)
def test_complete_search_can_fail_without_becoming_incomplete(failure):
    field, data = fixture()
    fitting = isinstance(failure, RuntimeError)
    with (
        patch(
            "phsi.fitting.fit_regression",
            side_effect=failure if fitting else fitted_copy,
        ),
        patch(
            "phsi.fitting._validation",
            side_effect=None if fitting else failure,
            return_value=1.0,
        ),
    ):
        with pytest.raises(FitError) as caught:
            fit_field(field, data, method="midpoint", ridge=[0.0, 0.1], prune=False)
    info = caught.value.fit_info
    assert info["grid_complete"]
    assert info["status"] == "failed"
    assert info["selected"] is None
    assert len(info["candidates"]) == 2
    assert {row["ridge"] for row in info["candidates"]} == {0.0, 0.1}


def test_timeout_after_one_strength_prevents_partial_winner_and_shares_deadline():
    field, data = fixture()
    clock, deadlines = [0.0], []

    def regression(candidate, *args, deadline, **kwargs):
        deadlines.append(deadline)
        if len(deadlines) == 2:
            raise TimeoutError("bounded timeout")
        clock[0] = 2.0
        return copy.deepcopy(candidate), []

    with (
        tempfile.TemporaryDirectory() as directory,
        patch("phsi.fitting.time.monotonic", side_effect=lambda: clock[0]),
        patch("phsi.fitting.fit_regression", side_effect=regression),
        patch("phsi.fitting._validation", return_value=1.0),
    ):
        with pytest.raises(FitError) as caught:
            fit_field(
                field,
                data,
                method="midpoint",
                ridge=[0, 0.1, 0.2],
                prune=False,
                time_limit=10.0,
                started=0.0,
                output_dir=Path(directory) / "run",
            )
    info = caught.value.fit_info
    assert deadlines == [5.0, 5.0]
    assert [row["status"] for row in info["candidates"]] == [
        "completed",
        "timed_out",
        "not_run",
    ]
    assert info["status"] == "incomplete"
    assert info["selected"] is None
    assert not info["grid_complete"]


def test_progress_counts_whole_default_search():
    field, data = fixture()
    bars = []

    class Bar:
        def __init__(self, *, total, **kwargs):
            self.total, self.n, self.closed = total, 0, False
            bars.append(self)

        def update(self, count=1):
            self.n += count

        def set_postfix_str(self, *args, **kwargs):
            pass

        def close(self):
            self.closed = True

    with (
        patch("phsi.fitting.tqdm", Bar),
        patch("phsi.fitting.fit_regression", side_effect=fitted_copy),
        patch("phsi.fitting._validation", return_value=1.0),
    ):
        fit_field(field, data, method="midpoint", prune=False, progress=True)
    assert len(bars) == 1
    assert (bars[0].n, bars[0].total, bars[0].closed) == (6, 6, True)


def test_public_metadata_survives_receipts_candidate_checkpoint_and_model_save():
    _, data = fixture()
    tr = data.train[0]
    with (
        tempfile.TemporaryDirectory() as directory,
        patch("phsi.fitting.fit_regression", side_effect=fitted_copy),
        patch("phsi.fitting._validation", side_effect=[2.0, 1.0]),
    ):
        path = Path(directory)
        model = phsi.model(Polynomial(2), forcing_gain="none").fit(
            tr.x,
            tr.t,
            validation=(tr.x, tr.t, None),
            method="midpoint",
            ridge=[0, 0.1],
            prune=False,
            output_dir=path / "run",
            progress=False,
        )
        info = model.fit_info_
        assert info["ridge_strengths"] == [0.0, 0.1]
        assert info["selected"]["ridge"] == 0.1
        receipt = json.loads((path / "run" / "fit.json").read_text())
        assert receipt["selected"]["ridge"] == 0.1
        assert receipt["ridge_strengths"] == [0.0, 0.1]
        _, checkpoint_info = read_field(info["selected"]["checkpoint"])
        assert checkpoint_info["ridge"] == 0.1
        model.save(path / "model.pt")
        restored = phsi.load(path / "model.pt")
        assert restored.fit_info_ == info


def test_public_omitted_ridge_uses_default_grid_and_preserves_chronological_split():
    _, data = fixture()
    tr = data.train[0]
    with (
        patch("phsi.fitting.fit_regression", side_effect=fitted_copy),
        patch("phsi.fitting._validation", return_value=1.0),
    ):
        model = phsi.model(Polynomial(2), forcing_gain="none").fit(
            tr.x, tr.t, method="derivative", prune=False, progress=False
        )
    assert model.fit_info_["ridge_strengths"] == DEFAULT_STRENGTHS
    assert len(model.fit_info_["candidates"]) == 6
    expected = prepare_data(tr.x, tr.t, forced=False, method="derivative")
    assert model.fit_info_["split"] == expected.split_info


def test_bounded_solution_matches_sum_penalty_formulation():
    # Convert mean-loss ridge to a sum-loss penalty by multiplying the strength
    # by the number of observation equations.
    from scipy.optimize import lsq_linear

    A = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0], [2.0, 2.0, 1.0]])
    b = np.array([1.0, -2.0, 0.2, 1.0])
    strength = 0.1
    reference = lsq_linear(
        np.vstack((A, np.sqrt(len(b) * strength) * np.eye(3))),
        np.r_[b, np.zeros(3)],
        bounds=([-np.inf, 0, 0], [np.inf] * 3),
        method="bvls",
        tol=1e-12,
        max_iter=5000,
    ).x
    actual, _ = bounded_lstsq(A, b, 1, ridge=strength)
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("strength", [True, -1.0, np.nan, np.inf, [0.1]])
def test_low_level_solver_rejects_invalid_scalar_strength(strength):
    with pytest.raises(ValueError, match="ridge"):
        bounded_lstsq(np.eye(2), np.ones(2), 1, ridge=strength)


def test_late_checkpoint_cannot_select_successful_earlier_strength():
    from phsi.persistence import save_field

    field, data = fixture()
    clock, exported = [0.0], []

    def save(candidate, path, *args, **kwargs):
        digest = save_field(candidate, path, *args, **kwargs)
        exported.append(Path(path))
        if len(exported) == 2:
            clock[0] = 11.0
        return digest

    history = [{"iteration": 1, "objective": 0.25}]
    with (
        tempfile.TemporaryDirectory() as directory,
        patch("phsi.fitting.time.monotonic", side_effect=lambda: clock[0]),
        patch(
            "phsi.fitting.fit_regression",
            side_effect=lambda candidate, *a, **k: (copy.deepcopy(candidate), history),
        ),
        patch("phsi.fitting._validation", return_value=1.0),
        patch("phsi.fitting.save_field", side_effect=save),
    ):
        folder = Path(directory) / "run"
        with pytest.raises(FitError) as caught:
            fit_field(
                field,
                data,
                method="midpoint",
                ridge=[0, 0.1],
                prune=False,
                time_limit=10.0,
                started=0.0,
                output_dir=folder,
            )
        info = caught.value.fit_info
        assert info["status"] == "incomplete"
        assert info["selected"] is None
        assert not info["grid_complete"]
        assert info["candidates"][0]["selection_eligible"]
        assert not info["candidates"][1]["selection_eligible"]
        assert info["candidates"][1]["status"] == "timed_out"
        for strength, checkpoint in zip([0.0, 0.1], exported, strict=True):
            _, checkpoint_info = read_field(checkpoint)
            assert checkpoint_info["ridge_strengths"] == [0.0, 0.1]
            assert checkpoint_info["ridge"] == strength
            assert checkpoint_info["thresholds"] == info["candidates"][0]["thresholds"]
            assert checkpoint_info["history"] == history
            assert checkpoint_info["selected"] is None
            assert not checkpoint_info["grid_complete"]
        receipt = json.loads((folder / "fit.json").read_text())
        assert receipt["selected"] is None
        assert receipt["candidates"][1]["ridge"] == 0.1


def test_a_failed_strength_is_terminal_and_does_not_discard_other_strengths():
    field, data = fixture()
    with (
        patch(
            "phsi.fitting.fit_regression",
            side_effect=[RuntimeError("bounded failure"), (field, [])],
        ),
        patch("phsi.fitting._validation", return_value=2.0),
    ):
        _, info = fit_field(field, data, method="midpoint", ridge=[0, 0.1], prune=False)
    assert info["grid_complete"]
    assert info["status"] == "completed"
    assert info["selected"]["ridge"] == 0.1
    assert [row["status"] for row in info["candidates"]] == ["failed_fit", "completed"]


def test_zero_ridge_history_does_not_overflow_norm_of_finite_coefficients():
    field, data = fixture()
    columns = len(field.energy.coef) + 1
    design = (1e-200 * np.eye(columns), np.ones(columns), columns - 1)
    with patch("phsi.regression.build_design", return_value=design):
        fitted, history = fit_regression(field, [data.training[0]], {}, ridge=0.0)
    assert np.isfinite(fitted.energy.normalized_coefficients()).all()
    assert history[0]["objective"] == pytest.approx(0.0, abs=1e-28)
    assert history[0]["ridge_penalty"] == 0.0
    assert history[0]["penalized_objective"] == history[0]["objective"]
