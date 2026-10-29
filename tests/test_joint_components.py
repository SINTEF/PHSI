"""
Tests for learning energy, damping and forcing gain together.
"""

import itertools
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_public_workflows as workflows
import torch

import phsi
from phsi.data import InputSignal, Trajectory, prepare_data
from phsi.fields import build_field
from phsi.fitting import apply_thresholds, fit_field, select_record, threshold_grid
from phsi.regression import build_design, fit_regression


def joint_field(dimension=6):
    x = np.random.default_rng(731).normal(size=(24, dimension))
    u = np.linspace(-0.4, 0.6, len(x))
    field = build_field(
        phsi.Polynomial(2),
        phsi.DampingLibrary("diagnostic", widths=[0.5]),
        phsi.Polynomial(2),
        sigma=np.linspace(0.8, 1.8, dimension),
        x_train=x,
        u_train=u,
        input_index=dimension - 2 if dimension > 2 else 1,
        input_argument=2 if dimension > 2 else 0,
        seed=31,
    )
    return field, x, u


class JointComponentsTests(unittest.TestCase):
    def test_six_state_joint_design_matches_physical_field(self):
        field, x, u = joint_field()
        field.energy.set_physical(np.linspace(-0.3, 0.4, len(field.energy.coef)))
        field.gain.set_physical(np.array([-0.4, 0.2, -0.1]))
        physical_damping = np.arange(1, 10).reshape(3, 3) * 0.03
        field.damping.set_physical(physical_damping)
        scales = np.linspace(0.3, 2.1, 9).reshape(3, 3)
        field.damping_scales.copy_(torch.tensor(scales))
        rates = field.numpy_rhs(x, u)
        A, target, n_signed = build_design(field, x, rates, u)
        coefficient = np.r_[
            field.energy.normalized_coefficients(),
            field.gain.normalized_coefficients(),
            (physical_damping * scales).ravel(),
        ]
        self.assertEqual(field.input_index, 4)
        self.assertEqual(field.input_argument, 2)
        self.assertEqual(n_signed, len(field.energy.coef) + len(field.gain.coef))
        np.testing.assert_allclose(A @ coefficient, target, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            field(torch.tensor(x), torch.tensor(u)), torch.tensor(rates)
        )

    def test_three_supports_shrink_independently_and_zero_threshold_resets(self):
        field, x, u = joint_field(2)
        K, L = len(field.energy.coef), len(field.gain.coef)
        field.damping_scales.copy_(torch.tensor([[2.0, 0.0, 4.0]]))
        for component in (field.energy, field.gain, field.damping):
            component.mask.zero_()
        times = np.arange(len(x)) * 0.1
        trajectory = Trajectory(x, times, InputSignal(times, u), np.zeros_like(x))
        calls = []

        def solution(A, target, signed):
            calls.append((A.shape[1], signed))
            result = np.full(A.shape[1], 0.5)
            if len(calls) == 1:
                result[[0, K, K + L]] = [0.05, 0.03, 0.02]
            return result, SimpleNamespace(success=True)

        with patch("phsi.regression.bounded_lstsq", side_effect=solution):
            fitted, history = fit_regression(
                field, [trajectory], {"energy": 0.1, "gain": 0.04, "damping": 0.03}
            )
        self.assertEqual(calls, [(K + L + 3, K + L), (K + L - 1, K + L - 2)])
        np.testing.assert_array_equal(
            fitted.energy.mask.numpy(), [False] + [True] * (K - 1)
        )
        np.testing.assert_array_equal(
            fitted.gain.mask.numpy(), [False] + [True] * (L - 1)
        )
        np.testing.assert_array_equal(
            fitted.damping.mask.numpy(), [[False, False, True]]
        )
        np.testing.assert_allclose(
            fitted.damping.physical_coefficients(), [[0.0, 0.0, 0.125]]
        )
        self.assertEqual(len(history), 2)
        for component in (field.energy, field.gain, field.damping):
            self.assertFalse(bool(component.mask.any()))

        def zeros(A, target, signed):
            return np.zeros(A.shape[1]), SimpleNamespace(success=True)

        with patch("phsi.regression.bounded_lstsq", side_effect=zeros):
            reset, history = fit_regression(
                field, [trajectory], {"energy": 0.0, "gain": 0.0, "damping": 0.0}
            )
        for component in (reset.energy, reset.gain, reset.damping):
            self.assertTrue(bool(component.mask.all()))
        self.assertEqual(len(history), 1)
        with patch("phsi.regression.bounded_lstsq", side_effect=zeros):
            empty, history = fit_regression(
                field, [trajectory], {"energy": 1.0, "gain": 1.0, "damping": 1.0}
            )
        for component in (empty.energy, empty.gain, empty.damping):
            self.assertFalse(bool(component.mask.any()))
        np.testing.assert_array_equal(empty.damping.physical_coefficients(), 0.0)
        np.testing.assert_array_equal(empty.numpy_rhs(x, u), np.zeros_like(x))
        self.assertEqual(len(history), 2)

    def test_repeated_three_component_thresholding_has_no_positive_reactivation(self):
        field, x, u = joint_field(2)
        with torch.no_grad():
            field.energy.coef.fill_(0.2)
            field.energy.coef[0] = 0.01
            field.gain.coef.copy_(torch.tensor([-0.02, 0.3, -0.4]))
            field.damping_scales.copy_(torch.tensor([[1.0, 0.0, 2.0]]))
        field.damping.set_physical([[0.01, 0.2, 0.3]])
        apply_thresholds(field, {"energy": 0.1, "gain": 0.1, "damping": 0.1})
        masks = [
            component.mask.clone()
            for component in (field.energy, field.gain, field.damping)
        ]
        apply_thresholds(field, {"energy": 0.001, "gain": 0.001, "damping": 0.001})
        for component, expected in zip(
            (field.energy, field.gain, field.damping), masks
        ):
            torch.testing.assert_close(component.mask, expected)
        apply_thresholds(field, {"energy": 0.0, "gain": 0.0, "damping": 0.0})
        for component in (field.energy, field.gain, field.damping):
            self.assertTrue(bool(component.mask.all()))
        apply_thresholds(field, {"energy": 10.0, "gain": 10.0, "damping": 10.0})
        for component in (field.energy, field.gain, field.damping):
            self.assertFalse(bool(component.mask.any()))
        np.testing.assert_array_equal(field.numpy_rhs(x, u), np.zeros_like(x))

    def test_full_grid_and_complete_selection_tie_rules(self):
        field, _, _ = joint_field(2)
        grid = threshold_grid(field)
        order = ("energy", "damping", "gain")
        tuples = [tuple(candidate[name] for name in order) for candidate in grid]
        values = (0.0, 0.0001, 0.001, 0.01, 0.1)
        self.assertEqual(len(grid), 125)
        self.assertEqual(set(tuples), set(itertools.product(values, repeat=3)))
        self.assertEqual(len(set(tuples)), 125)
        self.assertEqual(
            threshold_grid(field, prune=False), [dict.fromkeys(order, 0.0)]
        )
        records = [
            dict(
                name=str(i),
                thresholds=threshold,
                status="completed",
                score=1.0,
                selection_eligible=True,
            )
            for i, threshold in enumerate(grid)
        ]
        baseline = dict(
            name="dense", status="completed", score=1.0, selection_eligible=True
        )
        self.assertIs(select_record(records, baseline), baseline)
        zero = next(
            record for record in records if not any(record["thresholds"].values())
        )
        self.assertIs(select_record(records), zero)
        zero["score"] = 2.0
        expected = min(
            (record for record in records if record is not zero),
            key=lambda record: tuple(record["thresholds"][name] for name in order),
        )
        self.assertIs(select_record(list(reversed(records))), expected)
        for unfinished in ("not_run", "timed_out"):
            records[-1]["status"] = unfinished
            self.assertIsNone(select_record(records, baseline))

    def test_all_125_mocked_candidates_start_from_same_dense_parameters(self):
        x, t, u, validation = workflows.observations(2)
        data = prepare_data(x, t, u, validation=validation, forced=True)
        field = build_field(
            phsi.Polynomial(2),
            phsi.DampingLibrary("diagnostic", widths=[0.5]),
            phsi.Polynomial(1),
            sigma=data.sigma,
            x_train=x,
            u_train=u,
        )
        snapshots = []

        def stage(candidate, *args, name, thresholds, **kwargs):
            snapshots.append(
                {
                    key: value.detach().clone()
                    for key, value in candidate.named_parameters()
                }
            )
            return candidate, dict(
                name=name,
                thresholds=thresholds,
                status="completed",
                score=1.0,
                selection_eligible=True,
                history=[{"score": 1.0}],
            )

        with patch("phsi.fitting._adam_stage", side_effect=stage):
            _, info = fit_field(field, data, max_steps=0, refit_steps=0)
        self.assertEqual(len(snapshots), 126)
        self.assertEqual(len(info["candidates"]), 125)
        self.assertTrue(info["grid_complete"])
        self.assertEqual(info["selected"]["name"], "dense")
        for snapshot in snapshots[1:]:
            for key, parameter in snapshots[0].items():
                torch.testing.assert_close(snapshot[key], parameter, rtol=0.0, atol=0.0)


class JointPublicWorkflowsTests(unittest.TestCase):
    fit = workflows.PublicWorkflowTests.fit
    assert_roundtrip = workflows.PublicWorkflowTests.assert_roundtrip

    def test_all_four_public_fitters_with_all_three_analytic_components(self):
        with tempfile.TemporaryDirectory() as directory:
            for method in workflows.METHODS:
                with self.subTest(method=method):
                    model = self.fit(
                        6,
                        method,
                        damping=phsi.DampingLibrary("diagnostic", widths=[0.5]),
                        gain=phsi.Polynomial(1),
                        forced_state=4,
                        gain_state=2,
                    )
                    self.assertEqual(model.field_.input_index, 4)
                    self.assertEqual(model.field_.input_argument, 2)
                    np.testing.assert_array_equal(
                        model.input_map(np.zeros(6))[[0, 1, 2, 3, 5]], 0.0
                    )
                    equations = str(model).splitlines()
                    self.assertTrue(
                        any(
                            line.startswith("dp2/dt") and "*u" in line
                            for line in equations
                        )
                    )
                    self.assertFalse(
                        any(
                            line.startswith(("dp1/dt", "dp3/dt")) and "*u" in line
                            for line in equations
                        )
                    )
                    retained = model.retained_terms_
                    for component in ("energy", "damping", "gain"):
                        self.assertTrue(np.asarray(retained[component]["mask"]).all())
                    self.assert_roundtrip(model, directory, "joint-" + method)

    def test_neural_joint_components_use_both_gradient_methods(self):
        def neural():
            return phsi.Neural(hidden=(3,))

        cases = [
            (
                "neural_energy",
                neural(),
                phsi.DampingLibrary("diagnostic", widths=[0.5]),
                phsi.Polynomial(1),
            ),
            ("neural_damping", phsi.Polynomial(2), neural(), phsi.Polynomial(1)),
            (
                "neural_gain",
                phsi.Polynomial(2),
                phsi.DampingLibrary("diagnostic", widths=[0.5]),
                neural(),
            ),
            ("all_neural", neural(), neural(), neural()),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for name, energy, damping, gain in cases:
                for method in workflows.GRADIENT_METHODS:
                    with self.subTest(components=name, method=method):
                        model = self.fit(
                            2, method, energy=energy, damping=damping, gain=gain
                        )
                        self.assert_roundtrip(model, directory, name + "-" + method)


if __name__ == "__main__":
    unittest.main()
