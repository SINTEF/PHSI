import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from phsi.data import prepare_data
from phsi.errors import FitError
from phsi.fields import build_field
from phsi.fitting import (
    _training_loss,
    _validation,
    apply_thresholds,
    fit_field,
    select_record,
    set_damping_scales,
    threshold_grid,
)
from phsi.libraries import DampingLibrary, Polynomial


def fixture(dimension=2, learned=False, damping="constant"):
    t = np.linspace(0, 1, 21)
    x = np.column_stack(
        [np.cos(t + 0.3 * i) for i in range(dimension // 2)]
        + [-np.sin(t + 0.3 * i) for i in range(dimension // 2)]
    )
    u = np.sin(t) if learned else None
    data = prepare_data(x, t, u, validation=(x, t, u), forced=learned)
    field = build_field(
        Polynomial(2),
        damping,
        Polynomial() if learned else "none",
        sigma=data.sigma,
        x_train=x,
        u_train=np.zeros_like(t) if u is None else u,
        input_argument=0 if learned else None,
        seed=7,
    )
    return field, data


class PruningTests(unittest.TestCase):
    def test_independent_grid_and_zero_retains_zero_terms(self):
        field, data = fixture(learned=True)
        grid = threshold_grid(field)
        self.assertEqual(len(grid), 25)
        self.assertIn({"energy": 0.001, "gain": 0.1}, grid)
        with torch.no_grad():
            field.energy.coef.zero_()
            field.gain.coef.zero_()
        record = apply_thresholds(field, {"energy": 0.0, "gain": 0.0})
        self.assertTrue(np.asarray(record["masks"]["energy"]).all())
        record = apply_thresholds(field, {"energy": 0.001, "gain": 0.001})
        self.assertEqual(record["retained"], {"energy": 0, "gain": 0})
        self.assertTrue(record["empty_energy_damping_unidentified"])
        self.assertTrue(np.isfinite(field.numpy_rhs(data.train[0].x, 0)).all())

    def test_scale_from_raw_positions_only_and_no_zero_floor(self):
        field, data = fixture(damping=DampingLibrary())
        data.train[0].x[:, 0] = 1.0
        set_damping_scales(field, data)
        self.assertTrue(np.all(field.damping_scales.numpy() == 0.0))
        apply_thresholds(field, {"energy": 0.0, "damping": 0.0})
        self.assertTrue(bool(field.damping.mask.all()))
        apply_thresholds(field, {"energy": 0.0, "damping": 0.001})
        self.assertFalse(bool(field.damping.mask.any()))
        data.raw_sigma[1] = 0.0
        with self.assertRaises(ValueError):
            set_damping_scales(field, data)

    def test_selection_ties_and_failed_but_eligible_checkpoint(self):
        rows = [
            dict(
                name="sparse",
                status="completed",
                score=0.1,
                thresholds={"energy": 0.1},
                selection_eligible=True,
            ),
            dict(
                name="zero",
                status="failed_fit",
                score=0.1,
                thresholds={"energy": 0.0},
                selection_eligible=True,
            ),
        ]
        baseline = dict(
            name="dense", status="completed", score=0.1, selection_eligible=True
        )
        self.assertIs(select_record(rows, baseline), baseline)
        self.assertIs(select_record(rows), rows[1])
        rows[0]["status"] = "not_run"
        self.assertIsNone(select_record(rows, baseline))
        rows[0]["status"] = "timed_out"
        self.assertIsNone(select_record(rows, baseline))

    def test_adam_candidates_restart_and_masks_do_not_regrow(self):
        field, data = fixture(learned=True)
        snapshots = []

        def loss(candidate, *args, **kwargs):
            snapshots.append(
                {name: value.clone() for name, value in candidate.state_dict().items()}
            )
            return sum((parameter**2).sum() for parameter in candidate.parameters())

        with (
            patch("phsi.fitting._training_loss", side_effect=loss),
            patch("phsi.fitting._validation", return_value=1.0),
        ):
            fitted, info = fit_field(field, data, max_steps=0, refit_steps=1)
        self.assertEqual(len(info["candidates"]), 25)
        self.assertEqual(len(snapshots), 25)
        for snapshot in snapshots:
            self.assertTrue(torch.equal(snapshot["energy.coef"], field.energy.coef))
            self.assertTrue(torch.equal(snapshot["gain.coef"], field.gain.coef))
        self.assertEqual(info["selected"]["name"], "dense")
        self.assertTrue(info["grid_complete"])
        for record in info["candidates"]:
            self.assertEqual([row["update"] for row in record["history"]], [0, 1])

    def test_later_gradient_failure_preserves_earlier_valid_state(self):
        field, data = fixture()
        with (
            patch(
                "phsi.fitting._training_loss",
                side_effect=FloatingPointError("deliberate"),
            ),
            patch("phsi.fitting._validation", return_value=1.0),
        ):
            fitted, info = fit_field(field, data, max_steps=1, prune=False)
        self.assertEqual(info["dense"]["status"], "failed_fit")
        self.assertTrue(info["dense"]["selection_eligible"])
        self.assertEqual(info["selected"]["name"], "dense")

    def test_timed_mode_requires_directory_and_rejects_late_save(self):
        field, data = fixture()
        with self.assertRaises(ValueError):
            fit_field(field, data, time_limit=6.0)
        clock = [0.0]

        def late_save(*args, **kwargs):
            clock[0] = 20.0
            return "hash"

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("phsi.fitting.time.monotonic", side_effect=lambda: clock[0]),
            patch("phsi.fitting._validation", return_value=0.1),
            patch("phsi.fitting.save_field", side_effect=late_save),
        ):
            with self.assertRaises(FitError) as caught:
                fit_field(
                    field,
                    data,
                    max_steps=0,
                    prune=False,
                    started=0.0,
                    time_limit=10.0,
                    output_dir=Path(directory) / "run",
                )
            self.assertFalse(caught.exception.fit_info["dense"]["selection_eligible"])

    def test_regression_whole_grid_and_no_historical_baseline(self):
        field, data = fixture()
        fitted, info = fit_field(field, data, method="midpoint")
        self.assertEqual(len(info["candidates"]), 5)
        self.assertNotIn("dense", info)
        self.assertTrue(info["grid_complete"])
        self.assertEqual(info["status"], "completed")

    def test_both_gradient_losses_differentiate_and_exclude_initial_simulation_state(
        self,
    ):
        field, data = fixture()
        for method in ("integration_scheme", "simulation_error"):
            field.zero_grad()
            loss = _training_loss(
                field,
                data.training,
                np.random.default_rng(3),
                method,
                field.sigma,
                4,
                2,
                3,
                2,
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(field.energy.coef.grad)
            self.assertTrue(torch.isfinite(field.energy.coef.grad).all())


if __name__ == "__main__":
    unittest.main()


class ReviewRegressionTests(unittest.TestCase):
    def test_final_regression_ledger_must_finish_within_deadline(self):
        import phsi.fitting as fitting

        field, data = fixture()
        clock = [0.0]
        original_write = fitting._write_json

        def late_final(path, value):
            original_write(path, value)
            if Path(path).name == "fit.json":
                clock[0] = 11.0

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("phsi.fitting.time.monotonic", side_effect=lambda: clock[0]),
            patch("phsi.fitting.fit_regression", return_value=(field, [])),
            patch("phsi.fitting._validation", return_value=1.0),
            patch("phsi.fitting._write_json", side_effect=late_final),
        ):
            with self.assertRaises(FitError) as caught:
                fit_field(
                    field,
                    data,
                    method="midpoint",
                    time_limit=10.0,
                    started=0.0,
                    output_dir=Path(directory) / "run",
                )
            self.assertEqual(caught.exception.fit_info["status"], "incomplete")
            self.assertIsNone(caught.exception.fit_info["selected"])
            import json

            record = json.loads((Path(directory) / "run" / "fit.json").read_text())
            self.assertEqual(record["status"], "incomplete")

    def test_last_regression_validation_timeout_is_incomplete(self):
        from phsi.errors import SimulationError

        field, data = fixture()
        failure = SimulationError(
            "deadline",
            t=np.array([0.0]),
            x=np.zeros((1, 2)),
            diagnostics={"cause_type": "TimeoutError"},
        )
        with (
            patch("phsi.fitting.fit_regression", return_value=(field, [])),
            patch(
                "phsi.fitting._validation", side_effect=[1.0, 1.0, 1.0, 1.0, failure]
            ),
        ):
            with self.assertRaises(FitError) as caught:
                fit_field(field, data, method="midpoint")
            self.assertEqual(
                caught.exception.fit_info["candidates"][-1]["status"], "timed_out"
            )
            self.assertFalse(caught.exception.fit_info["grid_complete"])

    def test_nonfinite_validation_score_cannot_be_selected_or_saved(self):
        import json

        field, data = fixture()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("phsi.fitting.fit_regression", return_value=(field, [])),
            patch("phsi.fitting._validation", return_value=float("inf")),
        ):
            with self.assertRaises(FitError) as caught:
                fit_field(
                    field, data, method="midpoint", output_dir=Path(directory) / "run"
                )
            self.assertTrue(
                all(
                    not row["selection_eligible"]
                    for row in caught.exception.fit_info["candidates"]
                )
            )
            evidence = json.loads((Path(directory) / "run" / "fit.json").read_text())
            self.assertEqual(evidence["status"], "failed")
        self.assertIsNone(
            select_record(
                [
                    dict(
                        status="completed",
                        selection_eligible=True,
                        score=float("inf"),
                        thresholds={},
                    )
                ]
            )
        )

    def test_validation_rejects_squared_residual_overflow(self):
        field, data = fixture()
        data.validation[0].x[1:, :] = 1e200
        with patch(
            "phsi.fitting.simulate_field",
            return_value=np.zeros_like(data.validation[0].x),
        ):
            with self.assertRaises(FloatingPointError):
                _validation(field, data, "midpoint", 4)

    def test_regression_validation_preserves_frozen_solver_guards(self):
        field, data = fixture()
        with patch(
            "phsi.fitting.simulate_field", return_value=data.validation[0].x.copy()
        ) as simulate:
            _validation(field, data, "derivative", 4)
        self.assertEqual(simulate.call_args.kwargs["max_state"], 1e3)
        self.assertEqual(simulate.call_args.kwargs["max_evaluations"], 150000)

    def test_validation_switches_to_eval_and_restores_mode(self):
        field, data = fixture()
        modes = []

        def observe(x, u):
            modes.append(field.training)
            return np.zeros_like(x)

        field.train()
        with patch.object(field, "numpy_rhs", side_effect=observe):
            _validation(field, data, "integration_scheme", 4)
        self.assertFalse(any(modes))
        self.assertTrue(field.training)


class CompatibilityTests(unittest.TestCase):
    def test_unchanged_zero_candidate_must_match_dense_validation_exactly(self):
        field, data = fixture()
        with patch(
            "phsi.fitting._validation", side_effect=[1.0, 0.9, 1.0, 1.0, 1.0, 1.0]
        ):
            with self.assertRaises(FitError) as caught:
                fit_field(field, data, max_steps=0, refit_steps=0)
        info = caught.exception.fit_info
        self.assertEqual(info["candidates"][0]["status"], "compatibility_hold")
        self.assertEqual(
            info["compatibility"],
            {"dense_score": 1.0, "zero_initial_score": 0.9, "exact_match": False},
        )


class HeldSignalFittingTests(unittest.TestCase):
    def test_fixed_validation_and_training_keep_held_interval_boundaries(self):
        t = np.arange(4.0)
        x = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        u = np.array([0.0, 1.0, 0.0, 0.0])
        data = prepare_data(
            x,
            t,
            u,
            validation=(x, t, u),
            forced=True,
            interpolation="previous",
            method="simulation_error",
            window_steps=1,
        )
        field = build_field(
            Polynomial(2), "constant", "known", sigma=data.sigma, x_train=x, u_train=u
        )
        field.energy.set_physical(np.zeros_like(field.energy.physical_coefficients()))
        field.damping.set_physical(np.zeros(1))
        score = _validation(field, data, "simulation_error", 1)
        loss = _training_loss(
            field,
            data.training,
            np.random.default_rng(0),
            "simulation_error",
            field.sigma,
            20,
            20,
            1,
            1,
        )
        self.assertAlmostEqual(score, 0.0, places=14)
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=14)


class MissingCompatibilityEvidenceTests(unittest.TestCase):
    def test_nonfinite_zero_initial_score_cannot_bypass_compatibility(self):
        field, data = fixture()
        with patch(
            "phsi.fitting._validation",
            side_effect=[1.0, float("inf"), 1.0, 1.0, 1.0, 1.0],
        ):
            with self.assertRaises(FitError) as caught:
                fit_field(field, data, max_steps=0, refit_steps=0)
        self.assertEqual(
            caught.exception.fit_info["candidates"][0]["status"], "compatibility_hold"
        )
        self.assertIsNone(
            caught.exception.fit_info["compatibility"]["zero_initial_score"]
        )
