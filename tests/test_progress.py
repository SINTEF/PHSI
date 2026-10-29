"""
Tests for fitting progress, including failures and time limits.
"""

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from phsi.data import prepare_data
from phsi.errors import FitError
from phsi.fields import build_field
from phsi.fitting import _training_loss, fit_field
from phsi.libraries import DampingLibrary, Neural, Polynomial


def fixture(components=1, neural=False):
    t = np.linspace(0.0, 0.2, 12)
    x = np.column_stack((np.cos(t + 0.2), -np.sin(t + 0.2)))
    u = np.sin(t)
    data = prepare_data(x, t, u, validation=(x, t, u), forced=True)
    field = build_field(
        Neural(hidden=(3,)) if neural else Polynomial(2),
        DampingLibrary("diagnostic", widths=[0.5]) if components == 3 else "constant",
        Polynomial(1) if components >= 2 else "known",
        sigma=data.sigma,
        x_train=x,
        u_train=u,
        seed=19,
    )
    return field, data


class RecordingBar:
    def __init__(self, events, *, total, desc, disable=False, **kwargs):
        self.total, self.desc, self.disable = total, desc, disable
        self.n, self.closed = 0, False
        self.statuses = []
        self.update_statuses = []
        self.events = events
        events.append(("open", self))

    def update(self, count=1):
        self.n += count
        if self.n > self.total:
            raise AssertionError("Progress exceeded planned work.")
        self.update_statuses.append(self.statuses[-1] if self.statuses else "")
        self.events.append(("update", self))

    def set_postfix_str(self, text, **kwargs):
        self.statuses.append(text)

    def close(self):
        self.closed = True
        self.events.append(("close", self))


class ProgressTests(unittest.TestCase):
    def bars(self, events):
        return lambda **kwargs: RecordingBar(events, **kwargs)

    @staticmethod
    def loss(field, *args, **kwargs):
        return sum(parameter.square().sum() for parameter in field.parameters())

    def test_stage_order_and_full_fitting_refit_counts(self):
        field, data = fixture()
        events = []
        with (
            patch("phsi.fitting.tqdm", side_effect=self.bars(events)),
            patch("phsi.fitting._training_loss", side_effect=self.loss),
            patch("phsi.fitting._validation", return_value=1.0),
        ):
            _, info = fit_field(
                field,
                data,
                max_steps=2,
                refit_steps=1,
                validation_every=1,
                progress=True,
            )
        opened = [bar for event, bar in events if event == "open"]
        self.assertEqual(
            [bar.desc for bar in opened[:2]],
            ["Full-library fitting", "Pruning candidates"],
        )
        self.assertEqual((opened[0].n, opened[0].total), (2, 2))
        self.assertEqual((opened[1].n, opened[1].total), (5, 5))
        self.assertEqual(len(opened[2:]), 5)
        self.assertTrue(
            all("Refit" in bar.desc and bar.n == bar.total == 1 for bar in opened[2:])
        )
        self.assertTrue(all(bar.closed for bar in opened))
        self.assertLess(
            events.index(("close", opened[0])), events.index(("open", opened[1]))
        )
        self.assertTrue(
            any("Validat" in status for bar in opened for status in bar.statuses)
        )
        self.assertEqual(info["status"], "completed")

    def test_neural_fitting_and_refits_show_current_batch_loss(self):
        for method in ("integration_scheme", "simulation_error"):
            field, data = fixture(components=2, neural=True)
            events, losses = [], []

            def tracked_loss(*args, **kwargs):
                loss = _training_loss(*args, **kwargs)
                losses.append(float(loss.detach()))
                return loss

            with (
                self.subTest(method=method),
                patch("phsi.fitting.tqdm", side_effect=self.bars(events)),
                patch("phsi.fitting._training_loss", side_effect=tracked_loss),
                patch("phsi.fitting._validation", return_value=1.0),
            ):
                _, info = fit_field(
                    field,
                    data,
                    method=method,
                    max_steps=2,
                    refit_steps=1,
                    validation_every=1,
                    batch_size=3,
                    window_batch=2,
                    window_steps=3,
                    n_substeps=1,
                    progress=True,
                )
            bars = [
                bar
                for event, bar in events
                if event == "open" and bar.desc != "Pruning candidates"
            ]
            self.assertEqual(len(bars), 6)
            self.assertEqual(len(losses), 7)
            shown = [status for bar in bars for status in bar.update_statuses]
            self.assertEqual(shown, [f"Fitting, loss={loss:.4g}" for loss in losses])
            stages = [info["dense"], *info["candidates"]]
            for bar, stage in zip(bars, stages):
                label = f"loss={stage['final_loss']:.4g}"
                self.assertNotIn("loss=", bar.statuses[0])
                self.assertIn(f"Validating, {label}", bar.statuses)
                self.assertEqual(bar.statuses[-1], f"update_limit, {label}")
                self.assertTrue(bar.closed)

    def test_failure_retains_last_completed_update_loss(self):
        field, data = fixture(neural=True)
        events, losses = [], []

        def interrupted_loss(candidate, *args, **kwargs):
            if losses:
                raise FloatingPointError("Deliberate failure after one update.")
            loss = self.loss(candidate)
            losses.append(float(loss.detach()))
            return loss

        with (
            patch("phsi.fitting.tqdm", side_effect=self.bars(events)),
            patch("phsi.fitting._training_loss", side_effect=interrupted_loss),
            patch("phsi.fitting._validation", return_value=1.0),
        ):
            _, info = fit_field(field, data, max_steps=3, prune=False, progress=True)
        bar = next(bar for event, bar in events if event == "open")
        self.assertEqual(bar.n, 1)
        self.assertEqual(bar.statuses[-1], f"numerical_failure, loss={losses[0]:.4g}")
        self.assertEqual(info["dense"]["final_loss"], losses[0])
        self.assertTrue(bar.closed)

    def test_regression_candidate_totals_cover_full_grids(self):
        for components, count in ((1, 5), (2, 25), (3, 125)):
            field, data = fixture(components)
            events = []
            with (
                self.subTest(components=components),
                patch("phsi.fitting.tqdm", side_effect=self.bars(events)),
                patch(
                    "phsi.fitting.fit_regression",
                    side_effect=lambda candidate, *args, **kwargs: (
                        copy.deepcopy(candidate),
                        [],
                    ),
                ),
                patch("phsi.fitting._validation", return_value=1.0),
            ):
                _, info = fit_field(field, data, method="midpoint", progress=True)
            opened = [bar for event, bar in events if event == "open"]
            self.assertEqual(len(opened), 1)
            self.assertEqual(opened[0].desc, "Regression candidates")
            self.assertEqual((opened[0].n, opened[0].total), (count, count))
            self.assertTrue(opened[0].closed)
            self.assertEqual(len(info["candidates"]), count)

    def test_gradient_candidate_totals_and_zero_update_stages(self):
        for components, count in ((1, 5), (2, 25), (3, 125)):
            field, data = fixture(components)
            events = []
            with (
                self.subTest(components=components),
                patch("phsi.fitting.tqdm", side_effect=self.bars(events)),
                patch("phsi.fitting._validation", return_value=1.0),
                patch("phsi.fitting._training_loss") as loss,
            ):
                _, info = fit_field(
                    field, data, max_steps=0, refit_steps=0, progress=True
                )
            loss.assert_not_called()
            opened = [bar for event, bar in events if event == "open"]
            outer = next(bar for bar in opened if bar.desc == "Pruning candidates")
            self.assertEqual((outer.n, outer.total), (count, count))
            updates = [bar for bar in opened if bar is not outer]
            self.assertEqual(len(updates), count + 1)
            self.assertTrue(all(bar.n == bar.total == 0 for bar in updates))
            self.assertTrue(
                all("loss=" not in status for bar in updates for status in bar.statuses)
            )
            self.assertTrue(all(bar.closed for bar in opened))
            self.assertEqual(info["status"], "completed")

    def test_progress_false_is_silent(self):
        for method in (
            "integration_scheme",
            "simulation_error",
            "derivative",
            "midpoint",
        ):
            field, data = fixture()
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                self.subTest(method=method),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                fit_field(
                    field,
                    data,
                    method=method,
                    max_steps=1,
                    prune=False,
                    progress=False,
                    batch_size=3,
                    window_batch=2,
                    window_steps=3,
                )
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")

    def test_failed_update_stage_closes_without_completing_updates(self):
        field, data = fixture()
        events = []
        with (
            patch("phsi.fitting.tqdm", side_effect=self.bars(events)),
            patch("phsi.fitting._validation", return_value=1.0),
            patch(
                "phsi.fitting._training_loss",
                side_effect=FloatingPointError("Deliberate bounded failure."),
            ),
        ):
            _, info = fit_field(field, data, max_steps=4, prune=False, progress=True)
        bar = next(bar for event, bar in events if event == "open")
        self.assertEqual((bar.n, bar.total), (0, 4))
        self.assertTrue(bar.closed)
        self.assertEqual(info["dense"]["status"], "failed_fit")

    def test_regression_timeout_does_not_count_unprocessed_candidates(self):
        field, data = fixture()
        events = []
        calls = [0]

        def regression(candidate, *args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise TimeoutError("Deliberate bounded timeout.")
            return copy.deepcopy(candidate), []

        with (
            patch("phsi.fitting.tqdm", side_effect=self.bars(events)),
            patch("phsi.fitting.fit_regression", side_effect=regression),
            patch("phsi.fitting._validation", return_value=1.0),
        ):
            with self.assertRaises(FitError) as caught:
                fit_field(field, data, method="midpoint", progress=True)
        bar = next(bar for event, bar in events if event == "open")
        self.assertEqual((bar.n, bar.total), (1, 5))
        self.assertTrue(bar.closed)
        self.assertEqual(
            [row["status"] for row in caught.exception.fit_info["candidates"]],
            ["completed", "timed_out", "not_run", "not_run", "not_run"],
        )

    def test_update_timeout_and_unexpected_exception_close_partial_bars(self):
        for failure in (
            TimeoutError("Deliberate timeout."),
            KeyError("Deliberate unexpected exception."),
        ):
            field, data = fixture()
            events = []
            with (
                self.subTest(failure=type(failure).__name__),
                patch("phsi.fitting.tqdm", side_effect=self.bars(events)),
                patch("phsi.fitting._validation", return_value=1.0),
                patch("phsi.fitting._training_loss", side_effect=failure),
            ):
                if isinstance(failure, TimeoutError):
                    _, info = fit_field(
                        field, data, max_steps=3, prune=False, progress=True
                    )
                    self.assertEqual(info["dense"]["stop_reason"], "time_limit")
                else:
                    with self.assertRaises(FitError):
                        fit_field(field, data, max_steps=3, prune=False, progress=True)
            opened = [bar for event, bar in events if event == "open"]
            self.assertEqual(len(opened), 1)
            self.assertEqual((opened[0].n, opened[0].total), (0, 3))
            self.assertTrue(opened[0].closed)

    def test_nested_candidate_exception_closes_outer_and_partial_refit(self):
        field, data = fixture()
        events, calls = [], [0]

        def interrupted_loss(candidate, *args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise KeyError("Deliberate candidate interruption.")
            return self.loss(candidate)

        with (
            patch("phsi.fitting.tqdm", side_effect=self.bars(events)),
            patch("phsi.fitting._validation", return_value=1.0),
            patch("phsi.fitting._training_loss", side_effect=interrupted_loss),
        ):
            with self.assertRaises(FitError):
                fit_field(field, data, max_steps=0, refit_steps=3, progress=True)
        opened = [bar for event, bar in events if event == "open"]
        self.assertEqual(
            [(bar.desc, bar.n, bar.total) for bar in opened],
            [
                ("Full-library fitting", 0, 0),
                ("Pruning candidates", 0, 5),
                ("Refit candidate-00", 1, 3),
            ],
        )
        self.assertTrue(all(bar.closed for bar in opened))

    def test_regression_rendering_overhead_counts_before_final_deadline_audit(self):
        for charged_event in ("update", "close"):
            field, data = fixture()
            events, clock = [], [0.0]

            class ChargedBar(RecordingBar):
                def update(self, count=1):
                    super().update(count)
                    if charged_event == "update":
                        clock[0] = 11.0

                def close(self):
                    super().close()
                    if charged_event == "close":
                        clock[0] = 11.0

            with (
                self.subTest(event=charged_event),
                tempfile.TemporaryDirectory() as directory,
                patch("phsi.fitting.time.monotonic", side_effect=lambda: clock[0]),
                patch(
                    "phsi.fitting.tqdm",
                    side_effect=lambda **kwargs: ChargedBar(events, **kwargs),
                ),
                patch(
                    "phsi.fitting.fit_regression",
                    side_effect=lambda candidate, *args, **kwargs: (
                        copy.deepcopy(candidate),
                        [],
                    ),
                ),
                patch("phsi.fitting._validation", return_value=1.0),
            ):
                with self.assertRaises(FitError) as caught:
                    fit_field(
                        field,
                        data,
                        method="midpoint",
                        progress=True,
                        prune=False,
                        time_limit=10.0,
                        started=0.0,
                        output_dir=Path(directory) / "run",
                    )
                self.assertEqual(caught.exception.fit_info["status"], "incomplete")
                self.assertIsNone(caught.exception.fit_info["selected"])
                self.assertEqual(caught.exception.fit_info["elapsed_s"], 11.0)
            opened = [bar for event, bar in events if event == "open"]
            self.assertTrue(opened[0].closed)
            self.assertEqual((opened[0].n, opened[0].total), (1, 1))

    def test_refit_postfix_rendering_is_inside_each_candidate_allowance(self):
        for display_cost in (2.0, 7.0):
            field, data = fixture()
            events, clock = [], [0.0]

            class ChargedPostfixBar(RecordingBar):
                def set_postfix_str(self, text, **kwargs):
                    super().set_postfix_str(text, **kwargs)
                    if self.desc == "Pruning candidates" and text.startswith(
                        "Refitting "
                    ):
                        clock[0] += display_cost

            with (
                self.subTest(display_cost=display_cost),
                tempfile.TemporaryDirectory() as directory,
                patch("phsi.fitting.time.monotonic", side_effect=lambda: clock[0]),
                patch(
                    "phsi.fitting.tqdm",
                    side_effect=lambda **kwargs: ChargedPostfixBar(events, **kwargs),
                ),
                patch("phsi.fitting._validation", return_value=1.0) as validation,
                patch("phsi.fitting._training_loss", side_effect=self.loss) as loss,
            ):
                try:
                    _, info = fit_field(
                        field,
                        data,
                        max_steps=0,
                        refit_steps=1,
                        refit_time_limit=6.0,
                        started=0.0,
                        progress=True,
                        output_dir=Path(directory) / "run",
                    )
                except FitError as error:
                    info = error.fit_info
                self.assertTrue(
                    all(not row["selection_eligible"] for row in info["candidates"])
                )
                self.assertEqual(info["status"], "incomplete")
                self.assertIsNone(info["selected"])
                self.assertEqual(len(info["candidates"]), 5)
                self.assertTrue(
                    all(
                        row["updates"] == 0 and row["history"] == []
                        for row in info["candidates"]
                    )
                )
                self.assertTrue(
                    all(row["status"] == "timed_out" for row in info["candidates"])
                )
                validation.assert_called_once()
                loss.assert_not_called()
            opened = [bar for event, bar in events if event == "open"]
            self.assertTrue(all(bar.closed for bar in opened))
            outer = next(bar for bar in opened if bar.desc == "Pruning candidates")
            self.assertEqual((outer.n, outer.total), (0, 5))

    def test_progress_keeps_parameters_selection_and_random_streams_identical(self):
        for method in (
            "integration_scheme",
            "simulation_error",
            "derivative",
            "midpoint",
        ):
            initial, data = fixture()
            results = []
            before_numpy = np.random.get_state()
            before_torch = torch.random.get_rng_state().clone()
            for enabled in (False, True):
                events = []
                with patch("phsi.fitting.tqdm", side_effect=self.bars(events)):
                    field, info = fit_field(
                        copy.deepcopy(initial),
                        data,
                        method=method,
                        progress=enabled,
                        max_steps=2,
                        refit_steps=0,
                        prune=False,
                        batch_size=3,
                        window_batch=2,
                        window_steps=3,
                        n_substeps=1,
                        seed=17,
                    )
                results.append((field, info))
            with self.subTest(method=method):
                for key, tensor in results[0][0].state_dict().items():
                    torch.testing.assert_close(
                        tensor, results[1][0].state_dict()[key], rtol=0.0, atol=0.0
                    )
                self.assertEqual(results[0][1]["selected"], results[1][1]["selected"])
                self.assertEqual(results[0][1]["masks"], results[1][1]["masks"])
                self.assertEqual(np.random.get_state()[0], before_numpy[0])
                np.testing.assert_array_equal(np.random.get_state()[1], before_numpy[1])
                self.assertEqual(np.random.get_state()[2:], before_numpy[2:])
                torch.testing.assert_close(
                    torch.random.get_rng_state(), before_torch, rtol=0.0, atol=0.0
                )


if __name__ == "__main__":
    unittest.main()
