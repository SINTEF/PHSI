"""
Tests for model fitting, simulation, saving and loading.
"""

import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import phsi
from phsi.errors import FitError, SimulationError
from phsi.estimator import Model
from phsi.fields import build_field
from phsi.libraries import CustomLibrary, Neural, Polynomial
from phsi.persistence import read_field, save_field
from phsi.simulation import simulate_field


class EstimatorTests(unittest.TestCase):
    def field(self, input="none"):
        x = np.array([[0.0, 1.0], [1.0, 0.3], [-0.2, -0.5], [0.1, -0.2]])
        f = build_field(
            Polynomial(2),
            "constant",
            input,
            sigma=np.ones(2),
            x_train=x,
            u_train=np.ones(4),
        )
        coefficients = np.zeros(len(f.energy.coef))
        for k, exponent in enumerate(f.energy.library.exponents.tolist()):
            if sum(exponent) == 2 and max(exponent) == 2:
                coefficients[k] = 0.5
        f.energy.set_physical(coefficients)
        f.damping.set_physical(np.zeros(1))
        return f

    def model(self, input="none"):
        m = Model(Polynomial(2), forcing_gain=input)
        m.field_ = self.field(input)
        m.fit_info_ = {"status": "completed", "selected": {"kind": "fixture"}}
        return m

    def test_factory_and_unfitted_errors(self):
        m = phsi.model()
        self.assertIsInstance(m, Model)
        with self.assertRaises(RuntimeError):
            m.hamiltonian([0.0, 1.0])
        with self.assertRaises(RuntimeError):
            m.simulate([0.0, 1.0], [0.0, 1.0])
        self.assertFalse(hasattr(phsi, "phsi"))
        self.assertEqual(phsi.__version__, "1.0.0a1")

    def test_checkpoint_records_package_version_and_loads_development_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.phsi"
            model = self.model()
            model.save(path)
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload["package_version"], phsi.__version__)
            # Existing local models used this label with the same schema.
            payload["package_version"] = "0.1.0"
            torch.save(payload, path)
            restored = phsi.load(path)
            np.testing.assert_array_equal(
                restored.hamiltonian([1.0, 2.0]), model.hamiltonian([1.0, 2.0])
            )

    def test_public_analytic_component_aliases(self):
        t = np.linspace(0.0, 0.3, 30)
        x = np.column_stack((np.cos(t), -np.sin(t)))

        def completed(field, *args, **kwargs):
            return field, {"status": "completed", "selected": {"kind": "fixture"}}

        for damping, input in (("constant", "analytic"), ("analytic", "known")):
            with (
                self.subTest(damping=damping, forcing_gain=input),
                patch("phsi.fitting.fit_field", side_effect=completed),
            ):
                m = phsi.model(
                    hamiltonian="analytic", damping=damping, forcing_gain=input
                )
                self.assertIs(
                    m.fit(x, t, u=np.ones(len(t)), validation=(x, t, np.ones(len(t)))),
                    m,
                )
                self.assertEqual(m.field_.energy.kind, "analytic")
                self.assertEqual(len(m.field_.energy.coef), 14)
                if input == "analytic":
                    self.assertEqual(m.field_.gain.kind, "analytic")
                    self.assertEqual(
                        m.field_.gain.library.exponents[:, 0].tolist(), list(range(5))
                    )
                if damping == "analytic":
                    self.assertEqual(m.field_.damping.kind, "analytic")
                    self.assertEqual(m.field_.damping.library.n_features, 16)

    def test_evaluations_and_print(self):
        m = self.model("known")
        np.testing.assert_allclose(m.hamiltonian([[1.0, 2.0], [0.0, 1.0]]), [2.5, 0.5])
        np.testing.assert_array_equal(m.damping([1.0, 2.0]), [0.0])
        np.testing.assert_array_equal(m.input_map([1.0, 2.0]), [0.0, 1.0])
        self.assertEqual(float(m.output([1.0, 2.0])), 2.0)
        self.assertEqual(set(m.coefficients_), {"energy", "damping", "gain"})
        self.assertEqual(set(m.retained_terms_), {"energy", "damping", "gain"})
        text = str(m)
        for expected in ("H(", "R = diag", "G(", "dq1/dt", "dp1/dt"):
            self.assertIn(expected, text)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(m.print())
        self.assertEqual(output.getvalue().strip(), text)

    def test_checkpoint_roundtrip_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            m = self.model()
            digest = m.save(path)
            self.assertEqual(len(digest), 64)
            loaded = phsi.load(path)
            np.testing.assert_allclose(
                loaded.hamiltonian([0.3, 0.4]), m.hamiltonian([0.3, 0.4])
            )
            np.testing.assert_array_equal(loaded.damping([0.3, 0.4]), [0.0])
            self.assertEqual(loaded.fit_info_, m.fit_info_)
            self.assertEqual(str(loaded), str(m))
            with self.assertRaises(FileExistsError):
                m.save(path)
            m.save(path, overwrite=True)
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload["schema"], "phsi-model-v1")
            payload["state_dict"]["energy.mask"][0] = 0.5
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                read_field(path)

    def test_neural_and_custom_checkpoint_without_training_data(self):
        x = np.array([[0.0, 1.0], [1.0, 0.3], [-0.2, -0.5], [0.1, -0.2]])
        with tempfile.TemporaryDirectory() as directory:
            f = build_field(
                Neural(hidden=(3,)),
                Neural(hidden=(2,)),
                "known",
                sigma=np.ones(2),
                x_train=x,
                u_train=np.ones(4),
            )
            path = Path(directory) / "neural.pt"
            save_field(f, path, {"status": "completed"})
            loaded = phsi.load(path)
            np.testing.assert_allclose(
                loaded.hamiltonian(x), f.energy(torch.tensor(x)).detach().numpy()
            )
            self.assertIn("hidden=[3]", str(loaded))
            custom = CustomLibrary(
                lambda z: z.square(),
                lambda z: torch.diag_embed(2 * z),
                lambda names: [name + "^2" for name in names],
                "squares",
            )
            f = build_field(
                custom,
                "constant",
                "none",
                sigma=np.ones(2),
                x_train=x,
                u_train=np.ones(4),
            )
            path = Path(directory) / "custom.pt"
            save_field(f, path)
            with self.assertRaisesRegex(ValueError, "custom_components"):
                phsi.load(path)
            loaded = phsi.load(path, {"squares": custom})
            np.testing.assert_allclose(
                loaded.hamiltonian(x), f.energy(torch.tensor(x)).detach().numpy()
            )
            with self.assertRaises(ValueError):
                save_field(f, Path(directory) / "unsafe.pt", {"callable": lambda: 0})
            self.assertFalse((Path(directory) / "unsafe.pt").exists())

    def test_simulation_complete_and_partial_failure(self):
        m = self.model()
        times = np.array([0.0, 0.01, 0.1, 0.23, 0.4])
        result = m.simulate([1.0, 0.0], times)
        np.testing.assert_allclose(
            result, np.column_stack((np.cos(times), -np.sin(times))), atol=1e-9
        )
        with self.assertRaises(SimulationError) as caught:
            m.simulate([1.0, 0.0], times, max_evaluations=3)
        self.assertLess(len(caught.exception.t), len(times))
        self.assertEqual(caught.exception.x.shape, (len(caught.exception.t), 2))
        self.assertEqual(caught.exception.diagnostics["evaluations"], 3)
        with self.assertRaises(SimulationError):
            simulate_field(
                m.field_, [1.0, 0.0], times, None, deadline=time.monotonic() - 1
            )
        with self.assertRaises(ValueError):
            m.simulate([1.0, 0.0], [0.0, 0.1, 0.1])

    def test_forced_midpoint_input_and_explicit_guard(self):
        m = self.model("known")
        m.field_.energy.mask.zero_()
        t = np.array([0.0, 0.1, 0.4, 1.0])
        result = m.simulate([0.0, 0.0], t, lambda q: q)
        np.testing.assert_allclose(result[:, 1], 0.5 * t * t, atol=1e-10)
        with self.assertRaises(SimulationError):
            m.simulate([0.0, 0.0], t, lambda q: 1e9 + 0 * q, max_state=2.0)

    def test_held_simulation_knots_and_narrow_pulse(self):
        from phsi.data import InputSignal

        m = self.model("known")
        m.field_.energy.mask.zero_()
        signal = InputSignal([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], "previous")
        got = simulate_field(m.field_, [0.0, 0.0], [0.0, 1.0, 2.0], signal)
        np.testing.assert_allclose(got[:, 1], [0.0, 0.0, 1.0], atol=1e-11, rtol=0.0)
        signal = InputSignal(
            [0.0, 0.431, 0.43101, 1.0], [0.0, 2.0, 0.0, 0.0], "previous"
        )
        got = simulate_field(m.field_, [0.0, 0.0], [0.0, 1.0], signal)
        self.assertAlmostEqual(got[-1, 1], 0.00002, places=12)

    def test_simulation_evaluation_mode_and_restore_on_failure(self):
        class EvaluationField(torch.nn.Module):
            n_state = 2
            input_kind = "none"

            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)
                self.modes = []

            def numpy_rhs(self, state, u):
                self.modes.append((self.training, self.dropout.training))
                return self.dropout(torch.ones(2)).numpy()

        field = EvaluationField()
        field.train()
        simulate_field(field, [0.0, 0.0], [0.0, 0.01], None)
        self.assertTrue(field.modes)
        self.assertTrue(all(mode == (False, False) for mode in field.modes))
        self.assertTrue(field.training)
        self.assertTrue(field.dropout.training)
        field.modes.clear()
        with self.assertRaises(SimulationError):
            simulate_field(field, [0.0, 0.0], [0.0, 1.0], None, max_evaluations=1)
        self.assertTrue(all(mode == (False, False) for mode in field.modes))
        self.assertTrue(field.training)
        self.assertTrue(field.dropout.training)

    def test_callable_previous_label_does_not_change_true_callable(self):
        from phsi.data import InputSignal

        m = self.model("known")
        m.field_.energy.mask.zero_()
        signal = InputSignal([0.0, 0.25, 1.0], lambda q: q * q, "previous")
        self.assertFalse(signal.is_held)
        got = simulate_field(m.field_, [0.0, 0.0], [0.0, 1.0], signal)
        self.assertAlmostEqual(got[-1, 1], 1.0 / 3.0, places=11)

    def test_timed_fit_requires_output_before_build(self):
        m = Model(Polynomial(2))
        with patch("phsi.estimator.prepare_data") as preparation:
            with self.assertRaises(ValueError):
                m.fit(np.zeros((30, 2)), np.arange(30.0), time_limit=1.0)
            preparation.assert_not_called()
        self.assertEqual(m.fit_info_["status"], "failed")

    def test_small_public_fit_and_timed_directory_preflight(self):
        t = np.linspace(0.0, 0.3, 30)
        x = np.column_stack((np.cos(t), -np.sin(t)))
        m = phsi.model(Polynomial(2))
        m.fit(x, t, method="derivative", prune=False)
        self.assertEqual(m.fit_info_["status"], "completed")
        self.assertIsNotNone(m.fit_info_["selected"])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new-fit"
            with patch(
                "phsi.fitting.fit_field",
                return_value=(self.field(), {"status": "completed", "selected": {}}),
            ):
                m.fit(x, t, output_dir=target, time_limit=10.0)
            self.assertFalse(target.exists())
            with self.assertRaises(FileExistsError):
                m.fit(x, t, output_dir=directory)

    def test_failed_fit_has_information_and_partial_model(self):
        m = Model(Polynomial(2))
        x = np.column_stack((np.arange(30.0) * 0.01, np.arange(30.0) * 0.02))
        failure = FitError(
            "Fixture failure.",
            fit_info={"status": "incomplete"},
            partial_model=self.field(),
        )
        with patch("phsi.fitting.fit_field", side_effect=failure):
            with self.assertRaises(FitError):
                m.fit(x, np.arange(30.0) * 0.1)
        self.assertEqual(m.fit_info_["status"], "incomplete")
        self.assertFalse(hasattr(m, "field_"))


if __name__ == "__main__":
    unittest.main()


class NeuralDisplayTests(unittest.TestCase):
    def test_print_names_builtin_activation_and_custom_damping_arguments(self):
        x = np.random.default_rng(5).normal(size=(8, 4))
        custom = Neural(module=torch.nn.Linear(2, 2), key="coupled_damping")
        m = Model()
        m.field_ = build_field(
            Neural(hidden=(3,)),
            custom,
            "known",
            sigma=x.std(0),
            x_train=x,
            u_train=np.ones(8),
        )
        text = str(m)
        self.assertIn("activation=tanh", text)
        self.assertIn("coupled_damping", text)
        self.assertIn("[p1, p2] / sigma[2:]", text)
        self.assertNotIn("sigma = ", text)
        self.assertIn("Linear(in_features=2, out_features=2", text)


class InferenceModeTests(unittest.TestCase):
    def test_completed_and_reloaded_models_evaluate_custom_modules_in_eval_mode(self):
        t = np.linspace(0.0, 1.0, 10)
        x = np.column_stack((np.cos(t), np.sin(t)))
        custom = Neural(
            module=torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(2, 1)),
            key="dropout_energy",
        )

        def completed(field, *args, **kwargs):
            return field, {"status": "completed", "selected": {"kind": "fixture"}}

        with patch("phsi.fitting.fit_field", side_effect=completed):
            m = phsi.model(hamiltonian=custom).fit(x, t, validation=(x, t, None))
        self.assertFalse(m.field_.training)
        np.testing.assert_array_equal(m.hamiltonian(x), m.hamiltonian(x))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference.phsi"
            m.save(path)
            restored = phsi.load(path, custom_components={"dropout_energy": custom})
            self.assertFalse(restored.field_.training)
            np.testing.assert_array_equal(restored.hamiltonian(x), m.hamiltonian(x))


class ForcingParameterNamesTests(unittest.TestCase):
    def test_named_forcing_parameters_select_the_physical_coordinates(self):
        t = np.linspace(0.0, 1.0, 15)
        x = np.column_stack((np.cos(t), np.cos(2 * t), -np.sin(t), -2 * np.sin(2 * t)))

        def completed(field, *args, **kwargs):
            return field, {"status": "completed", "selected": {"kind": "fixture"}}

        with patch("phsi.fitting.fit_field", side_effect=completed):
            m = phsi.model(
                hamiltonian="analytic",
                damping="constant",
                forcing_gain="analytic",
                forced_state=3,
                gain_state=1,
            )
            m.fit(x, t, u=np.ones(len(t)), validation=(x, t, np.ones(len(t))))
        self.assertEqual(m.field_.input_index, 3)
        self.assertEqual(m.field_.input_argument, 1)
        coefficients = np.zeros_like(m.field_.gain.physical_coefficients())
        coefficients[1] = 1.0
        m.field_.gain.set_physical(coefficients)
        np.testing.assert_array_equal(
            m.input_map([2.0, 3.0, 4.0, 5.0]), [0.0, 0.0, 0.0, 3.0]
        )
        for name, value in (("forced_state", 0), ("gain_state", 4)):
            with self.subTest(parameter=name), self.assertRaisesRegex(ValueError, name):
                phsi.model(
                    hamiltonian="analytic", forcing_gain="analytic", **{name: value}
                ).fit(x, t, u=np.ones(len(t)), validation=(x, t, np.ones(len(t))))
