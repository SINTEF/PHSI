"""
Small end-to-end tests of fitting, simulation and saved models.

Short training runs check behavior, not prediction accuracy.
"""

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import phsi

METHODS = ("integration_scheme", "simulation_error", "derivative", "midpoint")
GRADIENT_METHODS = METHODS[:2]


def observations(dimension):
    """
    Create small training and validation arrays with varying states.
    """
    phase = 0.2 + np.arange(dimension // 2) * 0.3
    train_times = np.linspace(0.0, 0.4, 16)
    validation_times = np.linspace(0.42, 0.46, 4)

    def states(times):
        angles = times[:, None] + phase[None, :]
        return np.concatenate((np.cos(angles), -np.sin(angles)), axis=1)

    train_input = 0.1 * np.sin(2 * train_times)
    validation_input = 0.1 * np.sin(2 * validation_times)
    return (
        states(train_times),
        train_times,
        train_input,
        (states(validation_times), validation_times, validation_input),
    )


class PublicWorkflowTests(unittest.TestCase):
    def fit(
        self,
        dimension,
        method,
        *,
        energy=None,
        damping="constant",
        gain="known",
        forced_state=None,
        gain_state=None,
    ):
        x, t, u, validation = observations(dimension)
        model = phsi.model(
            phsi.Polynomial(2) if energy is None else energy,
            damping=damping,
            forcing_gain=gain,
            forced_state=forced_state,
            gain_state=gain_state,
        )
        returned = model.fit(
            x,
            t,
            u,
            validation=validation,
            method=method,
            prune=False,
            max_steps=1,
            refit_steps=0,
            learning_rate=1e-4,
            validation_every=1,
            batch_size=4,
            window_batch=2,
            window_steps=3,
            n_substeps=1,
            validation_substeps=2,
            seed=12,
        )
        self.assertIs(returned, model)
        self.assertEqual(model.fit_info_["status"], "completed")
        self.assertTrue(model.fit_info_["grid_complete"])
        self.assertIsNotNone(model.fit_info_["selected"])
        self.assertEqual(model.field_.n_state, dimension)
        if method in GRADIENT_METHODS:
            self.assertEqual(model.fit_info_["dense"]["updates"], 1)
        else:
            self.assertEqual(len(model.fit_info_["candidates"]), 1)
            self.assertEqual(model.fit_info_["candidates"][0]["status"], "completed")
        # These arrays leave scope here. Only the fitted model reaches persistence.
        return model

    def assert_roundtrip(self, model, directory, name):
        dimension = model.field_.n_state
        probe = np.linspace(-0.15, 0.2, dimension)
        times = np.array([0.0, 0.001, 0.003])
        signal = np.array([0.02, 0.03, 0.01])
        expected = {
            "hamiltonian": model.hamiltonian(probe),
            "damping": model.damping(probe),
            "input_map": model.input_map(probe),
            "output": model.output(probe),
            "simulation": model.simulate(probe, times, signal),
        }
        path = Path(directory) / (name + ".pt")
        digest = model.save(path)
        self.assertEqual(len(digest), 64)
        text = str(model)
        retained = model.retained_terms_
        del model
        gc.collect()
        loaded = phsi.load(path)
        self.assertEqual(loaded.fit_info_["status"], "completed")
        self.assertEqual(loaded.field_.n_state, dimension)
        self.assertEqual(str(loaded), text)
        for component, description in retained.items():
            restored = loaded.retained_terms_[component]
            self.assertEqual(restored["names"], description["names"])
            if description["mask"] is None:
                self.assertIsNone(restored["mask"])
            else:
                np.testing.assert_array_equal(restored["mask"], description["mask"])
        self.assertEqual(
            loaded.simulate(probe, times, signal).shape, (len(times), dimension)
        )
        self.assertEqual(loaded.input_map(probe).shape, (dimension,))
        self.assertEqual(loaded.damping(probe).shape, (dimension // 2,))
        for key, value in expected.items():
            actual = (
                loaded.simulate(probe, times, signal)
                if key == "simulation"
                else getattr(loaded, key)(probe)
            )
            np.testing.assert_allclose(actual, value, rtol=1e-12, atol=1e-12)
        # Persisted fields and metadata reconstruct without observation arrays.
        payload = torch.load(path, weights_only=True, map_location="cpu")
        self.assertEqual(
            set(payload),
            {"schema", "package_version", "configuration", "state_dict", "fit_info"},
        )
        self.assertNotIn("observations", payload["configuration"])
        self.assertNotIn("training", payload["configuration"])

    def test_all_four_fitters_for_two_four_and_six_states(self):
        with tempfile.TemporaryDirectory() as directory:
            for dimension in (2, 4, 6):
                for method in METHODS:
                    with self.subTest(dimension=dimension, method=method):
                        model = self.fit(dimension, method)
                        self.assert_roundtrip(model, directory, f"{method}-{dimension}")

    def test_supported_neural_and_hybrid_gradient_workflows(self):
        def neural():
            return phsi.Neural(hidden=(3,))

        cases = [
            ("neural_energy", neural(), "constant", "known"),
            ("neural_energy_damping", neural(), neural(), "known"),
            ("analytic_energy_neural_damping", phsi.Polynomial(2), neural(), "known"),
            (
                "neural_energy_analytic_damping",
                neural(),
                phsi.DampingLibrary("diagnostic", widths=[0.5]),
                "known",
            ),
            (
                "analytic_damping",
                phsi.Polynomial(2),
                phsi.DampingLibrary("diagnostic", widths=[0.5]),
                "known",
            ),
            ("neural_energy_neural_gain", neural(), "constant", neural()),
            ("analytic_energy_neural_gain", phsi.Polynomial(2), "constant", neural()),
            ("neural_energy_analytic_gain", neural(), "constant", phsi.Polynomial(1)),
            (
                "analytic_energy_analytic_gain",
                phsi.Polynomial(2),
                "constant",
                phsi.Polynomial(1),
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for name, energy, damping, gain in cases:
                for method in GRADIENT_METHODS:
                    with self.subTest(components=name, method=method):
                        model = self.fit(
                            2, method, energy=energy, damping=damping, gain=gain
                        )
                        self.assert_roundtrip(model, directory, name + "-" + method)

    def test_public_restrictions_report_failed_fit_information(self):
        x, t, u, validation = observations(2)
        for method in ("derivative", "midpoint"):
            model = phsi.model(phsi.Neural(hidden=(3,)), forcing_gain="known")
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "fixed libraries"):
                    model.fit(
                        x, t, u, validation=validation, method=method, prune=False
                    )
                self.assertEqual(model.fit_info_["status"], "failed")
                self.assertFalse(hasattr(model, "field_"))
        model = phsi.model(phsi.Polynomial(2), forcing_gain="known")
        with self.assertRaisesRegex(ValueError, "known input"):
            model.fit(x, t, validation=(validation[0], validation[1], None))
        with self.assertRaises(ValueError):
            model.fit(np.ones((16, 3)), t, u)
        with self.assertRaisesRegex(ValueError, "at least"):
            model.fit(
                x[:5],
                t[:5],
                u[:5],
                validation=validation,
                method="simulation_error",
                window_steps=10,
            )
        self.assertEqual(model.fit_info_["status"], "failed")
        self.assertFalse(hasattr(model, "field_"))


if __name__ == "__main__":
    unittest.main()
