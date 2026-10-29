"""
Tests for analytic and neural model components.
"""

import importlib
import json
import os
import sys
import types
import unittest

import numpy as np
import torch

from phsi.fields import build_field, field_from_configuration
from phsi.libraries import (
    CustomLibrary,
    DampingLibrary,
    GaussianExtension,
    Neural,
    Polynomial,
)


class ComponentsTest(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(71).normal(size=(20, 6))
        self.sigma = self.x.std(axis=0)

    def field(self, energy=None, damping="constant", gain="known", **kwargs):
        return build_field(
            Polynomial() if energy is None else energy,
            damping,
            gain,
            sigma=self.sigma,
            x_train=self.x,
            u_train=np.ones(20),
            **kwargs,
        )

    def test_six_state_canonical_field_and_physical_scaling(self):
        f = self.field()
        coefficients = np.zeros(len(f.energy.coef))
        for k, exponent in enumerate(f.energy.library.exponents):
            if sum(exponent) == 2 and max(exponent) == 2:
                coefficients[k] = 0.5
        f.energy.set_physical(coefficients)
        f.damping.set_physical(np.zeros(3))
        x = torch.tensor(self.x)
        expected = torch.cat((x[:, 3:], -x[:, :3]), -1)
        torch.testing.assert_close(f(x), expected)
        torch.testing.assert_close(f.energy_gradient(x), x)
        np.testing.assert_allclose(f.energy.physical_coefficients(), coefficients)
        f(x).sum().backward()
        self.assertTrue(torch.isfinite(f.damping.raw.grad).all())

    def test_resolved_round_trip_does_not_need_data(self):
        f = self.field(GaussianExtension(), DampingLibrary())
        f.damping.mask.zero_()
        config = json.loads(json.dumps(f.configuration()))
        g = field_from_configuration(config)
        g.load_state_dict(f.state_dict(), strict=True)
        x = torch.tensor(self.x)
        torch.testing.assert_close(f(x, 2), g(x, 2))
        self.assertEqual(f.damping.features(x[:, 3:]).shape, (20, 3, 16))
        self.assertTrue(torch.all(f.damping_values(x) >= 0))

    def test_gain_signed_and_scaled_by_input(self):
        f = self.field(gain=Polynomial(), input_argument=0)
        coefficients = np.zeros(len(f.gain.coef))
        coefficients[0] = -2
        f.gain.set_physical(coefficients)
        self.assertTrue(np.all(f.gain.physical_coefficients() <= 0))
        x = torch.tensor(self.x)
        difference = f(x, 3) - f(x, 0)
        expected = torch.zeros_like(x)
        expected[:, -1] = -6
        torch.testing.assert_close(difference, expected)

    def test_all_empty_masks_and_neural_gradients(self):
        f = self.field()
        f.energy.mask.zero_()
        self.assertTrue(
            torch.equal(f(torch.tensor(self.x)), torch.zeros_like(torch.tensor(self.x)))
        )
        f = self.field(Neural(), Neural())
        x = torch.tensor(self.x, requires_grad=True)
        f(x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_neural_energy_gradient_uses_physical_coordinates(self):
        self.sigma = np.array([0.5, 2.0, 0.25, 3.0, 4.0, 5.0])
        field = self.field(Neural(hidden=()), gain="none")
        weights = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=torch.float64)
        with torch.no_grad():
            field.energy.net[0].weight.copy_(weights[None, :])
            field.energy.net[0].bias.fill_(1.75)
        field.damping.set_physical(np.zeros(3))
        x = torch.tensor(self.x)
        sigma = torch.tensor(self.sigma)
        torch.testing.assert_close(
            field.energy(x), (x / sigma * weights).sum(-1) + 1.75
        )
        gradient = (weights / sigma).expand_as(x)
        torch.testing.assert_close(field.energy_gradient(x), gradient)
        torch.testing.assert_close(
            field(x), torch.cat((gradient[:, 3:], -gradient[:, :3]), -1)
        )

    def test_numpy_and_library_derivatives(self):
        f = self.field(GaussianExtension(), DampingLibrary())
        x = torch.tensor(self.x, requires_grad=True)
        torch.testing.assert_close(
            f.energy.library.gradients(x),
            torch.tensor(f.energy.library.numpy_gradients(self.x)),
        )
        torch.testing.assert_close(
            f.energy.library.values(x),
            torch.tensor(f.energy.library.numpy_values(self.x)),
        )
        derivative = torch.autograd.grad(f.energy(x).sum(), x)[0]
        torch.testing.assert_close(f.energy_gradient(x), derivative)
        np.testing.assert_allclose(
            f.numpy_rhs(self.x, 2), f(x, 2).detach().numpy(), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            f.numpy_rhs(self.x[0], 2),
            f(x[0], 2).detach().numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
        neural = self.field(Neural(), Neural())
        np.testing.assert_allclose(
            neural.numpy_rhs(self.x, 2), neural(x, 2).detach().numpy()
        )

    def test_custom_and_restrictions(self):
        custom = CustomLibrary(
            lambda x: x.square(),
            lambda x: torch.diag_embed(2 * x),
            lambda names: [f"{n}^2" for n in names],
            key="squares",
        )
        f = self.field(custom)
        cfg = f.configuration()
        with self.assertRaisesRegex(ValueError, "custom_components"):
            field_from_configuration(cfg)
        g = field_from_configuration(cfg, {"squares": custom})
        g.load_state_dict(f.state_dict())
        torch.testing.assert_close(f(torch.tensor(self.x)), g(torch.tensor(self.x)))
        joint = self.field(damping=DampingLibrary(), gain=Polynomial())
        self.assertEqual(joint.input_kind, "analytic")
        self.assertTrue(bool((joint.damping_values(torch.tensor(self.x)) >= 0).all()))
        with self.assertRaises(ValueError):
            self.field(input_index=0)


@unittest.skipUnless(
    os.environ.get("PHSI_FROZEN_SOURCE"),
    "Set PHSI_FROZEN_SOURCE for optional source-equivalence checks.",
)
class FrozenSourceTest(unittest.TestCase):
    setUp = ComponentsTest.setUp
    field = ComponentsTest.field

    @classmethod
    def setUpClass(cls):
        name = "_phsi_frozen_components"
        package = types.ModuleType(name)
        package.__path__ = [os.environ["PHSI_FROZEN_SOURCE"]]
        sys.modules[name] = package
        systems = types.ModuleType(name + ".systems")
        systems.INPUT_AMP = 1.0
        systems.OMEGAS = (1.0,)
        systems.input_u = None
        sys.modules[name + ".systems"] = systems
        cls.source = importlib.import_module(name + ".models")
        cls.source_features = importlib.import_module(name + ".features")

    def test_fixed_source_forward_and_parameter_gradient(self):
        f = self.field(GaussianExtension(), DampingLibrary())
        E = self.source_features.exponent_table(6, 4)
        np.testing.assert_array_equal(f.energy.library.exponents.numpy(), E)
        energy = self.source.FixedLibraryEnergy(
            E,
            f.energy.scale.numpy(),
            coef=f.energy.coef.detach().numpy(),
            smooth_features=f.energy.library.features,
        )
        damping = self.source.FeatureDamping(
            self.source.damping_feature_list("dict"), n_dof=3
        )
        with torch.no_grad():
            damping.raw.copy_(f.damping.raw)
        original = self.source.PHModel(
            energy, damping, self.sigma, amp=2.0, omegas=[1.0]
        ).double()
        x = torch.tensor(self.x, requires_grad=True)
        actual = f(x, 2.0)
        expected = original(
            x,
            torch.full((20,), np.pi / 2, dtype=torch.float64),
            torch.zeros((20, 1), dtype=torch.float64),
        )
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.square().sum().backward(retain_graph=True)
        expected.square().sum().backward()
        torch.testing.assert_close(
            f.energy.coef.grad, original.energy.coef.grad, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            f.damping.raw.grad, original.damping.raw.grad, rtol=1e-12, atol=1e-12
        )

    def test_learned_gain_initialization(self):
        f = self.field(gain=Polynomial(), input_argument=0, seed=51)
        dataset = types.SimpleNamespace(
            E=f.energy.library.exponents.numpy(),
            scales=f.energy.scale.numpy(),
            sigma=self.sigma,
            n_state=6,
            sha256="test",
            cfg=types.SimpleNamespace(amp=2.0, omegas=[1.0]),
            structure=types.SimpleNamespace(input_index=5),
        )
        gain_spec = dict(
            kind="library",
            argument_index=0,
            resolved=True,
            training_dataset_sha256="test",
            powers=list(range(5)),
            features=[],
            scales=f.gain.scale.numpy().tolist(),
        )
        original = self.source.make_model(
            dict(energy="poly", damping="constant", gain=gain_spec), dataset, 51
        )
        torch.testing.assert_close(
            f.energy.coef, original.energy.coef, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            f.gain.coef, original.gain.coef, rtol=1e-12, atol=1e-12
        )
        x = torch.tensor(self.x)
        torch.testing.assert_close(
            f.gain_values(x), original.gain_values(x), rtol=1e-12, atol=1e-12
        )

    def test_neural_source_initialization_and_field(self):
        seed = 47
        f = self.field(Neural(), Neural(), seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            energy = self.source.NeuralEnergy(self.sigma)
            damping = self.source.NeuralDamping(self.sigma[3:])
            original = self.source.PHModel(
                energy, damping, self.sigma, amp=2.0, omegas=[1.0]
            ).double()
        x = torch.tensor(self.x)
        actual = f(x, 2.0)
        expected = original(
            x,
            torch.full((20,), np.pi / 2, dtype=torch.float64),
            torch.zeros((20, 1), dtype=torch.float64),
        )
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            f.energy_gradient(x), original.gradH(x), rtol=1e-12, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
