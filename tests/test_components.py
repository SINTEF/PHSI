"""
Tests for analytic and neural model components.
"""

import json
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


if __name__ == "__main__":
    unittest.main()
