"""
Tests for regression matrices, coefficient bounds and term removal.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from phsi.data import InputSignal, Trajectory
from phsi.errors import SupportNonconvergence
from phsi.fields import build_field
from phsi.libraries import DampingLibrary, Polynomial
from phsi.regression import bounded_lstsq, build_design, fit_regression


class RegressionTests(unittest.TestCase):
    def field(self, dimension=2, gain="known", damping="constant", degree=2):
        x = np.random.default_rng(72).normal(size=(50, dimension))
        field = build_field(
            Polynomial(degree),
            damping,
            gain,
            sigma=np.arange(1, dimension + 1.0),
            x_train=x,
            u_train=np.ones(len(x)),
        )
        return field, x

    def trajectory(self, x, rates=None, input=None):
        t = np.arange(len(x)) * 0.1
        return Trajectory(
            x,
            t,
            InputSignal(t, input if input is not None else np.zeros(len(t))),
            rates,
        )

    def test_generic_known_and_absent_input_design(self):
        for dimension in (2, 4, 6):
            for gain in ("known", "none"):
                f, x = self.field(dimension, gain)
                physical = np.linspace(-0.2, 0.4, len(f.energy.coef))
                f.energy.set_physical(physical)
                damping = np.arange(1, f.n_dof + 1) * 0.1
                f.damping.set_physical(damping)
                u = np.linspace(-1, 1, len(x))
                rates = f.numpy_rhs(x, u)
                A, b, signed = build_design(f, x, rates, u)
                coefficient = np.r_[f.energy.normalized_coefficients(), damping]
                np.testing.assert_allclose(A @ coefficient, b, rtol=1e-12, atol=1e-12)
                self.assertEqual(signed, len(f.energy.coef))
                self.assertEqual(A.shape[0], len(x) * dimension)

    def test_signed_gain_and_nonnegative_analytic_damping(self):
        f, x = self.field(4, Polynomial(2))
        f.energy.set_physical(np.linspace(-0.3, 0.4, len(f.energy.coef)))
        f.gain.set_physical(np.array([-2.0, 0.1, -0.2]))
        f.damping.set_physical(np.array([0.2, 0.3]))
        rates = f.numpy_rhs(x, np.linspace(-1, 1, len(x)))
        A, b, signed = build_design(f, x, rates, np.linspace(-1, 1, len(x)))
        coefficients = np.r_[
            f.energy.normalized_coefficients(),
            f.gain.normalized_coefficients(),
            [0.2, 0.3],
        ]
        np.testing.assert_allclose(A @ coefficients, b, atol=1e-12)
        self.assertEqual(signed, len(f.energy.coef) + len(f.gain.coef))
        g, x = self.field(2, "none", DampingLibrary("diagnostic", widths=[0.5]))
        g.energy.set_physical(np.arange(len(g.energy.coef)) * 0.1)
        theta = np.array([[0.1, 0.2, 0.3]])
        g.damping.set_physical(theta)
        g.damping_scales.copy_(torch.tensor([[2.0, 3.0, 4.0]]))
        rates = g.numpy_rhs(x)
        A, b, signed = build_design(g, x, rates, np.zeros(len(x)))
        coefficients = np.r_[
            g.energy.normalized_coefficients(),
            (theta * g.damping_scales.numpy()).ravel(),
        ]
        np.testing.assert_allclose(A @ coefficients, b, atol=1e-12)

    def test_solver_bounds_and_empty(self):
        coef, result = bounded_lstsq(np.eye(3), np.array([-2.0, -1.0, 3.0]), 1)
        np.testing.assert_allclose(coef, [-2.0, 0.0, 3.0], atol=1e-12)
        self.assertTrue(result.success)
        coef, result = bounded_lstsq(np.empty((3, 0)), np.ones(3), 0)
        self.assertEqual(coef.shape, (0,))
        self.assertTrue(result.success)
        with self.assertRaises(ValueError):
            bounded_lstsq(np.array([[np.nan]]), np.ones(1), 1)

    def test_derivative_and_midpoint_adapters(self):
        f, _ = self.field()
        t = np.arange(12.0) * 0.1
        x = np.column_stack((t * t, t * t * t))
        tr = self.trajectory(x, input=t * t)
        captured = []

        def capture(field, state, rates, u):
            captured.append((state.copy(), rates.copy(), u.copy()))
            return (
                np.eye(1, len(field.energy.coef) + 1),
                np.zeros(1),
                len(field.energy.coef),
            )

        def zero(A, b, signed):
            return np.zeros(A.shape[1]), SimpleNamespace(success=True)

        with (
            patch("phsi.regression.build_design", capture),
            patch("phsi.regression.bounded_lstsq", zero),
        ):
            fit_regression(f, [tr], {}, method="derivative")
            fit_regression(f, [tr], {}, method="midpoint")
        np.testing.assert_allclose(captured[0][0], x[1:-1])
        np.testing.assert_allclose(captured[0][1], (x[2:] - x[:-2]) / 0.2)
        np.testing.assert_allclose(captured[1][0], (x[1:] + x[:-1]) / 2)
        np.testing.assert_allclose(captured[1][1], np.diff(x, axis=0) / 0.1)
        # Linear interpolation of known sampled input at actual midpoint times.
        np.testing.assert_allclose(captured[1][2], (t[1:] ** 2 + t[:-1] ** 2) / 2)
        tr = Trajectory(x, t, InputSignal(t, lambda q: q * q), np.ones_like(x) * 7)
        with (
            patch("phsi.regression.build_design", capture),
            patch("phsi.regression.bounded_lstsq", zero),
        ):
            fit_regression(f, [tr], {}, method="midpoint")
            fit_regression(f, [tr], {}, method="derivative")
        np.testing.assert_allclose(captured[-2][2], ((t[1:] + t[:-1]) / 2) ** 2)
        np.testing.assert_allclose(captured[-1][1], 7)

    def test_independent_masks_zero_threshold_and_empty_support(self):
        f, x = self.field(2, "none", DampingLibrary("diagnostic", widths=[0.5]))
        tr = self.trajectory(x, np.zeros_like(x))
        f.energy.mask.zero_()
        f.damping.mask.zero_()
        g, history = fit_regression(f, [tr], {"energy": 0.0, "damping": 0.0})
        self.assertTrue(torch.all(g.energy.mask == 1))
        self.assertTrue(torch.all(g.damping.mask == 1))
        self.assertTrue(torch.all(f.energy.mask == 0))
        g, history = fit_regression(f, [tr], {"energy": 1.0, "damping": 1.0})
        self.assertTrue(torch.all(g.energy.mask == 0))
        self.assertTrue(torch.all(g.damping.mask == 0))
        np.testing.assert_array_equal(g.damping.physical_coefficients(), 0)
        self.assertEqual(len(history), 2)

    def test_independent_component_thresholds_and_zero_damping_scale(self):
        f, x = self.field(2, Polynomial(1))
        tr = self.trajectory(x, np.zeros_like(x), np.ones(len(x)))

        def solution(A, b, signed):
            return np.full(A.shape[1], 0.5), SimpleNamespace(success=True)

        with patch("phsi.regression.bounded_lstsq", solution):
            g, history = fit_regression(
                f, [tr], {"energy": 0.6, "gain": 0.4, "damping": 10}
            )
        self.assertTrue(torch.all(g.energy.mask == 0))
        self.assertTrue(torch.all(g.gain.mask == 1))
        np.testing.assert_allclose(g.damping.physical_coefficients(), 0.5)
        f, x = self.field(2, "none", DampingLibrary("diagnostic", widths=[0.5]))
        f.damping_scales.zero_()
        with patch("phsi.regression.bounded_lstsq", solution):
            g, history = fit_regression(
                f, [self.trajectory(x, np.zeros_like(x))], {"damping": 0.1}
            )
        self.assertTrue(torch.all(g.damping.mask == 0))
        np.testing.assert_array_equal(g.damping.physical_coefficients(), 0)

    def test_nonconvergence_cap_deadline_and_finite(self):
        f, x = self.field(2, "none", degree=6)
        tr = self.trajectory(x, np.zeros_like(x))
        calls = []

        def shrinking(A, b, signed):
            calls.append(len(calls))
            coef = np.ones(A.shape[1])
            coef[0] = 0
            return coef, SimpleNamespace(success=True)

        with patch("phsi.regression.bounded_lstsq", shrinking):
            with self.assertRaises(SupportNonconvergence):
                fit_regression(f, [tr], {"energy": 0.5})
        self.assertEqual(len(calls), 20)
        with self.assertRaises(TimeoutError):
            fit_regression(f, [tr], {}, deadline=time.monotonic() - 1)
        with self.assertRaises(ValueError):
            fit_regression(f, [tr], {"energy": np.nan})
        with self.assertRaises(ValueError):
            build_design(f, x, np.full_like(x, np.inf), np.zeros(len(x)))
        with self.assertRaises(ValueError):
            fit_regression(f, [tr], {}, method="integration_scheme")


if __name__ == "__main__":
    unittest.main()
