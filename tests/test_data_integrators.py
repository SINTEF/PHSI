"""
Tests for trajectory preparation, input signals and integration steps.
"""

import unittest

import numpy as np
import torch

from phsi.data import InputSignal, prepare_data
from phsi.integrators import rk4_step, rollout, srk4_derivative


class DataTests(unittest.TestCase):
    def data(self, n=50):
        t = np.arange(n) * 0.1
        return np.column_stack((t, t * t)), t

    def test_split_scaling_and_input_isolation(self):
        x, t = self.data()
        x[40:] = 1e6
        obs = prepare_data(x, t, np.arange(50.0), forced=True)
        self.assertEqual([len(v.x) for v in obs.train], [40])
        self.assertEqual(len(obs.validation[0].x), 10)
        np.testing.assert_allclose(obs.raw_sigma, x[:40].std(axis=0))
        with self.assertRaises(ValueError):
            obs.train[0].signal(t[40])
        with self.assertRaises(ValueError):
            obs.validation[0].signal(t[39])
        self.assertEqual(obs.train[0].u[-1], 39.0)

    def test_signal_interpolation_and_tensor(self):
        s = InputSignal([0.0, 1.0, 2.0], [0.0, 2.0, 1.0])
        np.testing.assert_allclose(s([0.5, 1.5]), [1.0, 1.5])
        tt = torch.tensor([0.5, 1.5], requires_grad=True)
        out = s(tt)
        out.sum().backward()
        np.testing.assert_allclose(tt.grad.numpy(), [2.0, -1.0])
        self.assertEqual(InputSignal([0.0, 1.0], [2.0, 8.0], "previous")(0.9), 2.0)
        self.assertEqual(InputSignal([0.0, 1.0], [2.0, 8.0], "previous")(1.0), 8.0)
        for q in [-0.01, 2.01, np.nan]:
            with self.assertRaises(ValueError):
                s(q)
        self.assertEqual(s(2.0 + 1e-15), 1.0)
        with self.assertRaises(ValueError):
            InputSignal([0.0, 1.0], [0.0, np.nan])
        with self.assertRaises(ValueError):
            InputSignal([0.0, 1.0], None)
        np.testing.assert_array_equal(
            InputSignal([0.0, 1.0], None, forced=False)([0.0, 1.0]), [0.0, 0.0]
        )

    def test_callable_validation(self):
        s = InputSignal([0.0, 1.0], lambda t: t * t)
        np.testing.assert_allclose(s([0.2, 0.4]), [0.04, 0.16])
        np.testing.assert_allclose(
            s(torch.tensor([0.2, 0.4])).numpy(), [0.04, 0.16], rtol=1e-6
        )
        with self.assertRaises(ValueError):
            InputSignal([0.0, 1.0], lambda t: np.nan)(0.2)
        with self.assertRaises(ValueError):
            InputSignal([0.0, 1.0], lambda t: np.zeros(3))(0.2)

    def test_variable_lengths_explicit_validation_and_floor(self):
        a, t = self.data(30)
        b, tb = self.data(40)
        o = prepare_data([a, b], [t, tb * 2], validation=(a, t, None))
        self.assertEqual([len(z.x) for z in o.training], [30, 40])
        self.assertEqual(o.n_state, 2)
        self.assertEqual(len(o.validation), 1)
        o = prepare_data(np.ones((20, 2)), np.arange(20.0))
        np.testing.assert_array_equal(o.sigma, [1e-3, 1e-3])
        np.testing.assert_array_equal(o.raw_sigma, [0.0, 0.0])

    def test_smoothing_uses_only_raw_training(self):
        x, t = self.data(50)
        y = x.copy()
        y[40:] += 1e9
        a = prepare_data(x, t, preprocessing="savgol")
        b = prepare_data(y, t, preprocessing={"window": 11, "order": 3})
        np.testing.assert_allclose(a.training[0].x, b.training[0].x)
        np.testing.assert_allclose(a.training[0].t, t[5:35])
        np.testing.assert_allclose(
            a.training[0].derivative[:, 1], 2 * t[5:35], atol=1e-10
        )
        np.testing.assert_array_equal(b.validation[0].x, y[40:])

    def test_bad_data_and_window_length(self):
        x, t = self.data(20)
        bad = [
            (x[:, :1], t),
            (x, t[::-1]),
            (x, np.arange(20.0) ** 2),
            (x * float("nan"), t),
        ]
        for a, b in bad:
            with self.assertRaises(ValueError):
                prepare_data(a, b)
        with self.assertRaises(ValueError):
            prepare_data(x[:10], t[:10], method="simulation_error")
        with self.assertRaises(ValueError):
            prepare_data(
                x[:15], t[:15], preprocessing="savgol", method="simulation_error"
            )
        with self.assertRaises(ValueError):
            prepare_data(x, t, preprocessing={"window": 12})

    def test_shared_grid_batch_and_scalar_only_callable(self):
        import math

        x, t = self.data(30)
        o = prepare_data(
            np.stack([x, x + 1]),
            t,
            [lambda q: math.sin(q), lambda q: np.cos(q)],
            forced=True,
        )
        self.assertEqual(len(o.train), 2)
        np.testing.assert_allclose(o.training[0].u, np.sin(t[:24]))
        np.testing.assert_allclose(o.training[1].u, np.cos(t[:24]))
        with self.assertRaises(ValueError):
            InputSignal(t, np.ones(len(t)), forced=False)
        with self.assertRaises(ValueError):
            prepare_data(x, t, validation=(x[:, :1], t, None))
        with self.assertRaises(ValueError):
            prepare_data([x, x[:-1]], t)
        with self.assertRaises(ValueError):
            prepare_data(x, t, method="unknown")


class IntegratorTests(unittest.TestCase):
    def test_source_equivalence(self):
        # Literal scalar-step formulas from frozen phcompare/integrators.py.
        # Keep this check portable: do not depend on the private campaign path.
        import math

        def source_symmetric(f, x1, x2, t, dt):
            c = math.sqrt(3.0) / 6.0
            z1, z2 = (0.5 + c) * x1 + (0.5 - c) * x2, (0.5 - c) * x1 + (0.5 + c) * x2
            tm, tp = t + (0.5 - c) * dt, t + (0.5 + c) * dt
            z3, z4 = (
                0.5 * (x1 + x2) - c * dt * f(z2, tp),
                0.5 * (x1 + x2) + c * dt * f(z1, tm),
            )
            return 0.5 * (f(z3, tm) + f(z4, tp))

        def source_rk4(f, x, t, h):
            k1 = f(x, t)
            k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
            k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
            k4 = f(x + h * k3, t + h)
            return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        reference = {"srk4_derivative": source_symmetric, "rk4_step": source_rk4}
        x = torch.tensor([[0.1, -0.2], [0.3, 0.5]], dtype=torch.float64)

        def f(z, t):
            return z * z + t[..., None] if torch.is_tensor(t) and t.ndim else z * z + t

        for name in ("srk4_derivative", "rk4_step"):
            args = (
                (f, x, x + 0.01, torch.tensor(0.2), 0.1)
                if name == "srk4_derivative"
                else (f, x, torch.tensor(0.2), 0.1)
            )
            torch.testing.assert_close(globals()[name](*args), reference[name](*args))

    def test_all_stage_parameter_gradients(self):
        x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
        for kind in ("rk4", "symmetric"):

            def evaluate(a):
                def f(z, t):
                    return a * z * z + t

                return (
                    rk4_step(f, x, 0.2, 0.1)
                    if kind == "rk4"
                    else srk4_derivative(f, x, x + 0.02, 0.2, 0.1)
                ).sum()

            a = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
            (grad,) = torch.autograd.grad(evaluate(a), a)
            eps = 1e-5
            numerical = (evaluate(a.detach() + eps) - evaluate(a.detach() - eps)) / (
                2 * eps
            )
            torch.testing.assert_close(grad, numerical, rtol=1e-7, atol=1e-10)

    def test_vector_steps_and_stage_inputs(self):
        x = torch.zeros((2, 2), dtype=torch.float64)
        h = torch.tensor([0.1, 0.2], dtype=torch.float64)
        t = torch.tensor([0.0, 0.2], dtype=torch.float64)
        signal = InputSignal([0.0, 1.0], [0.0, 1.0])

        def f(z, tt):
            return signal(tt)[..., None].expand_as(z)

        got = rk4_step(f, x, t, h)
        torch.testing.assert_close(got, (t * h + h * h / 2)[:, None].expand_as(x))
        batched = srk4_derivative(f, x, x + 0.1, t, h)
        singles = torch.cat(
            [
                srk4_derivative(f, x[i : i + 1], x[i : i + 1] + 0.1, t[i], h[i])
                for i in range(2)
            ]
        )
        torch.testing.assert_close(batched, singles)

    def test_held_endpoints_and_long_batched_rollout(self):
        signal = InputSignal([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], "previous")

        def f(z, moment):
            return signal(moment)[..., None].expand_as(z)

        x = torch.zeros((1, 2), dtype=torch.float64)
        out = rollout(f, x, 0.0, 1.0, 2, n_sub=1, endpoint_left=True)
        torch.testing.assert_close(
            out[0, :, 0],
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            atol=1e-14,
            rtol=0.0,
        )
        t = np.arange(201) * 0.1
        u = np.arange(len(t)) % 2
        signal = InputSignal(t, u, "previous")
        starts = torch.tensor(t[[0, 11, 37]], dtype=torch.float64)
        initial = torch.zeros((3, 2), dtype=torch.float64)
        out = rollout(f, initial, starts, 0.1, 100, n_sub=3, endpoint_left=True)
        expected = np.stack(
            [
                np.r_[0.0, np.cumsum(u[start : start + 100]) * 0.1]
                for start in (0, 11, 37)
            ]
        )
        np.testing.assert_allclose(out[:, :, 0].numpy(), expected, atol=1e-13, rtol=0.0)

    def test_callable_keeps_actual_runge_kutta_stage_times(self):
        calls = []

        def known_input(moment):
            calls.append(np.asarray(moment).copy())
            return np.asarray(moment) ** 2

        signal = InputSignal([0.0, 1.0, 2.0], known_input, "previous")
        self.assertFalse(signal.is_held)
        calls.clear()

        def f(z, moment):
            return signal(moment).expand_as(z)

        result = rk4_step(f, torch.zeros((1, 2), dtype=torch.float64), 0.0, 1.0)
        np.testing.assert_array_equal(np.asarray(calls), [0.0, 0.5, 0.5, 1.0])
        torch.testing.assert_close(result, torch.full_like(result, 1.0 / 3.0))

    def test_rollout_generic_shape_guard(self):
        x = torch.zeros((2, 6), dtype=torch.float64)
        out = rollout(lambda z, t: torch.ones_like(z), x, 0.0, 0.1, 3)
        self.assertEqual(tuple(out.shape), (2, 4, 6))
        torch.testing.assert_close(out[:, -1], torch.full_like(x, 0.3))
        failed = rollout(
            lambda z, t: torch.ones_like(z) * 1e9, x, 0.0, 0.1, 3, detach=True
        )
        self.assertTrue(torch.isnan(failed[:, 1:]).all())
        with self.assertRaises(ValueError):
            rollout(lambda z, t: z, x, 0.0, 0.1, 2, n_sub=0)


if __name__ == "__main__":
    unittest.main()
