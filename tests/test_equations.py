"""
Check printed equations against the model's computed state derivatives.
"""

import contextlib
import copy
import io
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import phsi
from phsi.fields import build_field


def displayed(text, x, model, u=0.0):
    names = model.field_.state_names
    namespace = dict(zip(names, x))
    namespace.update(
        exp=np.exp, tanh=np.tanh, logcosh=lambda z: np.logaddexp(z, -z) - np.log(2), u=u
    )
    lines = dict(line.split(" = ", 1) for line in text.splitlines() if " = " in line)
    rates = [
        eval(lines[f"d{name}/dt"].replace("^", "**"), {"__builtins__": {}}, namespace)
        for name in names
    ]
    rates = np.array(rates)
    n = len(names) // 2
    # Recover the energy gradient from the displayed canonical state equations.
    gradient = np.r_[
        -rates[n:] - model.damping(x) * rates[:n] + model.input_map(x)[n:] * u,
        rates[:n],
    ]
    return gradient, rates


class EquationTests(unittest.TestCase):
    def fixture(self, specification, dimension=2, damping=None, gain=None):
        x = np.random.default_rng(42).normal(size=(20, dimension))
        field = build_field(
            specification,
            damping or phsi.DampingLibrary("diagnostic", widths=[0.37]),
            gain or phsi.Polynomial(2),
            sigma=np.linspace(0.7, 2.3, dimension),
            x_train=x,
            u_train=np.linspace(-1.0, 1.0, len(x)),
            input_index=dimension - 2 if dimension > 2 else 1,
            input_argument=1 if dimension > 2 else 0,
            state_names=[f"x{i + 1}" for i in range(dimension)],
        )
        if field.energy.kind == "analytic":
            field.energy.set_physical(
                np.linspace(-0.13, 0.27, field.energy.library.n_features)
            )
            field.energy.mask[::4] = 0
        field.gain.set_physical(
            np.resize([-0.3, 0.17, -0.04], field.gain.library.n_features)
        )
        field.damping.set_physical(
            np.full_like(field.damping.physical_coefficients(), 0.03)
        )
        model = phsi.Model()
        model.field_ = field.eval()
        model.fit_info_ = {"status": "completed", "selected": {"kind": "fixture"}}
        return model

    def test_polynomial_and_smooth_derivatives_in_physical_units(self):
        for dimension in (2, 4, 6):
            for library in (
                phsi.Polynomial(3),
                phsi.GaussianExtension(2),
                phsi.LogCoshExtension(
                    [{"axis": dimension - 1, "centre": -0.41, "width": 0.73}], 2
                ),
            ):
                with self.subTest(dimension=dimension, library=type(library).__name__):
                    model = self.fixture(library, dimension)
                    text = model.equations(expand=True)
                    x = np.linspace(-0.24, 0.38, dimension)
                    gradient, rates = displayed(text, x, model, 0.23)
                    expected = (
                        model.field_.energy_gradient(
                            torch.tensor(x), create_graph=False
                        )
                        .detach()
                        .numpy()
                    )
                    np.testing.assert_allclose(
                        gradient, expected, rtol=2e-13, atol=2e-14
                    )
                    np.testing.assert_allclose(
                        rates, model.field_.numpy_rhs(x, 0.23), rtol=2e-13, atol=2e-14
                    )

    def test_default_formatter_and_roundtrip(self):
        model = self.fixture(phsi.GaussianExtension(2))
        self.assertEqual(str(model), model.equations())
        self.assertNotIn("Expanded state equations:", str(model))
        self.assertIn("State equations:\n", str(model))
        expanded = model.equations(expand=True)
        self.assertNotIn("Hamiltonian derivatives:", expanded)
        self.assertFalse(any(line.startswith("dH/d") for line in expanded.splitlines()))
        self.assertLess(
            expanded.index("State equations:"),
            expanded.index("Expanded state equations:"),
        )
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            model.print()
        self.assertEqual(stream.getvalue().rstrip(), str(model))
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertIsNone(model.print(expand=True))
        self.assertEqual(stream.getvalue().rstrip(), model.equations(expand=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.phsi"
            model.save(path)
            loaded = phsi.load(path)
            self.assertEqual(
                loaded.equations(expand=True), model.equations(expand=True)
            )

    def test_coefficient_decimals_for_compact_and_expanded_equations(self):
        for damping in ("constant", phsi.DampingLibrary("diagnostic", widths=[0.37])):
            model = self.fixture(phsi.Polynomial(2), damping=damping)
            field = model.field_
            field.energy.mask.fill_(1)
            field.energy.set_physical(
                np.full(field.energy.library.n_features, 0.123456)
            )
            field.damping.set_physical(
                np.full_like(field.damping.physical_coefficients(), 0.345678)
            )
            field.gain.set_physical(np.array([-0.234567, 1.234567, 0.0]))
            before = {name: value.clone() for name, value in field.state_dict().items()}
            original = model.equations(expand=True)
            for expand in (False, True):
                with self.subTest(damping=field.damping.kind, expand=expand):
                    text = model.equations(expand=expand, decimals=4)
                    for coefficient in (
                        "0.1235*",
                        "0.3457",
                        "-0.2346",
                        "1.2346*",
                        "0.0000*",
                    ):
                        self.assertIn(coefficient, text)
                    if expand:
                        self.assertIn("0.2469*", text)
                    with contextlib.redirect_stdout(io.StringIO()) as stream:
                        self.assertIsNone(model.print(expand=expand, decimals=4))
                    self.assertEqual(stream.getvalue().rstrip(), text)
            for decimals in (0, 2, 6, np.int64(4)):
                with self.subTest(decimals=decimals):
                    self.assertIn(
                        f"{1.234567:.{decimals}f}*x1",
                        model.equations(decimals=decimals),
                    )
            self.assertEqual(model.equations(expand=True), original)
            self.assertEqual(model.equations(decimals=None), str(model))
            for name, value in field.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_decimals_include_smooth_feature_parameters(self):
        for library in (
            phsi.GaussianExtension(2),
            phsi.LogCoshExtension(
                [{"axis": 0, "centre": 0.123456789, "width": 0.987654321}], 2
            ),
        ):
            model = self.fixture(
                library,
                damping=phsi.DampingLibrary("diagnostic", widths=[0.345678901]),
                gain=phsi.GaussianExtension(2),
            )
            for component in (model.field_.energy, model.field_.gain):
                component.mask.fill_(1)
                for feature in component.library.features:
                    feature.update(centre=0.123456789, width=0.987654321)
            configuration = copy.deepcopy(model.field_.configuration())
            original = model.equations(expand=True)
            for expand in (False, True):
                for decimals in (2, 4):
                    with self.subTest(
                        library=type(library).__name__, expand=expand, decimals=decimals
                    ):
                        text = model.equations(expand=expand, decimals=decimals)
                        numbers = re.findall(r"(?<![\w.])\d+\.\d+(?:e[+-]?\d+)?", text)
                        self.assertTrue(numbers)
                        incorrect = sorted(
                            {
                                number
                                for number in numbers
                                if len(number.split("e")[0].split(".")[1]) != decimals
                            }
                        )
                        self.assertEqual(incorrect, [])
                        for value in (0.123456789, 0.987654321, 0.345678901):
                            self.assertIn(f"{value:.{decimals}f}", text)
                        with contextlib.redirect_stdout(io.StringIO()) as stream:
                            model.print(expand=expand, decimals=decimals)
                        self.assertEqual(stream.getvalue().rstrip(), text)
            self.assertEqual(model.field_.configuration(), configuration)
            self.assertEqual(model.equations(expand=True), original)

    def test_tiny_feature_widths_do_not_print_as_zero(self):
        model = self.fixture(
            phsi.LogCoshExtension([{"axis": 0, "centre": -2e-9, "width": 1e-9}], 2)
        )
        model.field_.energy.mask.fill_(1)
        text = model.equations(expand=True, decimals=4)
        self.assertIn("-2.0000e-09", text)
        self.assertIn("1.0000e-09", text)
        self.assertNotIn("/(0.0000)", text)
        self.assertNotIn("/0.0000", text)

    def test_decimal_rounding_preserves_tiny_retained_terms(self):
        model = self.fixture(phsi.Polynomial(2))
        model.field_.energy.mask.fill_(1)
        coefficients = np.zeros(model.field_.energy.library.n_features)
        coefficients[:2] = [1e-20, -1e-20]
        model.field_.energy.set_physical(coefficients)
        text = model.equations(expand=True, decimals=4)
        self.assertIn("1.0000e-20*", text)
        self.assertIn(" - 1.0000e-20*", text)
        self.assertIn("0.0000*", text)
        self.assertTrue(bool(model.field_.energy.mask.all()))

    def test_invalid_coefficient_decimals(self):
        for model in (phsi.model(), self.fixture(phsi.Polynomial(2))):
            for invalid in (-1, 1.5, "4", True, np.bool_(False)):
                with self.subTest(value=invalid):
                    with self.assertRaisesRegex(ValueError, "decimals"):
                        model.equations(decimals=invalid)
                    with self.assertRaisesRegex(ValueError, "decimals"):
                        model.print(decimals=invalid)

    def test_empty_support_and_tiny_retained_coefficient(self):
        model = self.fixture(phsi.Polynomial(2))
        model.field_.energy.mask.zero_()
        text = model.equations(expand=True)
        self.assertIn("dx1/dt = 0", text)
        gradient, rates = displayed(text, np.array([0.2, 0.4]), model, 0.5)
        np.testing.assert_array_equal(gradient, 0.0)
        np.testing.assert_allclose(rates, model.field_.numpy_rhs([0.2, 0.4], 0.5))
        model.field_.energy.mask[0] = 1.0
        coefficients = np.zeros(model.field_.energy.library.n_features)
        coefficients[0] = 1e-20
        model.field_.energy.set_physical(coefficients)
        gradient, _ = displayed(model.equations(expand=True), [0.2, 0.4], model)
        self.assertGreater(gradient[0], 0.0)
        self.assertAlmostEqual(gradient[0] / 1e-20, 1.0)

    def test_neural_hamiltonian_keeps_compact_equations(self):
        for dimension in (2, 4, 6):
            model = self.fixture(phsi.Neural(hidden=(3,)), dimension)
            for decimals in (None, 4):
                with self.subTest(dimension=dimension, decimals=decimals):
                    text = model.equations(expand=True, decimals=decimals)
                    self.assertNotIn("Expanded state equations:", text)
                    self.assertNotIn("z = x / sigma", text)
                    self.assertNotIn("dH_nn(z)/dz", text)
                    self.assertIn("State equations:", text)
                    self.assertIn("H_nn(x / sigma", text)
                    self.assertIn("hidden=[3]", text)
                    self.assertNotIn("sigma = ", text)
                    self.assertEqual(text, model.equations(decimals=decimals))
                    with contextlib.redirect_stdout(io.StringIO()) as stream:
                        model.print(expand=True, decimals=decimals)
                    self.assertEqual(stream.getvalue().rstrip(), text)

    def test_custom_derivative_fallback(self):
        custom = phsi.CustomLibrary(
            lambda x: x.square(),
            lambda x: torch.diag_embed(2 * x),
            lambda names: [f"feature_{name}" for name in names],
            "squares",
        )
        model = self.fixture(custom)
        text = model.equations(expand=True)
        self.assertIn("formal derivatives", text)
        self.assertIn("d(feature_x2)/dx1", text)
        self.assertIn("squares", text)


if __name__ == "__main__":
    unittest.main()
