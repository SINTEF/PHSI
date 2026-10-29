"""
Analytic function libraries and neural-network settings.

Polynomial energy libraries build on pseudo-Hamiltonian system identification
(https://github.com/SINTEF/PHSI/tree/legacy_phsi). See docs/attribution.md and
docs/source-provenance.json for attributions to external sources.
"""

import copy
import itertools
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _format_number(value, *, precision=17, decimals=None):
    """
    Format a displayed number without hiding small nonzero values.
    """
    value = float(value)
    if decimals is None:
        return f"{value:.{precision}g}"
    rounded = f"{value:.{decimals}f}"
    if value != 0 and float(rounded) == 0:
        return f"{value:.{decimals}e}"
    return rounded


@dataclass
class Polynomial:
    degree: int = 4


@dataclass
class GaussianExtension:
    degree: int = 4


@dataclass
class LogCoshExtension:
    """
    Add log-cosh terms using dictionaries with axis, width and optional centre.
    """

    features: object
    degree: int = 4


@dataclass
class DampingLibrary:
    """
    Default damping functions, or a library with chosen Gaussian widths.

    Custom functions accept coordinates shaped (..., 1); each returns (...,).
    Supply names and a key to identify these functions when reloading.
    """

    kind: str = "main"
    widths: object = None
    functions: object = None
    names: object = None
    key: str = None


@dataclass
class Neural:
    hidden: object = None
    module: object = None
    key: str = None


@dataclass
class CustomLibrary:
    """
    PyTorch functions: values (..., n_features), gradients (..., n_features, dimension).

    n_features counts functions; dimension counts state coordinates. names is a
    sequence or a callable receiving coordinate names. key identifies supplied
    functions when reloading. Damping libraries must declare nonnegative=True.
    """

    values: object
    gradients: object
    names: object
    key: str
    nonnegative: bool = False


class FixedLibrary(nn.Module):
    def __init__(self, specification, custom=None):
        super().__init__()
        self.specification = copy.deepcopy(specification)
        self.custom = custom
        self.dimension = specification["dimension"]
        self.register_buffer(
            "exponents",
            torch.tensor(specification.get("exponents", []), dtype=torch.long).reshape(
                -1, self.dimension
            ),
        )
        self.features = specification.get("features", [])
        self.n_features = specification["n_features"]

    def values(self, x):
        if self.custom is not None:
            result = self.custom.values(x)
        else:
            result = torch.prod(x.unsqueeze(-2) ** self.exponents, dim=-1)
            extras = []
            for f in self.features:
                z = (x[..., f["axis"]] - f["centre"]) / f["width"]
                extras.append(
                    torch.exp(-z.square())
                    if f["kind"] == "gaussian"
                    else torch.logaddexp(z, -z) - np.log(2)
                )
            if extras:
                result = torch.cat((result, torch.stack(extras, -1)), -1)
        if result.shape != x.shape[:-1] + (self.n_features,):
            raise ValueError("Library values must have shape (..., n_features).")
        if self.specification.get("nonnegative") and bool(torch.any(result < 0)):
            raise ValueError("Damping library functions must be nonnegative.")
        return result

    def gradients(self, x):
        if self.custom is not None:
            result = self.custom.gradients(x)
        else:
            columns = []
            for axis in range(self.dimension):
                powers = self.exponents.clone()
                powers[:, axis] = (powers[:, axis] - 1).clamp(min=0)
                columns.append(
                    self.exponents[:, axis] * torch.prod(x.unsqueeze(-2) ** powers, -1)
                )
            result = torch.stack(columns, -1)
            extras = []
            for f in self.features:
                z = (x[..., f["axis"]] - f["centre"]) / f["width"]
                derivative = (
                    -2 * z * torch.exp(-z.square())
                    if f["kind"] == "gaussian"
                    else torch.tanh(z)
                ) / f["width"]
                unit = torch.eye(self.dimension, dtype=x.dtype, device=x.device)[
                    f["axis"]
                ]
                extras.append(derivative.unsqueeze(-1) * unit)
            if extras:
                result = torch.cat((result, torch.stack(extras, -2)), -2)
        if result.shape != x.shape[:-1] + (self.n_features, self.dimension):
            raise ValueError(
                "Library gradients must have shape (..., n_features, dimension)."
            )
        return result

    def numpy_values(self, x):
        """
        Evaluate functions on NumPy arrays, using PyTorch for custom functions.
        """
        x = np.asarray(x, dtype=float)
        if self.custom is not None:
            return (
                self.values(torch.as_tensor(x, dtype=torch.float64))
                .detach()
                .cpu()
                .numpy()
            )
        result = np.prod(x[..., None, :] ** self.exponents.cpu().numpy(), axis=-1)
        extras = []
        for f in self.features:
            z = (x[..., f["axis"]] - f["centre"]) / f["width"]
            extras.append(
                np.exp(-(z**2))
                if f["kind"] == "gaussian"
                else np.logaddexp(z, -z) - np.log(2)
            )
        return np.concatenate((result, np.stack(extras, -1)), -1) if extras else result

    def numpy_gradients(self, x):
        x = np.asarray(x, dtype=float)
        if self.custom is not None:
            return (
                self.gradients(torch.as_tensor(x, dtype=torch.float64))
                .detach()
                .cpu()
                .numpy()
            )
        exponents = self.exponents.cpu().numpy()
        columns = []
        for axis in range(self.dimension):
            powers = exponents.copy()
            powers[:, axis] = np.maximum(powers[:, axis] - 1, 0)
            columns.append(
                exponents[:, axis] * np.prod(x[..., None, :] ** powers, axis=-1)
            )
        result = np.stack(columns, -1)
        extras = []
        for f in self.features:
            z = (x[..., f["axis"]] - f["centre"]) / f["width"]
            derivative = (
                -2 * z * np.exp(-(z**2)) if f["kind"] == "gaussian" else np.tanh(z)
            ) / f["width"]
            extras.append(derivative[..., None] * np.eye(self.dimension)[f["axis"]])
        return np.concatenate((result, np.stack(extras, -2)), -2) if extras else result

    def names(self, state_names=None, *, precision=8, decimals=None):
        variables = list(state_names or [f"x{i + 1}" for i in range(self.dimension)])
        if len(variables) != self.dimension:
            raise ValueError("Coordinate names must match the library dimension.")
        if self.custom is not None:
            return list(
                self.custom.names(variables)
                if callable(self.custom.names)
                else self.custom.names
            )
        result = []
        for e in self.exponents.tolist():
            result.append(
                "*".join(v if k == 1 else f"{v}^{k}" for v, k in zip(variables, e) if k)
                or "1"
            )
        for f in self.features:
            centre = _format_number(f["centre"], precision=precision, decimals=decimals)
            width = _format_number(f["width"], precision=precision, decimals=decimals)
            argument = f"({variables[f['axis']]}-{centre})/{width}"
            result.append(
                f"exp(-({argument})^2)"
                if f["kind"] == "gaussian"
                else f"logcosh({argument})"
            )
        return result

    def derivative_terms(self, axis, state_names, *, decimals=None):
        """
        Return term indices, multipliers and derivative formulas in physical units.

        Built-in functions are differentiated using their saved settings. Custom names
        are labels, not formulas, so their derivatives stay in derivative notation.
        """
        variables = list(state_names)
        if self.custom is not None:
            return [
                (index, 1.0, f"d({name})/d{variables[axis]}")
                for index, name in enumerate(self.names(variables))
            ]
        terms = []
        for index, powers in enumerate(self.exponents.tolist()):
            factor = powers[axis]
            if not factor:
                continue
            powers[axis] -= 1
            name = (
                "*".join(
                    variable if power == 1 else f"{variable}^{power}"
                    for variable, power in zip(variables, powers)
                    if power
                )
                or "1"
            )
            terms.append((index, factor, name))
        for index, feature in enumerate(self.features, start=len(self.exponents)):
            if feature["axis"] != axis:
                continue
            centre, width = (
                _format_number(feature["centre"], decimals=decimals),
                _format_number(feature["width"], decimals=decimals),
            )
            argument = f"(({variables[axis]} - ({centre})) / ({width}))"
            if feature["kind"] == "gaussian":
                factor, name = -2.0, f"{argument}*exp(-{argument}^2)/({width})"
            else:
                factor, name = 1.0, f"tanh({argument})/({width})"
            terms.append((index, factor, name))
        return terms


def _custom_damping(request):
    if not request.key or request.names is None:
        raise ValueError(
            "Custom damping functions require names and a reconstruction key."
        )
    functions = tuple(request.functions)

    def values(x):
        return torch.stack([f(x) for f in functions], -1)

    def gradients(x):
        raise ValueError("Damping-library gradients are not needed by this model.")

    return CustomLibrary(values, gradients, request.names, request.key, True)


def resolve_library(
    request, dimension, training=None, *, constant=False, damping=False
):
    """
    Build a function library from a request or saved settings.
    """
    if isinstance(request, dict):
        return FixedLibrary(request)
    if isinstance(request, DampingLibrary):
        if request.functions is not None:
            request = _custom_damping(request)
        else:
            if request.kind not in ("main", "diagnostic"):
                raise ValueError("Damping library kind must be main or diagnostic.")
            if request.kind == "main":
                features = [
                    dict(kind="gaussian", axis=0, centre=float(c), width=w)
                    for w in (0.3, 0.6)
                    for c in np.linspace(-1.5, 1.5, 7)
                ]
            else:
                widths = (0.25, 0.5, 1.0) if request.widths is None else request.widths
                features = [
                    dict(kind="gaussian", axis=0, centre=0.0, width=float(w))
                    for w in widths
                ]
            spec = dict(
                type="damping",
                dimension=1,
                exponents=[[0], [2]],
                features=features,
                n_features=2 + len(features),
                nonnegative=True,
            )
            _validate_smooth(features, 1)
            return FixedLibrary(spec)
    if isinstance(request, CustomLibrary):
        if (
            not request.key
            or not callable(request.values)
            or not callable(request.gradients)
        ):
            raise ValueError(
                "Custom libraries require callable values, gradients and a stable key."
            )
        if damping and not request.nonnegative:
            raise ValueError("Custom damping must declare nonnegative=True.")
        names = (
            request.names([f"x{i + 1}" for i in range(dimension)])
            if callable(request.names)
            else request.names
        )
        spec = dict(
            type="custom",
            dimension=dimension,
            n_features=len(names),
            key=request.key,
            nonnegative=bool(request.nonnegative),
        )
        library = FixedLibrary(spec, request)
        if training is not None:
            library.values(training)
            if not damping:
                library.gradients(training)
        return library
    if not isinstance(request, (Polynomial, GaussianExtension, LogCoshExtension)):
        raise TypeError(
            "Expected a polynomial, smooth extension, or custom library specification."
        )
    if not isinstance(request.degree, int) or request.degree < 1:
        raise ValueError("Polynomial degree must be a positive integer.")
    exponents = [
        e
        for e in itertools.product(range(request.degree + 1), repeat=dimension)
        if int(not constant) <= sum(e) <= request.degree
    ]
    exponents.sort(key=lambda e: (sum(e), tuple(-v for v in e)))
    features = []
    if isinstance(request, GaussianExtension):
        if training is None:
            raise ValueError("Gaussian features require raw training observations.")
        x = training.detach().cpu().numpy().reshape(-1, dimension)
        for axis in range(dimension):
            low, high = np.quantile(x[:, axis], [0.01, 0.99])
            if high - low <= np.finfo(float).eps:
                raise ValueError(
                    "Training coverage is too small to define Gaussian features."
                )
            for factor in (1.0, 2.0):
                features += [
                    dict(
                        kind="gaussian",
                        axis=axis,
                        centre=float(c),
                        width=float(factor * (high - low) / 6),
                    )
                    for c in np.linspace(low, high, 7)
                ]
    elif isinstance(request, LogCoshExtension):
        features = [
            dict(
                kind="logcosh",
                axis=f["axis"],
                centre=float(f.get("centre", 0)),
                width=float(f["width"]),
            )
            for f in request.features
        ]
    _validate_smooth(features, dimension)
    return FixedLibrary(
        dict(
            type="fixed",
            dimension=dimension,
            exponents=exponents,
            features=features,
            n_features=len(exponents) + len(features),
        )
    )


def _validate_smooth(features, dimension):
    for f in features:
        if (
            not isinstance(f["axis"], int)
            or not 0 <= f["axis"] < dimension
            or not np.isfinite(f["centre"])
            or not np.isfinite(f["width"])
            or f["width"] <= 0
        ):
            raise ValueError(
                "Smooth features need a valid axis, finite centre and positive width."
            )


def library_from_configuration(specification, custom_components=None):
    custom = None
    if specification["type"] == "custom":
        key = specification["key"]
        if custom_components is None or key not in custom_components:
            raise ValueError(
                f"Provide custom_components[{key!r}] to reconstruct this library."
            )
        custom = custom_components[key]
        if isinstance(custom, DampingLibrary):
            custom = _custom_damping(custom)
        if not isinstance(custom, CustomLibrary):
            raise ValueError("The custom reconstruction entry must be a CustomLibrary.")
    return FixedLibrary(specification, custom)
