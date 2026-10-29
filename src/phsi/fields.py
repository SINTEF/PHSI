"""
Model components and state derivatives for port-Hamiltonian systems.

Structured neural energy and damping build on pseudo-Hamiltonian neural networks
(https://github.com/SINTEF/pseudo-hamiltonian-neural-networks); analytic energies
build on pseudo-Hamiltonian system identification
(https://github.com/SINTEF/PHSI/tree/legacy_phsi). See docs/attribution.md and
docs/source-provenance.json for attributions to external sources.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .libraries import (
    CustomLibrary,
    DampingLibrary,
    Neural,
    library_from_configuration,
    resolve_library,
)


def _tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _inverse_softplus(value):
    return value + torch.log(-torch.expm1(-value))


class AnalyticComponent(nn.Module):
    kind = "analytic"

    def __init__(self, library, scale):
        super().__init__()
        self.library = library
        self.register_buffer("scale", _tensor(scale).clone())
        if self.scale.shape != (library.n_features,) or not bool(
            torch.isfinite(self.scale).all() & (self.scale > 0).all()
        ):
            raise ValueError("Each analytic feature requires a positive finite scale.")
        self.register_buffer("mask", torch.ones_like(self.scale))
        self.coef = nn.Parameter(torch.zeros_like(self.scale))

    def effective(self):
        return self.coef * self.mask / self.scale

    def forward(self, x):
        return self.library.values(x) @ self.effective()

    def grad(self, x, create_graph=True):
        return torch.einsum(
            "...ki,k->...i", self.library.gradients(x), self.effective()
        )

    def physical_coefficients(self):
        return self.effective().detach().cpu().numpy()

    def normalized_coefficients(self):
        return (self.coef * self.mask).detach().cpu().numpy()

    def set_physical(self, values):
        values = _tensor(values).to(self.coef)
        if values.shape != self.coef.shape or not bool(torch.isfinite(values).all()):
            raise ValueError(
                "Physical coefficients must be finite and match the library."
            )
        with torch.no_grad():
            self.coef.copy_(values * self.scale)

    def configuration(self):
        return dict(
            kind=self.kind,
            library=copy.deepcopy(self.library.specification),
            scale=self.scale.tolist(),
        )


class PositiveDamping(nn.Module):
    def __init__(self, n_dof, library=None):
        super().__init__()
        self.kind = "constant" if library is None else "analytic"
        self.library = library
        shape = (n_dof,) if library is None else (n_dof, library.n_features)
        initial = torch.full(shape, 0.001, dtype=torch.float64)
        if library is None:
            initial.fill_(0.1)
        elif library.n_features:
            initial[:, 0] = 0.1
        self.raw = nn.Parameter(_inverse_softplus(initial))
        if library is not None:
            self.register_buffer("mask", torch.ones_like(initial))

    def theta(self):
        values = functional.softplus(self.raw)
        return values if self.library is None else values * self.mask

    def features(self, p):
        return (
            torch.ones_like(p).unsqueeze(-1)
            if self.library is None
            else self.library.values(p.unsqueeze(-1))
        )

    def forward(self, p):
        if self.library is None:
            return self.theta().expand(p.shape)
        return (self.features(p) * self.theta()).sum(-1)

    def physical_coefficients(self):
        return self.theta().detach().cpu().numpy()

    def set_physical(self, values):
        values = _tensor(values).to(self.raw)
        if (
            values.shape != self.raw.shape
            or not bool(torch.isfinite(values).all())
            or bool((values < 0).any())
        ):
            raise ValueError(
                "Damping coefficients must be nonnegative, finite and match its shape."
            )
        with torch.no_grad():
            self.raw.copy_(_inverse_softplus(values))

    def configuration(self):
        return dict(
            kind=self.kind,
            library=None
            if self.library is None
            else copy.deepcopy(self.library.specification),
        )


def _network(n_input, hidden):
    layers = []
    for width in hidden:
        layers += [nn.Linear(n_input, width), nn.Tanh()]
        n_input = width
    layers.append(nn.Linear(n_input, 1))
    return nn.Sequential(*layers)


class NeuralComponent(nn.Module):
    kind = "nn"

    def __init__(self, sigma, hidden, role, module=None, key=None):
        super().__init__()
        self.role, self.hidden, self.key = role, tuple(hidden), key
        self.register_buffer("sigma", _tensor(sigma).clone())
        if module is not None:
            self.net = module
        elif role == "damping":
            self.net = nn.ModuleList([_network(1, hidden) for _ in sigma])
            for net in self.net:
                nn.init.zeros_(net[-1].weight)
                nn.init.constant_(net[-1].bias, float(_inverse_softplus(_tensor(0.1))))
        else:
            self.net = _network(len(sigma), hidden)
            if role == "gain":
                nn.init.zeros_(self.net[-1].weight)
                nn.init.ones_(self.net[-1].bias)
        self.double()

    def forward(self, x):
        if self.role == "damping" and self.key is None:
            return torch.stack(
                [
                    functional.softplus(
                        net(x[..., i : i + 1] / self.sigma[i]).squeeze(-1)
                    )
                    for i, net in enumerate(self.net)
                ],
                -1,
            )
        output = self.net(x / self.sigma)
        if self.role == "damping":
            if output.shape != x.shape:
                raise ValueError("Custom damping module must return (..., n_dof).")
            return functional.softplus(output)
        if output.shape != x.shape[:-1] + (1,):
            raise ValueError("Custom energy/gain module must return (..., 1).")
        return output.squeeze(-1)

    def grad(self, x, create_graph=True):
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            return torch.autograd.grad(self(x).sum(), x, create_graph=create_graph)[0]

    def configuration(self):
        return dict(
            kind=self.kind,
            role=self.role,
            hidden=list(self.hidden),
            sigma=self.sigma.tolist(),
            key=self.key,
        )


class FixedGain(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        return x.new_full(x.shape[:-1], 1.0 if self.kind == "known" else 0.0)

    def configuration(self):
        return dict(kind=self.kind)


class Field(nn.Module):
    def __init__(
        self, energy, damping, gain, sigma, input_index, input_argument, state_names
    ):
        super().__init__()
        self.energy, self.damping, self.gain = energy, damping, gain
        self.register_buffer("sigma", _tensor(sigma).clone())
        self.n_state = len(sigma)
        self.n_dof = self.n_state // 2
        self.input_index, self.input_argument = input_index, input_argument
        self.input_kind = gain.kind
        self.state_names = list(state_names)
        self.initialization_seeds = {}
        shape = damping.raw.shape if damping.kind == "analytic" else (self.n_dof, 1)
        self.register_buffer("damping_scales", torch.ones(shape, dtype=torch.float64))
        self.double()

    def energy_gradient(self, x, create_graph=True):
        return self.energy.grad(x, create_graph=create_graph)

    def damping_values(self, x):
        return self.damping(x[..., self.n_dof :])

    def gain_values(self, x):
        return self.gain(x[..., self.input_argument : self.input_argument + 1])

    def forward(self, x, u=0.0, create_graph=True):
        gradient = self.energy_gradient(x, create_graph=create_graph)
        momentum = gradient[..., self.n_dof :]
        dp = -gradient[..., : self.n_dof] - self.damping_values(x) * momentum
        direction = torch.zeros(self.n_dof, dtype=x.dtype, device=x.device)
        direction[self.input_index - self.n_dof] = 1.0
        force = self.gain_values(x) * torch.as_tensor(u, dtype=x.dtype, device=x.device)
        return torch.cat((momentum, dp + force.unsqueeze(-1) * direction), -1)

    def numpy_rhs(self, x, u=0.0):
        """
        Evaluate state derivatives as NumPy arrays for differential-equation solvers.
        """
        x = np.asarray(x, dtype=float)
        if (
            self.energy.kind == "analytic"
            and self.damping.kind in ("constant", "analytic")
            and self.gain.kind in ("known", "none", "analytic")
        ):
            gradient = np.einsum(
                "...ki,k->...i",
                self.energy.library.numpy_gradients(x),
                self.energy.physical_coefficients(),
            )
            p = x[..., self.n_dof :]
            damping = self.damping.physical_coefficients()
            if self.damping.kind == "analytic":
                damping = (
                    self.damping.library.numpy_values(p[..., None]) * damping
                ).sum(-1)
            gain = (
                (
                    self.gain.library.numpy_values(
                        x[..., self.input_argument : self.input_argument + 1]
                    )
                    @ self.gain.physical_coefficients()
                )
                if self.gain.kind == "analytic"
                else float(self.gain.kind == "known")
            )
            out = np.concatenate(
                (
                    gradient[..., self.n_dof :],
                    -gradient[..., : self.n_dof]
                    - damping * gradient[..., self.n_dof :],
                ),
                axis=-1,
            )
            out[..., self.input_index] += gain * np.asarray(u)
            return out
        device = self.sigma.device
        return (
            self(
                torch.as_tensor(x, dtype=torch.float64, device=device),
                u,
                create_graph=False,
            )
            .detach()
            .cpu()
            .numpy()
        )

    def port_output(self, x, create_graph=True):
        return (
            self.gain_values(x)
            * self.energy_gradient(x, create_graph)[..., self.input_index]
        )

    def configuration(self):
        return dict(
            energy=self.energy.configuration(),
            damping=self.damping.configuration(),
            gain=self.gain.configuration(),
            sigma=self.sigma.tolist(),
            input_index=self.input_index,
            input_argument=self.input_argument,
            state_names=self.state_names,
            damping_scales=self.damping_scales.tolist(),
            initialization_seeds=self.initialization_seeds,
        )


def _neural(request, sigma, role):
    hidden = (
        request.hidden
        if request.hidden is not None
        else ((64, 64) if role == "energy" else (16, 16))
    )
    if not isinstance(hidden, (tuple, list)) or any(
        not isinstance(w, int) or w < 1 for w in hidden
    ):
        raise ValueError("Neural hidden widths must be positive integers.")
    if request.module is not None and (
        not isinstance(request.module, nn.Module) or not request.key
    ):
        raise ValueError(
            "A custom neural module requires a torch module and reconstruction key."
        )
    return NeuralComponent(sigma, hidden, role, request.module, request.key)


def build_field(
    hamiltonian,
    damping,
    input,
    *,
    sigma,
    x_train,
    u_train,
    input_index=None,
    input_argument=None,
    state_names=None,
    seed=0,
    initialization_seeds=None,
):
    sigma = np.asarray(sigma, dtype=float)
    n = sigma.size
    if (
        sigma.ndim != 1
        or n < 2
        or n % 2
        or not np.all(np.isfinite(sigma) & (sigma > 0))
    ):
        raise ValueError(
            "sigma must contain a positive finite scale for each of an even number of states."
        )
    index = n - 1 if input_index is None else input_index
    argument = 0 if input_argument is None else input_argument
    if not isinstance(index, int) or not n // 2 <= index < n:
        raise ValueError(
            "forced_state must select a momentum coordinate in the second state block."
        )
    if not isinstance(argument, int) or not 0 <= argument < n:
        raise ValueError("gain_state must select a valid state coordinate.")
    names = (
        list(state_names)
        if state_names is not None
        else [f"q{i + 1}" for i in range(n // 2)] + [f"p{i + 1}" for i in range(n // 2)]
    )
    if len(names) != n or any(not isinstance(name, str) for name in names):
        raise ValueError("state_names must contain one string per state coordinate.")
    raw_x = _tensor(x_train)
    if raw_x.ndim < 1 or raw_x.shape[-1] != n:
        raise ValueError("Training states must have one final axis matching sigma.")
    x = raw_x.reshape(-1, n)
    if not x.numel() or not bool(torch.isfinite(x).all()):
        raise ValueError("Raw training states must be nonempty and finite.")
    learned = not isinstance(input, str) or input not in ("known", "none")
    # Named streams preserve source initialization for learned gains; no global RNG changes.
    streams = initialization_seeds or {}
    energy_seed = int(streams.get("energy", seed))
    gain_seed = int(
        streams.get("gain", np.random.SeedSequence([seed, 17027]).generate_state(1)[0])
    )
    damping_seed = (
        int(
            streams.get(
                "damping", np.random.SeedSequence([seed, 17011]).generate_state(1)[0]
            )
        )
        if learned
        else seed
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(energy_seed)
        if isinstance(hamiltonian, Neural):
            energy = _neural(hamiltonian, sigma, "energy")
        else:
            library = resolve_library(hamiltonian, n, x)
            gradients = library.gradients(x)
            canonical = torch.cat(
                (gradients[..., n // 2 :], -gradients[..., : n // 2]), -1
            ) / _tensor(sigma)
            scale = canonical.square().mean((0, 2)).sqrt().clamp(min=1e-6)
            energy = AnalyticComponent(library, scale)
            coefficients = np.zeros(library.n_features)
            for k, e in enumerate(library.exponents.tolist()):
                if sum(e) == 2 and max(e) == 2:
                    coefficients[k] = 0.5
            normalized = coefficients * scale.numpy()
            count = len(library.exponents) if learned else library.n_features
            normalized[:count] += np.random.default_rng([energy_seed, 7]).normal(
                0, 0.001, count
            )
            with torch.no_grad():
                energy.coef.copy_(_tensor(normalized))
        if learned:
            torch.manual_seed(damping_seed)
        if isinstance(damping, str) and damping == "constant":
            dissipation = PositiveDamping(n // 2)
        elif isinstance(damping, Neural):
            dissipation = _neural(damping, sigma[n // 2 :], "damping")
        elif isinstance(damping, (DampingLibrary, CustomLibrary)):
            dissipation = PositiveDamping(
                n // 2,
                resolve_library(
                    damping, 1, x[:, n // 2 :].reshape(-1, 1), damping=True
                ),
            )
        else:
            raise ValueError(
                "Damping must be constant, Neural, DampingLibrary or a nonnegative CustomLibrary."
            )
        if not learned:
            gain = FixedGain(input)
        elif isinstance(input, Neural):
            torch.manual_seed(gain_seed)
            gain = _neural(input, [sigma[argument]], "gain")
        else:
            library = resolve_library(
                input, 1, x[:, argument : argument + 1], constant=True
            )
            u = _tensor(u_train).reshape(-1)
            if u.shape != x.shape[:1] or not bool(torch.isfinite(u).all()):
                raise ValueError(
                    "Gain scaling requires one finite raw training input per state."
                )
            weighted = (
                library.values(x[:, argument : argument + 1])
                * u[:, None]
                / sigma[index]
            )
            gain = AnalyticComponent(
                library, weighted.square().mean(0).sqrt().clamp(min=1e-6)
            )
            if len(library.exponents) and not bool(library.exponents[0].any()):
                with torch.no_grad():
                    gain.coef[0] = gain.scale[0]
    field = Field(energy, dissipation, gain, sigma, index, argument, names)
    field.initialization_seeds = dict(
        energy=energy_seed, damping=damping_seed, gain=gain_seed
    )
    return field


def _restore_component(configuration, role, n_dof, custom_components):
    kind = configuration["kind"]
    if kind == "analytic":
        library = library_from_configuration(
            configuration["library"], custom_components
        )
        return (
            PositiveDamping(n_dof, library)
            if role == "damping"
            else AnalyticComponent(library, configuration["scale"])
        )
    if kind == "constant":
        return PositiveDamping(n_dof)
    if kind in ("known", "none"):
        return FixedGain(kind)
    if kind == "nn":
        key = configuration.get("key")
        module = None
        if key is not None:
            if custom_components is None or key not in custom_components:
                raise ValueError(
                    f"Provide custom_components[{key!r}] to reconstruct this module."
                )
            factory = custom_components[key]
            if isinstance(factory, Neural):
                module = copy.deepcopy(factory.module)
            else:
                module = (
                    copy.deepcopy(factory)
                    if isinstance(factory, nn.Module)
                    else factory()
                )
            if isinstance(module, Neural):
                module = module.module
            if not isinstance(module, nn.Module):
                raise ValueError(
                    "A custom neural reconstruction must supply a torch module or factory."
                )
        return NeuralComponent(
            configuration["sigma"], configuration["hidden"], role, module, key
        )
    raise ValueError(f"Unsupported saved {role} component kind: {kind}")


def field_from_configuration(configuration, custom_components=None):
    """
    Build model components from saved settings.
    """
    cfg = copy.deepcopy(configuration)
    n_dof = len(cfg["sigma"]) // 2
    field = Field(
        _restore_component(cfg["energy"], "energy", n_dof, custom_components),
        _restore_component(cfg["damping"], "damping", n_dof, custom_components),
        _restore_component(cfg["gain"], "gain", n_dof, custom_components),
        cfg["sigma"],
        cfg["input_index"],
        cfg["input_argument"],
        cfg["state_names"],
    )
    field.initialization_seeds = cfg.get("initialization_seeds", {})
    with torch.no_grad():
        field.damping_scales.copy_(_tensor(cfg["damping_scales"]))
    return field
