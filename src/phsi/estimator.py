"""
Create, fit, inspect and simulate models from observed trajectories.
"""

import copy
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .data import prepare_data
from .errors import FitError
from .fields import build_field
from .libraries import DampingLibrary, Neural, Polynomial, _format_number
from .persistence import read_field, save_field
from .simulation import simulate_field


def _request(value, role):
    if isinstance(value, str) and value == "nn":
        return Neural()
    if isinstance(value, str) and value == "analytic":
        return DampingLibrary() if role == "damping" else Polynomial()
    if (
        isinstance(value, str)
        and not (role == "damping" and value == "constant")
        and not (role == "gain" and value in ("none", "known"))
    ):
        raise ValueError(
            f"Unsupported {role} specification {value!r}; supply an explicit library or Neural."
        )
    return value


def _output_directory(output_dir, timed):
    if timed and output_dir is None:
        raise ValueError(
            "Timed fitting requires output_dir for durable checkpoints before computation begins."
        )
    if output_dir is None:
        return
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(
            "Use a new output_dir so existing fit evidence is preserved."
        )
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=".phsi-write-check-", dir=directory.parent
    ) as stream:
        stream.flush()
        os.fsync(stream.fileno())


class Model:
    """
    Port-Hamiltonian model with configurable energy, damping and forcing gain.

    'analytic' selects degree-four polynomials for energy and gain, or nonnegative
    functions for damping. forced_state receives the applied signal u (default:
    last momentum); gain_state selects the coordinate the learned gain depends on.
    State vectors contain positions followed by momenta (charges then fluxes for
    circuits). damping(x) returns one value per momentum; input_map(x) returns the
    vector multiplying the applied signal.
    """

    def __init__(
        self,
        hamiltonian="nn",
        damping="constant",
        forcing_gain="none",
        *,
        forced_state=None,
        gain_state=None,
        state_names=None,
    ):
        self.hamiltonian_spec = copy.deepcopy(hamiltonian)
        self.damping_spec = copy.deepcopy(damping)
        self.input_spec = copy.deepcopy(forcing_gain)
        self.input_index, self.input_argument = forced_state, gain_state
        self.state_names = None if state_names is None else list(state_names)

    def fit(
        self,
        X,
        t,
        u=None,
        *,
        validation=None,
        method="integration_scheme",
        max_steps=1000,
        refit_steps=200,
        learning_rate=0.001,
        validation_every=100,
        batch_size=256,
        window_batch=32,
        window_steps=10,
        n_substeps=2,
        validation_substeps=4,
        seed=0,
        initialization_seeds=None,
        prune=True,
        time_limit=None,
        refit_time_limit=None,
        output_dir=None,
        nonfinite_limit=50,
        terminal_validation=True,
        refit_terminal_validation=True,
        preprocessing=None,
        input_interpolation="linear",
        progress=True,
    ):
        """
        Fit observed trajectories, select by validation error, and return this model.

        Progress bars show initial fitting, candidate models and their refits.
        Fitting and refit bars include the latest batch loss.
        Set progress=False to hide them. Display time counts towards any time limit,
        so slower displays can leave time for fewer fitting updates.
        """
        started = time.monotonic()
        if hasattr(self, "field_"):
            del self.field_
        self.fit_info_ = {"status": "failed", "method": method}
        try:
            for name, limit in (
                ("time_limit", time_limit),
                ("refit_time_limit", refit_time_limit),
            ):
                if limit is not None and (not np.isfinite(limit) or limit <= 0):
                    raise ValueError(f"{name} must be positive and finite.")
            _output_directory(
                output_dir, time_limit is not None or refit_time_limit is not None
            )
            energy, damping, gain = (
                _request(self.hamiltonian_spec, "energy"),
                _request(self.damping_spec, "damping"),
                _request(self.input_spec, "gain"),
            )
            if method in ("derivative", "midpoint") and any(
                isinstance(component, Neural) for component in (energy, damping, gain)
            ):
                raise ValueError(
                    "Derivative and midpoint regression require fixed libraries; neural components need integration_scheme or simulation_error."
                )
            observations = prepare_data(
                X,
                t,
                u,
                validation=validation,
                forced=gain != "none" if isinstance(gain, str) else True,
                interpolation=input_interpolation,
                preprocessing=preprocessing,
                method=method,
                window_steps=window_steps,
            )
            raw_x = np.concatenate([trajectory.x for trajectory in observations.train])
            raw_u = np.concatenate([trajectory.u for trajectory in observations.train])
            field = build_field(
                energy,
                damping,
                gain,
                sigma=observations.sigma,
                x_train=raw_x,
                u_train=raw_u,
                input_index=self.input_index,
                input_argument=self.input_argument,
                state_names=self.state_names,
                seed=seed,
                initialization_seeds=initialization_seeds,
            )
            from .fitting import fit_field

            fitted, info = fit_field(
                field,
                observations,
                method=method,
                max_steps=max_steps,
                refit_steps=refit_steps,
                learning_rate=learning_rate,
                validation_every=validation_every,
                batch_size=batch_size,
                window_batch=window_batch,
                window_steps=window_steps,
                n_substeps=n_substeps,
                validation_substeps=validation_substeps,
                seed=seed,
                prune=prune,
                time_limit=time_limit,
                refit_time_limit=refit_time_limit,
                output_dir=output_dir,
                nonfinite_limit=nonfinite_limit,
                terminal_validation=terminal_validation,
                refit_terminal_validation=refit_terminal_validation,
                started=started,
                progress=progress,
            )
            self.fit_info_ = info
            if info.get("status") != "completed" or info.get("selected") is None:
                raise FitError(
                    "Fitting did not complete a valid validation selection.",
                    fit_info=info,
                    partial_model=fitted,
                )
            self.field_ = fitted.eval()
            return self
        except FitError as error:
            self.fit_info_ = error.fit_info
            raise
        except Exception as error:
            self.fit_info_["error"] = str(error)
            self.fit_info_["error_type"] = type(error).__name__
            raise

    def _field(self):
        if not hasattr(self, "field_"):
            raise RuntimeError("Fit or load a completed model before evaluation.")
        return self.field_

    def _state(self, x):
        field = self._field()
        value = torch.as_tensor(x, dtype=field.sigma.dtype, device=field.sigma.device)
        if (
            value.ndim < 1
            or value.shape[-1] != field.n_state
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                "Evaluation requires finite states with the fitted state dimension."
            )
        return field, value

    @staticmethod
    def _return(value, original):
        return value if torch.is_tensor(original) else value.detach().cpu().numpy()

    def hamiltonian(self, x):
        field, state = self._state(x)
        return self._return(field.energy(state), x)

    def damping(self, x):
        """
        Return nonnegative damping values, one per momentum or flux coordinate.
        """
        field, state = self._state(x)
        return self._return(field.damping_values(state), x)

    def input_map(self, x):
        """
        Return the vector multiplying the applied signal, with the same shape as x.
        """
        field, state = self._state(x)
        direction = torch.zeros(field.n_state, dtype=state.dtype, device=state.device)
        direction[field.input_index] = 1.0
        return self._return(field.gain_values(state).unsqueeze(-1) * direction, x)

    def output(self, x):
        """
        Return the output whose product with the applied signal is input power.
        """
        field, state = self._state(x)
        return self._return(
            field.port_output(state, create_graph=torch.is_tensor(x)), x
        )

    @property
    def coefficients_(self):
        field = self._field()
        return {
            "energy": field.energy.physical_coefficients()
            if field.energy.kind == "analytic"
            else None,
            "damping": field.damping.physical_coefficients()
            if field.damping.kind in ("analytic", "constant")
            else None,
            "gain": field.gain.physical_coefficients()
            if field.gain.kind == "analytic"
            else np.array([float(field.gain.kind == "known")])
            if field.gain.kind in ("known", "none")
            else None,
        }

    @property
    def retained_terms_(self):
        field = self._field()
        result = {}
        for role, component in (("energy", field.energy), ("gain", field.gain)):
            names = (
                field.state_names
                if role == "energy"
                else [field.state_names[field.input_argument]]
            )
            result[role] = (
                {
                    "names": component.library.names(names),
                    "mask": component.mask.detach().cpu().numpy().astype(bool),
                }
                if component.kind == "analytic"
                else {"names": [], "mask": None}
            )
        damping = field.damping
        if damping.kind == "analytic":
            names = [
                damping.library.names([name])
                for name in field.state_names[field.n_dof :]
            ]
            result["damping"] = {
                "names": names,
                "mask": damping.mask.detach().cpu().numpy().astype(bool),
            }
        else:
            result["damping"] = {
                "names": ["1"] * field.n_dof if damping.kind == "constant" else [],
                "mask": None,
            }
        return result

    def simulate(
        self,
        x0,
        t,
        u=None,
        *,
        input_interpolation="linear",
        rtol=1e-9,
        atol=1e-11,
        max_state=1e6,
        max_evaluations=100000,
    ):
        return simulate_field(
            self._field(),
            x0,
            t,
            u,
            input_interpolation=input_interpolation,
            rtol=rtol,
            atol=atol,
            max_state=max_state,
            max_evaluations=max_evaluations,
        )

    def save(self, path, *, overwrite=False):
        return save_field(self._field(), path, self.fit_info_, overwrite=overwrite)

    def print(self, *, expand=False, decimals=None):
        """
        Print state equations; expand=True substitutes analytic Hamiltonian derivatives.

        decimals sets decimal places for coefficients, centres and widths.
        """
        print(self.equations(expand=expand, decimals=decimals))

    def __str__(self):
        return self.equations()

    def equations(self, *, expand=False, decimals=None):
        """
        Return equations in original state units, optionally expanding derivatives.

        expand=True writes out analytic Hamiltonian derivatives. Neural Hamiltonians
        keep compact equations. Custom function names are labels, not formulas,
        so their derivatives stay in derivative notation.

        decimals rounds displayed coefficients, centres and widths. None keeps the
        default precision. Tiny nonzero values use scientific notation instead of zero.
        """
        if decimals is not None and (
            isinstance(decimals, bool)
            or not isinstance(decimals, (int, np.integer))
            or decimals < 0
        ):
            raise ValueError("decimals must be a nonnegative integer or None.")
        if not hasattr(self, "field_"):
            return "Unfitted port-Hamiltonian model"
        field = self.field_
        expand = expand and field.energy.kind == "analytic"
        states = ", ".join(field.state_names)
        precision = 17 if expand else 12
        feature_precision = 17 if expand else 8

        def format_coefficient(value):
            return _format_number(value, precision=precision, decimals=decimals)

        def expression(names, coefficients, mask):
            terms = []
            for name, coefficient, keep in zip(names, coefficients, mask):
                if keep:
                    magnitude = format_coefficient(abs(coefficient))
                    term = magnitude if name == "1" else magnitude + "*" + name
                    terms.append(("-" if coefficient < 0 else "+", term))
            if not terms:
                return "0"
            return (
                ("-" if terms[0][0] == "-" else "")
                + terms[0][1]
                + "".join(" " + sign + " " + term for sign, term in terms[1:])
            )

        def neural(component, label, arguments):
            if component.key is not None:
                architecture = " ".join(repr(component.net).split())
                return f"{label}({arguments}; key={component.key!r}, module={architecture})"
            return f"{label}({arguments}; hidden={list(component.hidden)}, activation=tanh, output=linear)"

        energy = field.energy
        H = (
            expression(
                energy.library.names(
                    field.state_names, precision=feature_precision, decimals=decimals
                ),
                energy.physical_coefficients(),
                energy.mask.detach().cpu().numpy(),
            )
            if energy.kind == "analytic"
            else neural(energy, "H_nn", "x / sigma")
        )
        lines = [f"x = ({states})", f"H({states}) = {H}"]
        diagonal = []
        for i, name in enumerate(field.state_names[field.n_dof :]):
            damping = field.damping
            if damping.kind == "constant":
                diagonal.append(format_coefficient(damping.physical_coefficients()[i]))
            elif damping.kind == "analytic":
                diagonal.append(
                    expression(
                        damping.library.names(
                            [name], precision=feature_precision, decimals=decimals
                        ),
                        damping.physical_coefficients()[i],
                        damping.mask[i].detach().cpu().numpy(),
                    )
                )
            else:
                arguments = (
                    f"[{', '.join(field.state_names[field.n_dof :])}] / sigma[{field.n_dof}:]"
                    if damping.key is not None
                    else f"{name} / sigma[{field.n_dof + i}]"
                )
                diagonal.append(
                    "softplus(" + neural(damping, f"D{i + 1}_nn", arguments) + ")"
                )
        lines.append("R = diag(" + ", ".join(["0"] * field.n_dof + diagonal) + ")")
        gain = field.gain
        if gain.kind == "analytic":
            g = expression(
                gain.library.names(
                    [field.state_names[field.input_argument]],
                    precision=feature_precision,
                    decimals=decimals,
                ),
                gain.physical_coefficients(),
                gain.mask.detach().cpu().numpy(),
            )
        elif gain.kind in ("known", "none"):
            g = "1" if gain.kind == "known" else "0"
        else:
            g = neural(
                gain,
                "g_nn",
                field.state_names[field.input_argument]
                + f" / sigma[{field.input_argument}]",
            )
        vector = ["0"] * field.n_state
        vector[field.input_index] = g
        lines.append(f"G({states}) = [" + ", ".join(vector) + "]")
        lines.append("State equations:")
        for i in range(field.n_dof):
            q, p = field.state_names[i], field.state_names[field.n_dof + i]
            lines.append(f"d{q}/dt = dH/d{p}")
            force = (
                f" + ({g})*u"
                if field.n_dof + i == field.input_index and gain.kind != "none"
                else ""
            )
            lines.append(f"d{p}/dt = -dH/d{q} - ({diagonal[i]})*dH/d{p}" + force)
        if expand:
            coefficients = energy.physical_coefficients()
            mask = energy.mask.detach().cpu().numpy()
            gradients = []
            for axis in range(field.n_state):
                terms = energy.library.derivative_terms(
                    axis, field.state_names, decimals=decimals
                )
                gradients.append(
                    expression(
                        [name for _, _, name in terms],
                        [coefficients[index] * factor for index, factor, _ in terms],
                        [mask[index] for index, _, _ in terms],
                    )
                )
            if energy.library.custom is not None:
                lines.append(
                    f"Custom library {energy.library.custom.key!r}: formal derivatives of feature labels."
                )
            lines.append("Expanded state equations:")
            for i in range(field.n_dof):
                q, p = field.state_names[i], field.state_names[field.n_dof + i]
                dq, dp = gradients[i], gradients[field.n_dof + i]
                lines.append(f"d{q}/dt = {dp}")
                force = (
                    f" + ({g})*u"
                    if field.n_dof + i == field.input_index and gain.kind != "none"
                    else ""
                )
                lines.append(f"d{p}/dt = -({dq}) - ({diagonal[i]})*({dp})" + force)
        return "\n".join(lines)


def model(hamiltonian="nn", damping="constant", forcing_gain="none", **kwargs):
    """
    Configure a new model; call its fit method with measured observations.
    """
    return Model(hamiltonian, damping, forcing_gain, **kwargs)


def load(path, custom_components=None):
    """
    Load saved parameters and settings; supply custom components separately.
    """
    field, info = read_field(path, custom_components)
    result = Model(
        forcing_gain=field.input_kind
        if field.input_kind in ("known", "none")
        else "none",
        forced_state=field.input_index,
        gain_state=field.input_argument,
        state_names=field.state_names,
    )
    result.field_ = field.eval()
    result.fit_info_ = (
        info
        if info is not None
        else {"status": "completed", "selected": {"kind": "loaded"}}
    )
    return result
