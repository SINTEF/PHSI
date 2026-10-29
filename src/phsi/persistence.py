"""
Save and load model parameters and settings safely.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from .fields import field_from_configuration

SCHEMA = "phsi-model-v1"
PACKAGE_VERSION = "1.0.0a1"


def _plain(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        if any(not isinstance(k, (str, bool, int, float)) for k in value):
            raise ValueError(
                "Checkpoint metadata keys must be plain strings or numbers."
            )
        return {k: _plain(v) for k, v in value.items()}
    raise ValueError(f"Checkpoint metadata cannot contain {type(value).__name__}.")


def _validate_field(field):
    if field.n_state < 2 or field.n_state % 2 or field.n_dof != field.n_state // 2:
        raise ValueError(
            "Saved models require an even positive canonical state dimension."
        )
    if (
        not field.n_dof <= field.input_index < field.n_state
        or not 0 <= field.input_argument < field.n_state
    ):
        raise ValueError("Saved input coordinates are invalid.")
    if len(field.state_names) != field.n_state or any(
        not isinstance(s, str) for s in field.state_names
    ):
        raise ValueError("Saved state names must match the state dimension.")
    for name, tensor in field.state_dict().items():
        valid = torch.isfinite(tensor)
        if name == "damping.raw":
            valid = valid | torch.isneginf(tensor)
        if not bool(valid.all()):
            raise ValueError(f"Saved tensor {name} contains invalid nonfinite values.")
        if name.endswith("mask") and not bool(((tensor == 0) | (tensor == 1)).all()):
            raise ValueError(f"Saved mask {name} must contain only zero and one.")
        if name.endswith("sigma") or name.endswith(".scale"):
            if not bool((tensor > 0).all()):
                raise ValueError(f"Saved scale {name} must be positive.")
        if name == "damping_scales" and not bool((tensor >= 0).all()):
            raise ValueError("Saved damping scales must be nonnegative.")
        if name.endswith("exponents") and not bool((tensor >= 0).all()):
            raise ValueError("Saved polynomial exponents must be nonnegative.")


def save_field(field, path, fit_info=None, *, overwrite=False):
    """
    Save a complete model to disk and return its file checksum.
    """
    _validate_field(field)
    if fit_info is not None and not isinstance(fit_info, dict):
        raise ValueError("fit_info must be a dictionary or None.")
    payload = {
        "schema": SCHEMA,
        "package_version": PACKAGE_VERSION,
        "configuration": _plain(field.configuration()),
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in field.state_dict().items()
        },
        "fit_info": _plain(fit_info),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="." + path.name + ".", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if overwrite:
            os.replace(temporary, path)
            temporary = None
        else:
            # link is atomic and cannot replace an existing destination, including in a race.
            os.link(temporary, path)
            temporary.unlink()
            temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_field(path, custom_components=None):
    """
    Restore a model from saved settings and parameters, checking they match.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema", "package_version", "configuration", "state_dict", "fit_info"}
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema"] != SCHEMA
        or payload["package_version"] not in (PACKAGE_VERSION, "0.1.0")
    ):
        raise ValueError("Unsupported model checkpoint schema or package version.")
    cfg, state = payload["configuration"], payload["state_dict"]
    if (
        not isinstance(cfg, dict)
        or set(cfg)
        != {
            "energy",
            "damping",
            "gain",
            "sigma",
            "input_index",
            "input_argument",
            "state_names",
            "damping_scales",
            "initialization_seeds",
        }
        or not isinstance(state, dict)
    ):
        raise ValueError("Invalid saved model configuration.")
    if payload["fit_info"] is not None and not isinstance(payload["fit_info"], dict):
        raise ValueError("Invalid saved fit information.")
    try:
        field = field_from_configuration(cfg, custom_components)
        expected = field.state_dict()
        if set(state) != set(expected):
            raise ValueError("Saved tensor names do not match the reconstructed model.")
        configuration_buffers = {
            "sigma",
            "damping_scales",
            "energy.scale",
            "gain.scale",
            "energy.sigma",
            "gain.sigma",
            "damping.sigma",
            "energy.library.exponents",
            "gain.library.exponents",
            "damping.library.exponents",
        }
        for name, tensor in state.items():
            if (
                not torch.is_tensor(tensor)
                or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype
            ):
                raise ValueError(f"Saved tensor {name} has an invalid shape or dtype.")
            if name in configuration_buffers and not torch.equal(
                tensor, expected[name]
            ):
                raise ValueError(
                    f"Saved buffer {name} disagrees with the model configuration."
                )
        field.load_state_dict(state, strict=True)
        _validate_field(field)
    except (KeyError, TypeError, RuntimeError, IndexError) as error:
        raise ValueError(f"Invalid model checkpoint: {error}") from error
    return field, payload["fit_info"]
