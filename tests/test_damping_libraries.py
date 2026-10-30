"""
Checks for named damping libraries and their public fitting workflows.
"""

import numpy as np
import pytest
import torch

import phsi
from phsi.libraries import resolve_library


def test_polynomial_damping_uses_constant_and_squared_momentum():
    library = resolve_library(phsi.DampingLibrary(kind="polynomial"), 1)
    p = torch.tensor([[-2.0], [0.0], [1.5]], dtype=torch.float64, requires_grad=True)
    expected = torch.cat((torch.ones_like(p), p.square()), dim=-1)
    torch.testing.assert_close(library.values(p), expected)
    np.testing.assert_allclose(
        library.numpy_values(p.detach().numpy()), expected.detach()
    )
    derivative = torch.autograd.grad(library.values(p).sum(), p)[0]
    torch.testing.assert_close(derivative, 2 * p)
    assert library.names(["p"]) == ["1", "p^2"]


def test_gaussian_damping_preserves_main_and_default():
    gaussian = resolve_library(phsi.DampingLibrary(kind="gaussian"), 1)
    for specification in (phsi.DampingLibrary(kind="main"), phsi.DampingLibrary()):
        assert resolve_library(specification, 1).specification == gaussian.specification
    p = np.array([[-2.0], [0.0], [1.5]])
    expected = np.column_stack(
        [np.ones(3), p[:, 0] ** 2]
        + [
            np.exp(-(((p[:, 0] - centre) / width) ** 2))
            for width in (0.3, 0.6)
            for centre in np.linspace(-1.5, 1.5, 7)
        ]
    )
    np.testing.assert_allclose(gaussian.numpy_values(p), expected)
    torch.testing.assert_close(gaussian.values(torch.tensor(p)), torch.tensor(expected))


@pytest.mark.parametrize("kind", ["polynomial", "gaussian"])
@pytest.mark.parametrize(
    "method", ["integration_scheme", "simulation_error", "derivative", "midpoint"]
)
def test_named_damping_fit_print_and_reload(kind, method, tmp_path):
    t = np.linspace(0.0, 0.4, 16)
    x = np.column_stack((np.cos(t), -np.sin(t)))
    model = phsi.model(
        hamiltonian=phsi.Polynomial(2),
        damping=phsi.DampingLibrary(kind=kind),
        forcing_gain="none",
    )
    options = {}
    if method in ("integration_scheme", "simulation_error"):
        options.update(max_steps=1, refit_steps=0)
    model.fit(
        x,
        t=t,
        validation=(x, t, None),
        method=method,
        prune=False,
        progress=False,
        **options,
    )
    count = 2 if kind == "polynomial" else 16
    assert model.field_.damping.library.n_features == count
    assert np.isfinite(model.damping(x)).all()
    assert (model.damping(x) >= 0).all()
    if kind == "polynomial":
        assert "exp(" not in model.equations(expand=True)
    path = tmp_path / "model.phsi"
    model.save(path)
    restored = phsi.load(path)
    np.testing.assert_array_equal(restored.damping(x), model.damping(x))
    assert restored.equations() == model.equations()
    assert (
        restored.field_.damping.library.specification
        == model.field_.damping.library.specification
    )
