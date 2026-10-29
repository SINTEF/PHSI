"""
Tests for example data generation and notebook structure.
"""

import importlib.util
from pathlib import Path

import nbformat
import numpy as np
import pytest

import phsi
from phsi.fields import build_field

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "example_systems", ROOT / "examples" / "_systems.py"
)
systems = importlib.util.module_from_spec(spec)
spec.loader.exec_module(systems)


def settings(case="two_mass", variant="constant_unknown_gain"):
    mechanics = case == "two_mass"
    physics = (
        dict(
            m1=1.0,
            m2=1.2,
            k0=1.0,
            k1=1.5,
            b0=0.5,
            b1=0.3,
            c=[0.15, 0.20],
            a=[0.25, 0.20],
            s=[0.5, 1.0],
            gain_scale=0.5,
        )
        if mechanics
        else dict(
            capacitance=1.0,
            resistance=0.15,
            c=[0.15],
            current_scale=1.0,
            flux_scale=0.7,
            gain_scale=0.75,
        )
    )
    return dict(
        case=case,
        variant=variant,
        seed=17,
        n_train=4,
        n_val=2,
        n_test=2,
        duration=0.1,
        dt=0.05,
        noise_fraction=0.0,
        initial_ranges=[0.75] * 4 if mechanics else [1.0, 1.5],
        amplitude=0.3,
        omegas=list(systems.OMEGAS),
        physics=physics,
    )


@pytest.mark.parametrize("case", ["two_mass", "electric_circuit"])
def test_reference_gradient_and_energy_balance(case):
    cfg = settings(case)
    x = np.array([0.2, -0.3, 0.4, -0.1] if case == "two_mass" else [0.2, -0.3])
    gradient = systems.energy_gradient(x, case, cfg["physics"])
    epsilon = 1e-6
    finite_difference = np.array(
        [
            (
                systems.energy(x + epsilon * e, case, cfg["physics"])
                - systems.energy(x - epsilon * e, case, cfg["physics"])
            )
            / (2 * epsilon)
            for e in np.eye(len(x))
        ]
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-7, atol=1e-9)

    def signal(t):
        return 0.4

    field = systems.rhs(0.0, x, signal, case, cfg["physics"], cfg["variant"])
    momentum_gradient = gradient[len(x) // 2 :]
    expected = (
        -(
            systems.true_damping(x[len(x) // 2 :], case, cfg["physics"])
            * momentum_gradient**2
        ).sum()
        + systems.true_gain(x, case, cfg["physics"]) * gradient[-1] * 0.4
    )
    np.testing.assert_allclose(gradient @ field, expected, atol=1e-12)


@pytest.mark.parametrize("case", ["two_mass", "electric_circuit"])
def test_seeded_pairs_and_heldout_streams(case):
    cfg = settings(case)
    first, second = systems.generate_dataset(cfg), systems.generate_dataset(cfg)
    for split in ("train", "val", "test"):
        data = first[split]
        np.testing.assert_array_equal(data["clean"], second[split]["clean"])
        np.testing.assert_array_equal(data["x0"][::2], data["x0"][1::2])
        np.testing.assert_allclose(data["u"][::2], -data["u"][1::2], atol=1e-14)
        assert (np.abs(data["u"][:, 0]) >= 0.2).all()
        assert np.isfinite(data["clean"]).all()
    assert not np.array_equal(first["train"]["phases"][:2], first["test"]["phases"])
    assert not np.array_equal(first["train"]["x0"][:2], first["test"]["x0"])
    assert "observed" not in first["test"]
    np.testing.assert_array_equal(first["train"]["clean"], first["train"]["observed"])
    cfg["n_train"] = 3
    with pytest.raises(ValueError, match="even"):
        systems.generate_dataset(cfg)


def test_noise_uses_only_clean_training_scale_and_preserves_truth():
    cfg = settings()
    clean = systems.generate_dataset(cfg)
    cfg["noise_fraction"] = 0.05
    noisy = systems.generate_dataset(cfg)
    for stream, split in enumerate(("train", "val")):
        draws = np.random.default_rng([cfg["seed"], 100, stream]).standard_normal(
            noisy[split]["clean"].shape
        )
        np.testing.assert_allclose(
            noisy[split]["observed"] - clean[split]["clean"],
            0.05 * clean["clean_train_std"] * draws,
            atol=1e-15,
        )
    np.testing.assert_array_equal(clean["test"]["clean"], noisy["test"]["clean"])


def test_nonlinear_damping_variant_uses_unit_gain_and_independent_states():
    cfg = settings(variant="state_dependent_known_gain")
    data = systems.generate_dataset(cfg)
    np.testing.assert_array_equal(
        systems.true_gain(
            data["train"]["clean"], cfg["case"], cfg["physics"], cfg["variant"]
        ),
        np.ones((4, 3)),
    )
    assert not np.array_equal(data["train"]["x0"][0], data["train"]["x0"][1])
    np.testing.assert_allclose(
        systems.true_damping(np.zeros(2), cfg["case"], cfg["physics"], cfg["variant"]),
        [0.4, 0.4],
    )


def test_notebook_schema_and_immediate_model_print():
    notebooks = sorted((ROOT / "examples").glob("*.ipynb"))
    assert [p.name for p in notebooks] == ["electric_circuit.ipynb", "two_mass.ipynb"]
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")
                if "model.fit(" in cell.source:
                    assert (
                        notebook.cells[index + 1].source
                        == "model.print(expand=True, decimals=4)"
                    )


def test_circuit_resistance_setting_controls_reference_damping():
    cfg = settings("electric_circuit")
    cfg["physics"]["resistance"] = 0.37
    np.testing.assert_allclose(
        systems.true_damping(np.array([0.2]), cfg["case"], cfg["physics"]), [0.37]
    )


def test_known_gain_variant_refuses_modified_nonunit_truth(monkeypatch):
    monkeypatch.setattr(
        systems, "true_gain", lambda x, *args: np.full(np.asarray(x).shape[:-1], 0.7)
    )
    with pytest.raises(ValueError, match="known unit gain"):
        systems.generate_dataset(settings(variant="state_dependent_known_gain"))


def test_training_observations_plot_saves_every_coordinate_and_input(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = systems.generate_dataset(settings())
    figure = systems.plot_observations(data, ["q1", "q2", "p1", "p2"], tmp_path)
    assert len(figure.axes) == 5
    for coordinate, axis in enumerate(figure.axes[:-1]):
        assert len(axis.lines) == 4
        for trajectory, line in enumerate(axis.lines):
            np.testing.assert_array_equal(
                line.get_ydata(), data["train"]["observed"][trajectory, :, coordinate]
            )
    for trajectory, line in enumerate(figure.axes[-1].lines):
        np.testing.assert_array_equal(line.get_ydata(), data["train"]["u"][trajectory])
    assert (tmp_path / "training_observations.png").is_file()
    arrays = np.load(tmp_path / "training_observations_plot_data.npz")
    np.testing.assert_array_equal(arrays["observed"], data["train"]["observed"])
    np.testing.assert_array_equal(arrays["input_u"], data["train"]["u"])
    np.testing.assert_array_equal(arrays["t"], data["train"]["t"])
    plt.close(figure)


def test_notebooks_show_observations_before_fitting():
    for path in (ROOT / "examples").glob("*.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        cells = [
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "code" and cell.source.strip()
        ]
        observation = next(
            (
                i
                for i, source in enumerate(cells)
                if "plot_observations(dataset," in source
            ),
            None,
        )
        fit = next(i for i, source in enumerate(cells) if "model.fit(" in source)
        assert observation is not None and observation < fit
        assert "display(observation_figure)" in cells[observation]
        assert 'model.save(output / "model.phsi", overwrite=True)' in cells[-1]


@pytest.mark.parametrize("case", ["two_mass", "electric_circuit"])
def test_notebook_save_cell_can_be_repeated(case, tmp_path):
    notebook = nbformat.read(ROOT / "examples" / f"{case}.ipynb", as_version=4)
    save_cell = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "model.save(" in cell.source
    )
    model = phsi.model(hamiltonian="analytic")
    model.field_ = build_field(
        phsi.Polynomial(2),
        "constant",
        "none",
        sigma=np.ones(2),
        x_train=np.array([[0.1, 0.2], [0.3, 0.4], [-0.2, 0.5]]),
        u_train=np.zeros(3),
    )
    model.fit_info_ = {"status": "completed"}
    namespace = {"model": model, "output": tmp_path, "phsi": phsi}
    exec(save_cell, namespace)
    model.field_.damping.set_physical(np.array([0.375123]))
    exec(save_cell, namespace)
    np.testing.assert_allclose(namespace["restored"].damping([0.1, 0.2]), [0.375123])
    assert namespace["restored"].equations() == model.equations()
