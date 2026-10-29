"""
Reference systems and reproducible datasets for the example notebooks.

Equations and paired signals follow the accompanying comparison paper.
The circuit's nonlinear current-flux law and log-cosh energy are adapted from
Castaños, Jayawardhana, Ortega and García-Canseco (2009), "Proportional plus
integral control for set-point regulation of a class of nonlinear RLC circuits",
equations (2.17)-(2.18), https://doi.org/10.1007/s00034-009-9103-x.
Here RLC means resistor-inductor-capacitor. These reference equations generate
observations and test predictions; they are not supplied to the fitted model.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

OMEGAS = (0.5, 0.9, 1.4, 2.1)


def true_gain(x, case, physics, variant="constant_unknown_gain"):
    """
    Reference forcing gain used to generate observations.
    """
    x = np.asarray(x)
    if variant == "state_dependent_known_gain":
        return np.ones(x.shape[:-1])
    if case == "two_mass":
        return 1.0 + 0.5 * np.tanh(x[..., 1] / physics["gain_scale"])
    return 0.5 + 1.0 / (1.0 + (x[..., 0] / physics["gain_scale"]) ** 2)


def true_damping(p, case, physics, variant="constant_unknown_gain"):
    """
    Reference damping used to generate observations.
    """
    p = np.asarray(p)
    c = np.asarray(
        [physics["resistance"]] if case == "electric_circuit" else physics["c"]
    )
    if case == "two_mass" and variant == "state_dependent_known_gain":
        return c + np.asarray(physics["a"]) * np.exp(
            -((p / np.asarray(physics["s"])) ** 2)
        )
    return np.broadcast_to(c, p.shape).copy()


def energy(x, case, physics):
    x = np.asarray(x)
    if case == "electric_circuit":
        q, flux = x[..., 0], x[..., 1]
        z = flux / physics["flux_scale"]
        return q**2 / (2 * physics["capacitance"]) + physics["current_scale"] * physics[
            "flux_scale"
        ] * (np.logaddexp(z, -z) - np.log(2.0))
    q1, q2, p1, p2 = np.moveaxis(x, -1, 0)
    r = q2 - q1
    return (
        p1**2 / (2 * physics["m1"])
        + p2**2 / (2 * physics["m2"])
        + physics["k0"] * q1**2 / 2
        + physics["k1"] * r**2 / 2
        + physics["b0"] * q1**4 / 4
        + physics["b1"] * r**4 / 4
    )


def energy_gradient(x, case, physics):
    x = np.asarray(x, dtype=float)
    if case == "electric_circuit":
        return np.stack(
            (
                x[..., 0] / physics["capacitance"],
                physics["current_scale"] * np.tanh(x[..., 1] / physics["flux_scale"]),
            ),
            axis=-1,
        )
    q1, q2, p1, p2 = np.moveaxis(x, -1, 0)
    r = q2 - q1
    force = physics["k1"] * r + physics["b1"] * r**3
    return np.stack(
        (
            physics["k0"] * q1 + physics["b0"] * q1**3 - force,
            force,
            p1 / physics["m1"],
            p2 / physics["m2"],
        ),
        axis=-1,
    )


def input_signal(phases, amplitude=0.3, omegas=OMEGAS):
    """
    Sum of sine waves, evaluated at one time or an array of times.
    """
    phases = np.array(phases, copy=True)
    frequencies = np.asarray(omegas)

    def signal(t):
        # Torch times occur inside differentiable integration stages. Inputs are
        # prescribed functions of time, so converting fixed times is sufficient.
        if hasattr(t, "detach"):
            t = t.detach().cpu().numpy()
        return amplitude * np.sin(np.asarray(t)[..., None] * frequencies + phases).sum(
            -1
        )

    return signal


def rhs(t, x, signal, case, physics, variant="constant_unknown_gain"):
    gradient = energy_gradient(x, case, physics)
    n = len(x) // 2
    result = np.concatenate(
        (
            gradient[n:],
            -gradient[:n] - true_damping(x[n:], case, physics, variant) * gradient[n:],
        )
    )
    result[-1] += true_gain(x, case, physics, variant) * signal(t)
    return result


def reference_trajectory(x0, times, signal, case, physics, variant):
    solution = solve_ivp(
        lambda t, x: rhs(t, x, signal, case, physics, variant),
        (times[0], times[-1]),
        x0,
        t_eval=times,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if (
        not solution.success
        or solution.y.shape[1] != len(times)
        or not np.isfinite(solution.y).all()
    ):
        raise RuntimeError(f"Reference integration failed: {solution.message}")
    return solution.y.T


def _initial_conditions(settings, count, stream, paired):
    ranges = np.asarray(settings["initial_ranges"], dtype=float)
    states = np.random.default_rng([settings["seed"], stream, 0])
    phases = np.random.default_rng([settings["seed"], stream, 1])
    if paired and (count < 2 or count % 2):
        raise ValueError(
            "Opposite-input pairs require positive even train, validation and test counts."
        )
    size = count // 2 if paired else count
    x0 = states.uniform(-ranges, ranges, size=(size, len(ranges)))
    ph = np.empty((size, len(settings["omegas"])))
    if paired:
        argument = 1 if settings["case"] == "two_mass" else 0
        x0[:, argument] = (
            -ranges[argument]
            + 2 * ranges[argument] * (np.arange(size) + states.random(size)) / size
        )
    for i in range(size):
        for _ in range(100 if paired else 1):
            candidate = phases.uniform(0, 2 * np.pi, len(settings["omegas"]))
            if (
                not paired
                or abs(
                    input_signal(candidate, settings["amplitude"], settings["omegas"])(
                        0.0
                    )
                )
                >= 0.2
            ):
                ph[i] = candidate
                break
        else:
            raise RuntimeError(
                "Could not meet the initial input magnitude within 100 draws."
            )
    if paired:
        order = np.random.default_rng([settings["seed"], stream, 2]).permutation(size)
        x0, ph = np.repeat(x0[order], 2, axis=0), np.repeat(ph[order], 2, axis=0)
        ph[1::2] += np.pi
    return x0, ph


def generate_dataset(settings):
    """
    Generate reproducible training, validation and test data; keep test data noise-free.
    """
    if settings["case"] not in ("two_mass", "electric_circuit"):
        raise ValueError("Choose two_mass or electric_circuit.")
    variant = settings["variant"]
    if variant not in ("constant_unknown_gain", "state_dependent_known_gain"):
        raise ValueError("Unknown physical variant.")
    if settings["case"] == "electric_circuit" and variant != "constant_unknown_gain":
        raise ValueError("The circuit example learns its gain with constant damping.")
    dt, duration = settings["dt"], settings["duration"]
    if dt <= 0 or duration <= 0 or not np.isclose(duration / dt, round(duration / dt)):
        raise ValueError("Use a positive duration that is an integer multiple of dt.")
    if settings["noise_fraction"] < 0:
        raise ValueError("noise_fraction must be nonnegative.")
    ranges = np.asarray(settings["initial_ranges"])
    dimension = 4 if settings["case"] == "two_mass" else 2
    if ranges.shape != (dimension,) or not np.all(np.isfinite(ranges) & (ranges > 0)):
        raise ValueError(
            "Supply one positive finite initial-state range per coordinate."
        )
    times = np.linspace(0, duration, round(duration / dt) + 1)
    dataset = {}
    paired = variant == "constant_unknown_gain"
    for stream, split in enumerate(("train", "val", "test")):
        count = settings[f"n_{split}"]
        if not isinstance(count, int) or count < 1:
            raise ValueError("Each split needs a positive integer trajectory count.")
        x0, phases = _initial_conditions(settings, count, stream, paired)
        signals = [
            input_signal(p, settings["amplitude"], settings["omegas"]) for p in phases
        ]
        clean = np.array(
            [
                reference_trajectory(
                    x, times, signal, settings["case"], settings["physics"], variant
                )
                for x, signal in zip(x0, signals)
            ]
        )
        if variant == "state_dependent_known_gain" and not np.all(
            true_gain(clean, settings["case"], settings["physics"], variant) == 1
        ):
            raise ValueError(
                "This variant requires known unit gain; arbitrary known gains are unsupported."
            )
        dataset[split] = dict(
            t=times.copy(),
            clean=clean,
            x0=x0,
            phases=phases,
            signals=signals,
            u=np.array([s(times) for s in signals]),
        )
    clean_std = dataset["train"]["clean"].reshape(-1, dimension).std(axis=0)
    for stream, split in enumerate(("train", "val")):
        rng = np.random.default_rng([settings["seed"], 100, stream])
        dataset[split]["observed"] = dataset[split]["clean"] + settings[
            "noise_fraction"
        ] * clean_std * rng.standard_normal(dataset[split]["clean"].shape)
    dataset["clean_train_std"] = clean_std
    return dataset


def save_dataset(dataset, settings, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    arrays = {"clean_train_std": dataset["clean_train_std"]}
    for split in ("train", "val", "test"):
        arrays.update(
            {
                f"{split}_{name}": value
                for name, value in dataset[split].items()
                if name != "signals"
            }
        )
    np.savez_compressed(output / "data.npz", **arrays)


def plot_observations(dataset, state_names, output):
    """
    Save and plot observed training states and their applied signals.
    """
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = dataset["train"]
    figure, axes = plt.subplots(
        len(state_names) + 1,
        1,
        sharex=True,
        figsize=(9, 2 * (len(state_names) + 1)),
        constrained_layout=True,
    )
    for trajectory, (observed, applied) in enumerate(
        zip(train["observed"], train["u"])
    ):
        for coordinate, name in enumerate(state_names):
            axes[coordinate].plot(
                train["t"],
                observed[:, coordinate],
                label=f"trajectory {trajectory + 1}",
            )
            axes[coordinate].set_ylabel(name)
        axes[-1].plot(train["t"], applied)
    axes[0].legend(ncol=2)
    axes[-1].set_ylabel("applied input")
    axes[-1].set_xlabel("time")
    figure.suptitle("Training observations and measured inputs")
    figure.savefig(output / "training_observations.png", dpi=130)
    np.savez_compressed(
        output / "training_observations_plot_data.npz",
        t=train["t"],
        observed=train["observed"],
        input_u=train["u"],
    )
    return figure


def plot_predictions(dataset, predictions, state_names, output):
    """
    Save and plot predicted states, applied signals, errors and simulation failures.
    """
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    test = dataset["test"]
    figures = []
    for index, prediction in enumerate(predictions):
        times, states = np.asarray(prediction["t"]), np.asarray(prediction["x"])
        reference = test["clean"][index]
        residual = (
            states
            - np.column_stack(
                [
                    np.interp(times, test["t"], reference[:, j])
                    for j in range(len(state_names))
                ]
            )
            if len(times)
            else np.empty((0, len(state_names)))
        )
        figure, axes = plt.subplots(
            len(state_names) + 2,
            1,
            sharex=True,
            figsize=(9, 2 * (len(state_names) + 2)),
            constrained_layout=True,
        )
        for coordinate, name in enumerate(state_names):
            axes[coordinate].plot(
                test["t"], reference[:, coordinate], label="reference truth"
            )
            if len(times):
                axes[coordinate].plot(
                    times, states[:, coordinate], "--", label="learned model"
                )
            axes[coordinate].set_ylabel(name)
        axes[0].legend()
        axes[-2].plot(test["t"], test["u"][index], color="black")
        axes[-2].set_ylabel("applied input")
        for coordinate, name in enumerate(state_names):
            axes[-1].plot(times, residual[:, coordinate], label=name)
        axes[-1].axhline(0, color="black", linewidth=0.5)
        axes[-1].set_ylabel("model − truth")
        axes[-1].set_xlabel("time")
        axes[-1].legend(ncol=len(state_names))
        title = f"Independent test trajectory {index + 1}"
        if not prediction["success"]:
            title += ": simulation failed; partial prediction only"
        figure.suptitle(title)
        figure.savefig(output / f"test_{index + 1}.png", dpi=130)
        np.savez_compressed(
            output / f"test_{index + 1}_plot_data.npz",
            truth_t=test["t"],
            truth_x=reference,
            input_u=test["u"][index],
            prediction_t=times,
            prediction_x=states,
            residual=residual,
        )
        (output / f"test_{index + 1}_status.json").write_text(
            json.dumps(
                {
                    "success": prediction["success"],
                    "diagnostics": prediction["diagnostics"],
                },
                default=str,
                indent=2,
            )
            + "\n"
        )
        figures.append(figure)
    return figures
