"""
Run one recorded paper setting on a newly supplied observation archive.

Dry runs inspect configuration only and never import the fitting package.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def load_presets():
    path = Path(__file__).with_name("presets.json")
    presets = json.loads(path.read_text())
    for identity, recipe in presets["recipes"].items():
        model = presets["models"][recipe["model"]]
        settings = presets["fit_settings"][recipe["fit"]]
        analytic = sum(
            isinstance(model[key], dict) and model[key]["factory"] != "Neural"
            for key in ("hamiltonian", "damping", "input")
        )
        expected = 5**analytic if analytic else 0
        if expected != recipe["candidate_count"]:
            raise ValueError(
                f"{identity}: recorded candidate count does not match its representation."
            )
        if settings["time_limit"] <= 0 or settings["method"] not in (
            "integration_scheme",
            "simulation_error",
            "derivative",
            "midpoint",
        ):
            raise ValueError(f"{identity}: invalid method or allowance.")
        if (
            settings["method"] in ("integration_scheme", "simulation_error")
            and recipe["initialization"]["index"] != 0
        ):
            raise ValueError(
                f"{identity}: the paper recipe must use its recorded first initialization."
            )
        if any(
            len(recipe[key]) != 64
            for key in ("resolved_config_sha256", "original_config_sha256")
        ):
            raise ValueError(f"{identity}: missing source configuration identity.")
    if len(presets["recipes"]) != sum(
        source["row_count"] for source in presets["sources"].values()
    ):
        raise ValueError("Recipe count differs from the source-manifest accounting.")
    return presets


def component(specification):
    from phsi import (
        DampingLibrary,
        GaussianExtension,
        LogCoshExtension,
        Neural,
        Polynomial,
    )

    if isinstance(specification, str):
        return specification
    kwargs = {key: value for key, value in specification.items() if key != "factory"}
    name = specification["factory"]
    if name == "Polynomial":
        return Polynomial(**kwargs)
    if name == "GaussianExtension":
        return GaussianExtension(**kwargs)
    if name == "LogCoshExtension":
        return LogCoshExtension(**kwargs)
    if name == "DampingLibrary":
        return DampingLibrary(**kwargs)
    if name == "Neural":
        return Neural(**kwargs)
    raise ValueError(f"Unsupported component factory: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recipe", nargs="?", help="Original job identity listed by --list."
    )
    parser.add_argument(
        "--list", action="store_true", help="List recorded recipe identities."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check configuration without loading observations or fitting.",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        help="NumPy archive with explicit train and validation observations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="A fresh output directory for fit records and the model.",
    )
    args = parser.parse_args(argv)
    presets = load_presets()
    if args.list:
        print("\n".join(presets["recipes"]))
        return
    if args.recipe is None:
        if not args.dry_run:
            parser.error(
                "Select one recipe, use --list, or check all recipes with --dry-run."
            )
        print(
            json.dumps(
                dict(
                    status="configuration checked; no fits run",
                    recipes=len(presets["recipes"]),
                    models=len(presets["models"]),
                    fit_settings=len(presets["fit_settings"]),
                )
            )
        )
        return
    if args.recipe not in presets["recipes"]:
        parser.error("Unknown recipe identity; use --list.")
    recipe = presets["recipes"][args.recipe]
    settings = dict(presets["fit_settings"][recipe["fit"]])
    if settings["refit_time_limit"] == 0:
        settings["refit_time_limit"] = None
    config = dict(
        recipe_id=args.recipe,
        identity=recipe,
        model=presets["models"][recipe["model"]],
        fit=settings,
        conventions=presets["conventions"],
        source=presets["sources"][recipe["source"]],
    )
    if args.dry_run:
        print(json.dumps(config, indent=2, allow_nan=False))
        return
    if args.observations is None or args.output_dir is None:
        parser.error("A real run requires --observations and --output-dir.")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist.")
    import numpy as np
    import torch

    import phsi

    # Access only these six observation arrays; no held-out or reference key is read.
    required = ("X", "t", "u", "X_validation", "t_validation", "u_validation")
    with np.load(args.observations, allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"Missing observation arrays: {sorted(missing)}")
        arrays = {key: archive[key] for key in required}
    expected = len(config["model"]["state_names"])
    if (
        arrays["X"].shape[-1] != expected
        or arrays["X_validation"].shape[-1] != expected
    ):
        raise ValueError(
            f"This recipe expects {expected} state coordinates in its recorded order."
        )
    torch.set_num_threads(presets["conventions"]["adam"]["threads"])
    model_kwargs = dict(config["model"])
    for key in ("hamiltonian", "damping", "input"):
        model_kwargs[key] = component(model_kwargs[key])
    for source, public in (
        ("input", "forcing_gain"),
        ("input_index", "forced_state"),
        ("input_argument", "gain_state"),
    ):
        model_kwargs[public] = model_kwargs.pop(source)
    estimator = phsi.model(**model_kwargs)
    config["supplied_observations_sha256"] = hashlib.sha256(
        args.observations.read_bytes()
    ).hexdigest()
    record = json.dumps(config, indent=2, allow_nan=False) + "\n"
    sidecar = args.output_dir.with_name(args.output_dir.name + ".recipe.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation keeps the intended recipe even when the fit fails.
    with sidecar.open("x") as target:
        target.write(record)
        target.flush()
        os.fsync(target.fileno())
    try:
        # The package owns all fitting, independent pruning, timing and selection.
        estimator.fit(
            arrays["X"],
            arrays["t"],
            u=arrays["u"],
            validation=(
                arrays["X_validation"],
                arrays["t_validation"],
                arrays["u_validation"],
            ),
            output_dir=args.output_dir,
            progress=False,
            **settings,
        )
    except Exception:
        print(f"Recipe identity preserved at {sidecar}")
        if (args.output_dir / "fit.json").exists():
            print(f"Fit record: {args.output_dir / 'fit.json'}")
        raise
    (args.output_dir / "recipe.json").write_text(record)
    estimator.save(args.output_dir / "model.phsi")


if __name__ == "__main__":
    main()
