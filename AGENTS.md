# Working with the PHSI package

`phsi-models` fits port-Hamiltonian models from observed trajectories, combining analytic functions and neural networks for energy, damping and forcing gain. Import it as `phsi`; `phsi.model(...)` constructs the model, and `model.fit(...)` estimates its parameters.

The current model uses fully observed canonical coordinates: positions followed by momenta, or charges followed by fluxes. It supports any even number of states, diagonal damping on the momentum/flux coordinates, and one applied signal. These assumptions can be extended deliberately, with corresponding changes to the implementation, tests and documentation.

## Start with your task

- For installation and basic use, see the [README](README.md).
- Before changing fitting, integration, normalization or pruning, read the relevant sections of [the method notes](docs/methods.md). They describe current package behavior and its numerical conventions.
- For complete data-to-prediction examples, use the [two-mass](examples/two_mass.ipynb) and [electric-circuit](examples/electric_circuit.ipynb) notebooks.
- For the study's recorded settings, see [paper recipes](reproduce/README.md). Keep those settings distinct from the package's convenience defaults. [Attribution notes](docs/attribution.md) identify the underlying methods and adaptations.

## Find the implementation

| Change | Starting point | Relevant tests |
|---|---|---|
| Observations, trajectory splits or measured inputs | [data.py](src/phsi/data.py) | [Data and integration](tests/test_data_integrators.py) |
| Example physics, signals or noise | [_systems.py](examples/_systems.py) and notebook settings | [Examples](tests/test_examples.py) |
| Analytic functions, networks or model structure | [libraries.py](src/phsi/libraries.py), [fields.py](src/phsi/fields.py) | [Components](tests/test_components.py), [joint components](tests/test_joint_components.py) |
| Fitting objectives or term selection | [fitting.py](src/phsi/fitting.py), [integrators.py](src/phsi/integrators.py), [regression.py](src/phsi/regression.py) | [Fitting](tests/test_fitting.py), [regression](tests/test_regression.py), [data and integration](tests/test_data_integrators.py) |
| Public interface or printed equations | [estimator.py](src/phsi/estimator.py); feature expressions also use [libraries.py](src/phsi/libraries.py) | [Estimator](tests/test_estimator.py), [equations](tests/test_equations.py) |
| Simulation or saved models | [simulation.py](src/phsi/simulation.py), [persistence.py](src/phsi/persistence.py) | [Estimator](tests/test_estimator.py), [public workflows](tests/test_public_workflows.py) |

## Extend the package

Start with existing settings: component choices, library degree, network architecture, fitting method and observation preprocessing. Changing an example's physical law belongs in its reference system; fitting should receive only observations and applied signals.

For new analytic features, use `CustomLibrary(values, gradients, names, key)`; for a custom neural component, use `Neural(module=..., key=...)`. Their definitions in `libraries.py` document the interfaces. Use differentiable PyTorch operations and follow the documented array shapes. Custom damping functions must be nonnegative. When loading, supply matching components through `phsi.load(..., custom_components={key: component})`.

Check feature values and derivatives, conversion to physical coefficients, equation display, and saving/loading. For changes to the model structure or fitting method, update the corresponding tests and method notes together. Multiple forcing channels and partially observed states need changes to the model and data handling; they are not existing configuration options. Alternating fitting is not implemented.

## Important behavior to preserve

- `u` is the applied signal; `forcing_gain` describes its multiplier. Currently, `forcing_gain="known"` means unit gain on the selected coordinate.
- Keep trajectory boundaries intact. Derive normalization and data-dependent features from raw training observations; use validation data for model selection and reserve test data for evaluation.
- Gradient-based fitting differentiates through every integration stage. Derivative and midpoint regression require analytic components.
- Preserve independent, complete pruning searches and valid empty supports. Neural weights and separately estimated constant damping are not pruned. Details are in [term selection](docs/methods.md#function-libraries-and-term-selection).
- Display coefficients and equations in physical coordinates. Neural input normalization must remain consistent with derivatives in those coordinates.
- Simulation errors carry partial predictions and diagnostics; do not present them as completed trajectories. Saved models support later evaluation and simulation, but loading does not resume optimizer state.

## Check a change

Use a Python 3.11-or-newer virtual environment. From the repository root, install development dependencies with:

```bash
python -m pip install -e '.[dev,examples]'
```

Alternatively, use `uv pip install -e '.[dev,examples]'` in a uv environment. Run affected tests while editing; for shared code changes, run the full suite and formatting checks:

```bash
python -m pytest
ruff check .
ruff format --check .
```

Build package archives with `python -m build` when changing packaging. Follow the configured Ruff style and use standalone opening and closing triple-quote lines for docstrings.

Keep changes focused. Reproduce bugs with a failing test before fixing them. Use small fixtures for correctness checks; claims about identification accuracy need saved data and inspected plots on held-out trajectories. Preserve source attributions and upstream notices. Keep generated model files and example outputs out of commits unless deliberately included as small test fixtures.
