# Port-Hamiltonian system identification

> **Paper results.** This branch preserves the version 1.0.0 package for *Comparing model representations and fitting methods for port-Hamiltonian system identification*, with the paper experiment settings in [`reproduce/`](reproduce/README.md). It will not track later development on `main`, but may be updated for revisions before publication. The accompanying documentation records which campaign settings are included and what is needed to run them. Run their configuration tests with `python -m pytest reproduce/tests`.

The Python package `phsi-models` (imported as `phsi`) learns port-Hamiltonian models from observed trajectories. Either analytic functions or neural networks can be fitted to represent the Hamiltonian function $H$, damping $R$ and forcing gain $G$. With external input $u$ and canonical interconnection matrix $J$, this gives ordinary differential equations in port-Hamiltonian form:

$$\dot x=(J-R(x))\nabla H(x)+G(x)u(t).$$

The package supports fully observed canonical coordinates, diagonal damping and one external input. The models can be fitted with gradient-based optimization using either an integration-scheme loss or a simulation-error loss, or, for fully analytic models, sparse regression using estimated derivatives or midpoint observations.

## Installation

`phsi-models` requires Python 3.11 or newer. Install with pip:

```bash
python -m pip install phsi-models
```

In your own uv project:

```bash
uv add phsi-models
uv run your_script.py
```

From a cloned repository, run `uv run your_script.py` at the repository root. For notebooks, use `uv run --extra examples jupyter lab`.

Version 1.0.0 is the first stable release. Python code uses `import phsi`.

## Quickstart

### Set up the model

```python
import phsi

model = phsi.model(
    hamiltonian="analytic",
    damping="constant",
    forcing_gain="analytic",
    forced_state=1,  # the external input acts on the momentum equation
    gain_state=0,  # its gain depends on position
)
```

This example uses states ordered as `(position, momentum)` and learns polynomial energy and forcing gain. Use `"nn"` for a neural network model instead of an analytic expression. For analytic damping, use `phsi.DampingLibrary(kind="polynomial")` or `phsi.DampingLibrary(kind="gaussian")`. See the [method notes](https://github.com/SINTEF/PHSI/blob/main/docs/methods.md) for component choices and fitting options.

### Fit observations

```python
model.fit(X, t=t, u=u, method="integration_scheme")
```

Rows of `X` are observed states, `t` contains equally spaced observation times, and `u` is the applied force or voltage at those times. Lists of trajectories are also accepted. The `method` options are `"integration_scheme"`, `"simulation_error"`, `"derivative"` and `"midpoint"`; the latter two require fully analytic models. Unless separate validation data are supplied, the final 20% of each trajectory is reserved for validation.

### Print, predict, save and restore

```python
model.print()

prediction = model.simulate(x0, t=t_new, u=u_new)

model.save("model.phsi")
restored = phsi.load("model.phsi")
```

Printing shows the fitted equations, with analytic coefficients and neural network functions. Simulation starts from `x0` with the new external input `u_new`.

## Agent guide

This is a deliberately narrow implementation. The agent guide helps users and coding agents extend it, for example by adding more candidate functions for the analytic models, editing the neural network architectures, or testing the methods on different data. See [AGENTS.md](https://github.com/SINTEF/PHSI/blob/main/AGENTS.md) for the guide.

## Sources

This package accompanies [*Comparing model representations and fitting methods for port-Hamiltonian system identification*](https://arxiv.org/abs/2611.05031), by Sølve Eidnes and Håkon Noren Myhr. A [Markdown version](https://github.com/SINTEF/PHSI/blob/main/docs/paper.md) is included in this repository.

The package builds on the following code and publications, with adaptations described in the [attribution notes](https://github.com/SINTEF/PHSI/blob/main/docs/attribution.md):

- Holmsen, Eidnes and Riemer-Sørensen: [Pseudo-Hamiltonian system identification](https://doi.org/10.3934/jcd.2024001) ([code](https://github.com/SINTEF/PHSI/tree/legacy_phsi)).
- Eidnes and coauthors: [Pseudo-Hamiltonian neural networks with state-dependent external forces](https://doi.org/10.1016/j.physd.2023.133673) ([code](https://github.com/SINTEF/pseudo-hamiltonian-neural-networks)).
- Meltzer, Eidnes and Stasik: [Learning partially observed systems with neural Hamiltonian ordinary differential equations](https://arxiv.org/abs/2605.23510) ([code](https://github.com/sunnivameltzer/nhode)).
- Myhr, Eidnes and Kutz: [SIPHy: Sparse identification of port-Hamiltonian systems from noisy data](https://arxiv.org/abs/2609.20399) (code not yet public).

Licensed under the [MIT license](https://github.com/SINTEF/PHSI/blob/main/LICENSE).
