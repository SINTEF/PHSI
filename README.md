# Port-Hamiltonian system identification

The Python package `phsi-models` (imported as `phsi`) learns port-Hamiltonian models from observed trajectories. Either analytic functions or neural networks can be fitted to represent the Hamiltonian function $H$, damping $R$ and forcing gain $G$. With applied signal $u$ and canonical interconnection matrix $J$, this gives ordinary differential equations in port-Hamiltonian form:

$$\dot x=(J-R(x))\nabla H(x)+G(x)u(t).$$

The package supports fully observed canonical coordinates, diagonal damping and one applied signal. The models can be fitted with gradient-based optimization using either an integration-scheme loss or a simulation-error loss, or, for fully analytic models, sparse regression using estimated derivatives or midpoint observations.

## Installation

`phsi-models` requires Python 3.11 or newer. Install the alpha preview with pip:

```bash
python -m pip install 'phsi-models==1.0.0a1'
```

In your own uv project, add the package once, then run your script:

```bash
uv add 'phsi-models==1.0.0a1'
uv run your_script.py
```

In a cloned PHSI repository, run from the repository root; uv installs the local package and its dependencies automatically:

```bash
uv run your_script.py
```

To open the notebooks with their optional dependencies, use `uv run --extra examples jupyter lab`. With pip, install an editable checkout using `python -m pip install -e '.[examples]'`.

This alpha precedes the paper release; the interface may still change.

## Quickstart

### Set up the model

```python
import phsi

model = phsi.model(
    hamiltonian="analytic",
    damping="constant",
    forcing_gain="analytic",
    forced_state=1,  # the signal acts on the momentum equation
    gain_state=0,  # its gain depends on position
)
```

This example uses states ordered as `(position, momentum)` and learns polynomial energy and forcing gain. Use `"nn"` for a neural network model instead of an analytic expression. See the [method notes](docs/methods.md) for component choices and fitting options.

### Fit observations

```python
model.fit(X, t=t, u=u, method="integration_scheme")
```

Fitting bars show the latest batch loss. Set `progress=False` to hide the fitting progress bars.

Rows of `X` are observed states, `t` contains equally spaced observation times, and `u` is the applied force or voltage at those times. Lists of trajectories are also accepted. The `method` options are `"integration_scheme"`, `"simulation_error"`, `"derivative"` and `"midpoint"`; the latter two require fully analytic models. Unless separate validation data are supplied, the final 20% of each trajectory is reserved for validation.

### Print, predict, save and restore

```python
model.print()
prediction = model.simulate(x0, t=t_new, u=u_new)

model.save("model.phsi")
restored = phsi.load("model.phsi")
```

Printing shows the fitted equations, with analytic coefficients and neural network functions. For analytic Hamiltonians, use `model.print(expand=True, decimals=4)` to also show state equations with the Hamiltonian derivatives substituted and coefficients, centres and widths displayed to four decimal places. Simulation starts from `x0` under the new applied signal.

## Agent guide

This is a deliberately narrow implementation. The agent guide helps users and coding agents extend it—for example, by adding analytic candidate functions, changing a neural network architecture or adapting the reference physics. See [AGENTS.md](AGENTS.md) for the code map, extension points and checks to run.

## Sources

This package accompanies the manuscript *Comparing model representations and fitting methods for port-Hamiltonian system identification*, by Sølve Eidnes and Håkon Noren Myhr (in preparation).

The package builds on the following code and publications, with adaptations described in the [attribution notes](docs/attribution.md):

- Holmsen, Eidnes and Riemer-Sørensen: [Pseudo-Hamiltonian system identification](https://doi.org/10.3934/jcd.2024001), [legacy code](https://github.com/SINTEF/PHSI/tree/legacy_phsi).
- Eidnes and coauthors: [Pseudo-Hamiltonian neural networks with state-dependent external forces](https://doi.org/10.1016/j.physd.2023.133673), [code](https://github.com/SINTEF/pseudo-hamiltonian-neural-networks).
- Meltzer, Eidnes and Stasik: [Learning partially observed systems with neural Hamiltonian ordinary differential equations](https://arxiv.org/abs/2605.23510), [code](https://github.com/sunnivameltzer/nhode).
- Myhr, Eidnes and Kutz: [SIPHy: Sparse identification of port-Hamiltonian systems from noisy data](https://arxiv.org/abs/2609.20399), code **not yet public**.

Licensed under the [MIT license](LICENSE); upstream notices are preserved in `third_party/`.
