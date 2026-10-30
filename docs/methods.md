# Method notes

This package learns a canonical, fully observed dynamical model by separating energy, damping and measured forcing, then selecting analytic terms using validation predictions.

## Structure and conventions

Let the state be `x=(q,p)`, with equally sized position and momentum blocks. Charge and flux play these roles for the circuit. Let `H(x)` be energy, `D(p)` a diagonal damping matrix with nonnegative entries, `b` a fixed vector selecting one momentum coordinate, `g(z)` the scalar input gain, `z` one specified state coordinate, and `u(t)` the known scalar input. The equations are

$$\dot q=\nabla_p H,\qquad \dot p=-\nabla_q H-D(p)\nabla_p H+b\,g(z)u(t).$$

The gain is not required to be positive. With conjugate output `y=g(z)bᵀ∇ₚH`, the energy rate is `dH/dt=−(∇ₚH)ᵀD(p)∇ₚH+yu`. This balance is a structural identity, not a guarantee of accurate trajectories or bounded states. Analytic Hamiltonians have no imposed global lower bound. A bounded neural energy also does not, by itself, imply bounded state trajectories.

The `"known"` option currently means scalar gain `g=1` on the selected momentum equation; the full map is a selected-coordinate unit vector, not an identity matrix. It does not yet accept an arbitrary prescribed gain. No input means no forcing. Learned forcing gain and state-dependent damping can be fitted together. Hybrid A combines neural energy and analytic gain, hybrid B combines analytic energy and neural gain, and the damping hybrid combines analytic energy and neural damping with known unit gain. Components are fitted jointly; alternating fitting is explicitly deferred. Interconnection, canonical coordinates, full observation and gain dependence are supplied by the user rather than discovered.

## Public component settings

Use `forcing_gain` to choose the multiplier of the applied signal `u`; it is separate from the observed state data. Its choices are `"none"` (unforced), `"known"` (unit gain), `"analytic"` (polynomial gain), `"nn"` (neural gain), or a component specification. `forced_state` selects the state equation receiving the signal, and defaults to the last momentum/flux coordinate. `gain_state` selects the coordinate on which a learned gain depends. Both use zero-based indices. For states `(q1, q2, p1, p2)`, `forced_state=3, gain_state=1` means the signal acts on `p2` with a gain depending on `q2`.

The Hamiltonian accepts `"analytic"`, `"nn"` or a library/network specification. Damping accepts `"constant"`, `"analytic"`, `"nn"` or a specification. Regression requires analytic components. State-dependent damping can be combined with a learned forcing gain, including neural mixtures for the gradient fitters. Alternating fitting remains deferred; supported components are fitted jointly.

## Observation handling

States have shape `(samples, states)`, or `(trajectories, samples, states)` for common-length trajectories. A list can supply differing trajectory lengths. Each trajectory needs uniformly spaced, increasing times and measured inputs. State coordinates remain physical; pooled raw-training standard deviations scale losses and features, with a positive floor for constant coordinates.

Explicit `validation=(states, times, inputs)` keeps the supplied training and validation trajectories separate. Otherwise, the first 80% of each trajectory is training and the last 20% is validation. Splitting precedes smoothing, input restriction and data-dependent feature construction. Test observations must never be passed as validation or used to choose a model.

Sampled input signals use linear or previous-sample interpolation. For example, if measured values are 0 at time 0 and 1 at time 1, linear interpolation supplies 0.5 at time 0.5. Previous-sample hold instead keeps 0 until the next sample. If `u` is a Python function such as `lambda t: np.sin(t)`, the solver calls that function whenever it needs the signal, including between observation times. Such a function is often convenient for a known simulated excitation; the package does not infer the true continuous signal from measurements. Use the same convention in fitting and prediction. Prescribed measured inputs are not inferred external forces.

`preprocessing="savgol"` means an 11-sample cubic Savitzky–Golay local-polynomial smoother, applied independently to training coordinates and trajectories. Five samples are trimmed at each end. Derivative regression differentiates the same polynomial; other methods use the smoothed states. Validation scoring uses unsmoothed observations. These local polynomials are not Hamiltonian flow splines.

## Four fitting methods

The paper groups the four methods into gradient-based optimisation and sparse regression. The public method names correspond to these table labels (Adam means adaptive moment estimation; LS means least squares):

| `fit(method=...)` | Paper label |
|---|---|
| `"integration_scheme"` | Scheme (Adam) |
| `"simulation_error"` | Roll-out (Adam) |
| `"derivative"` | Derivative (LS) |
| `"midpoint"` | Midpoint (LS) |

`integration_scheme` fits the relation between adjacent observations using a symmetric fourth-order Runge–Kutta construction. Every integration-stage state remains in the computational graph, so all stages contribute gradients to all fitted components. This all-stage differentiation is a deliberate change from a legacy stage-detachment convention. It should not be described as an exact implementation of the original symmetric integration loss.

`simulation_error` differentiates through short simulated windows, rather than comparing each pair alone. Both gradient methods use the Adam optimizer, followed by support selection and refitting for analytic components. `max_steps`, `refit_steps` and `learning_rate` apply to these methods.

`derivative` uses centred finite differences on raw data, or derivatives of the optional local-polynomial smoother. `midpoint` uses state increments divided by the sampling interval and averages of adjacent state observations. Both exploit `dq/dt=∇ₚH` to replace the momentum gradient inside damping, giving a joint linear least-squares problem for analytic energy, damping and gain coefficients. Damping coefficients are constrained nonnegative. These fitters cannot contain neural components and do not consume optimizer-only settings.

The midpoint substitution and local derivative estimates are adaptations inspired by sparse identification of port-Hamiltonian systems (SIPHy); they omit that method's Hamiltonian flow splines. The neural trajectory objective is related to neural ordinary differential equation work, but this package requires all state coordinates to be observed.

## Function libraries and term selection

The standard energy library contains nonconstant monomials of total degree at most four. The gain library also includes the constant. Polynomial degree is configurable. `GaussianExtension` adds univariate Gaussian features with seven centres from the 1st through 99th percentiles of raw training coordinates and two widths based on their spacing. Centres and widths are fixed before fitting.

`DampingLibrary(kind="polynomial")` contains a constant and the squared momentum (or flux) for each damped coordinate. `DampingLibrary(kind="gaussian")` adds 14 nonnegative Gaussian features: seven centres from −1.5 to 1.5 at widths 0.3 and 0.6. `kind="main"` remains an alias for `"gaussian"`, preserving the paper's library. Omitting `kind`, or using `damping="analytic"`, also selects this Gaussian-enriched library. The polynomial damping library has fixed degree two, independent of the energy and gain libraries. `DampingLibrary(kind="diagnostic", widths=...)` explicitly adds supplied widths as a diagnostic; true widths are not provided to the main learner. Neural damping uses positive outputs, and analytic damping uses nonnegative coefficients. Constant damping is estimated separately and is not thresholded.

Feature contribution scales are fixed from raw training observations in physical units. Energy scales use its canonical vector field; gain scales include the measured input. Damping contribution scales use position rates estimated from raw observations. Thresholds apply to physical nonnegative damping coefficients, rather than their unconstrained internal parameters.

The independent threshold grid is `{0, 0.0001, 0.001, 0.01, 0.1}` for every analytic energy, gain and state-dependent damping component. Three analytic components give 125 combinations, two give 25, and one gives five. Neural components and constant damping add no threshold axis. The frozen paper recipes retain their original five- or 25-candidate grids. Zero keeps the full library, and empty supports are valid. Analytic terms are removed after fitting the full library with gradients, with each support refitted. Regression repeats constrained solves and simultaneous threshold removal until the support stops changing or reaches the prescribed iteration limit. Validation prediction error selects a candidate; test errors do not select thresholds. The candidate grid and failures belong in the fit information.

Joint fitting does not guarantee that the individual components are uniquely identifiable. Damping and forcing gain can compensate for one another when the applied signal does not vary independently of the state. For example, if the applied signal always equals velocity, equal increases in constant damping and constant gain leave their combined effect unchanged on the forced coordinate. Varied excitation and overlapping state coverage help distinguish them; positivity of damping alone does not remove the ambiguity. This is a data-design issue, not a reason to forbid the model combination.

## Demonstrations and attribution

The two notebooks use independent high-accuracy reference integrations, one seed, and short demonstration fits. Unknown-gain trajectories use paired opposite inputs from identical initial states, binned initial gain coordinates, and initial input magnitude at least 0.2. Separate streams generate training, validation and test phases and initial states. Independent Gaussian measurement noise affects only training and validation, with coordinate scales taken from clean training standard deviations. The learner receives only observations, times and measured signals. The reference equations and clean test truth are retained for plotted comparisons.

The two-mass reference uses masses `(1,1.2)`, spring coefficients `(1,1.5,0.5,0.3)`, and constant damping `(0.15,0.20)` with gain `1+0.5 tanh(q₂/0.5)`. Its alternative uses known unit gain and damping `cᵢ+aᵢ exp[−(pᵢ/sᵢ)²]`, with amplitudes `(0.25,0.20)` and widths `(0.5,1)`. The circuit uses capacitance and current scale 1, resistance 0.15, flux scale 0.7 and gain `0.5+1/[1+(q/0.75)²]`. Finite polynomial libraries cannot exactly represent the circuit energy or either non-polynomial reference gain.

See [attribution notes](attribution.md) for Holmsen and coauthors' original analytic approach, Eidnes and coauthors' neural framework, Meltzer and coauthors' partially observed trajectory fitting, Myhr and coauthors' SIPHy, and Castaños and coauthors' published nonlinear-inductor law. The current code is a scoped adaptation of these ideas, not a claim to implement all features of those papers. [Paper recipes](../reproduce/README.md) describe comparison-scale settings separately from the introductory examples.

See [attribution](attribution.md) and [license status](license-status.md) for source-specific provenance and reuse authorization.

## Progress and expanded equations

`model.fit(..., progress=True)` shows full-library fitting updates, then the complete pruning-candidate search and each gradient refit. Fitting and refit bars display the latest training-batch loss, including alongside validation and termination status. This is the loss used for the latest completed update, not the validation score or a full-dataset average. Regression shows candidate progress. Notebook environments use widgets when installed; terminal runs use text bars. Set `progress=False` for silence. Bars count actual completed updates and resolved candidates, retain partial counts on failure or timeout, and do not change the selection rules. Display overhead counts toward timed allowances. Frozen paper recipes disable progress output.

`model.equations(expand=True)` returns the expanded text, and `model.print(expand=True)` prints it. The default `str(model)` and `model.print()` retain the compact representation under `State equations:`. For an analytic Hamiltonian with `expand=True`, `Expanded state equations:` follows; individual Hamiltonian derivatives are not listed separately. Both example notebooks use `model.print(expand=True, decimals=4)`. Expansion differentiates built-in polynomial, Gaussian and log-cosh energy features in physical coordinates, preserves masks and tiny retained coefficients, and substitutes those derivatives into every state equation. Damping products remain factored. By default, expanded output uses round-trip precision for physical coefficients, centres and widths. Set `decimals=4` in either printing method to display coefficients, centres and widths to four decimal places, including inside exponentials and expanded derivatives. Stored values are unchanged. Tiny nonzero values use scientific notation so retained terms and nonzero widths never appear to vanish.

Neural Hamiltonians keep the compact equations even with `expand=True`; their network architecture is still displayed, while numerical input scales are omitted. The network receives each state divided by its training standard deviation. Dynamics use derivatives with respect to the original states, so automatic differentiation includes the reciprocal scales. Custom feature names are labels, so their derivatives are explicitly marked as formal expressions rather than guessed from those names.
