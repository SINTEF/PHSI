# Paper recipes

These recipes retain the paper's recorded settings and initialization identities for fresh fits on supplied observations. They aim at comparable behavior; they do not replay historical weights or promise identical numerical results.

`presets.json` contains all 246 missing-results and 153 pruning-manifest rows inspected on 28 October 2026. Repeated model definitions and fitting settings are shared. Every recipe keeps its original job identity, source campaign, dataset index, recorded first-initialization identity where applicable, and original/resolved configuration hashes. Initial component hashes are retained where the missing-results manifest recorded them. The pruning manifest did not record those initialization hashes, so its job/campaign/index identity is retained without inventing them. Linear regression does not require a random initialization.

List recipes or check configuration without importing the package, opening observations or starting a fit:

```bash
python reproduce/run.py --list
python reproduce/run.py --dry-run
python reproduce/run.py missing_constant_mechanics_d0_analytic_integration_scheme_noise01_raw_i0 --dry-run
```

For an actual run after installing the package:

```bash
python reproduce/run.py missing_constant_mechanics_d0_analytic_integration_scheme_noise01_raw_i0 \
  --observations measured.npz --output-dir outputs/paper-setting
```

The output directory must be fresh. The observation archive must contain exactly named arrays `X`, `t`, `u`, `X_validation`, `t_validation`, and `u_validation`; additional archive keys are ignored and never opened. `X` and `X_validation` hold states with shape `(samples, states)` or `(trajectories, samples, states)`. Each trajectory is independent. Times and measured scalar inputs follow the public fitting interface; see the main package documentation. Both state arrays must use the recorded coordinate order: positions followed by momenta for the two-mass cases, or charge followed by flux for the circuit. The runner checks that dimension before fitting. Supply raw observations: smoothing is applied by the recipe, separately within each training trajectory. Validation stays raw.

The runner uses the public `phsi.model(...).fit(...)` interface. The fitting package owns the numerical operations, timing, pruning and selection. Model output and a recipe record are saved in the requested directory. Before fitting, an exclusively created sibling `<output-dir>.recipe.json` durably records the recipe and observation-archive identity, so failed fits retain their provenance; an existing sidecar is refused. A recipe's noise label records its paper condition; the runner never generates noise or assumes that newly supplied measurements have the same noise level.

The saved settings preserve the following conventions:

- Adaptive moment estimation (Adam) uses learning rate 0.001, no sparsity penalty, a fixed learning rate, batches of 256 scheme residuals or 32 windows of 10 observation intervals, two fourth-order Runge–Kutta substeps per training interval, and four per validation interval. Validation occurs every 200 updates. Source models use 80,000 or 2,000,000 emergency update caps as recorded, two processor threads, and 50 consecutive nonfinite updates as the failure limit. The recipes preserve separate full-library fitting and candidate-refit limits: 30/90 minutes for full-library fitting and 5/15 minutes per scheme/simulation refit, respectively.
- Derivative and midpoint regression use the bounded variable least-squares solver through SciPy's `lsq_linear`, with tolerance `1e-12`, at most 5,000 solver iterations, no ridge penalty, and at most 20 support updates. The full independent pruning grid has a 50-minute allowance. Validation uses the adaptive Dormand–Prince eighth-order solver with relative tolerance `1e-9` and absolute tolerance `1e-11`.
- Analytic components receive independent threshold choices `0`, `0.0001`, `0.001`, `0.01`, and `0.1`. The complete grid has five choices for one analytic component and 25 for two. Zero retains the whole corresponding library, empty retained sets are permitted, and constant damping scalars and neural weights are not pruned. Each Adam candidate starts from the same selected full model with a fresh optimizer. Source full-library fitting and candidate-refit terminal-validation behavior is recorded separately.
- Savitzky–Golay smoothing, where recorded, uses an 11-point window and degree three. Candidate functions and contribution scales are resolved from raw training observations. The seven Gaussian centres span raw-training 1%–99% quantiles, with widths one and two centre spacings. No historical feature scales are silently reused on new measurements.

The original campaign's pruning rows reused previously fitted full models. This runner instead fits a fresh full model with the recorded original settings before applying the approved pruning protocol. Source minibatch and component seed conventions are preserved, but a new observation archive changes scales, feature locations and training examples. The measured input is interpolated linearly at integration stages; historical benchmark input functions are deliberately outside the runtime. Hardware speed and timed stopping can also change the selected model. These are fresh paper-setting fits, not a claim to reproduce reported table values.

The zero-centred damping library is labelled as a diagnostic because it supplies additional physical information. Gaussian circuit alternatives are explicit recipes. The `LogCoshExtension` factory is supported by the package, but neither inspected manifest contains a log-cosh recipe; none is invented here. Alternating hybrid fitting and constrained-energy studies remain outside these recipes. No numerical experiment or fit was run to prepare this folder; only configuration dry runs were performed.
