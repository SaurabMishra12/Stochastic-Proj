# Audit of the supplied PLL project

Audit date: 13 November 2026. The audit describes the supplied implementation before the new experiment suite; later results belong in the executed-results report and numerical validation record.

## Evidence and preservation

The supplied working tree contains a Python program, `baseline_1.py`, a short README and two generated PNG illustrations (`stochastic_pll_dynamics.png` and `stochastic_pll_enhanced.png`). These are source and illustrative output, not an empirical dataset or a Monte Carlo research archive. No measured hardware dataset accompanies the model. Simulated paths generated for this project are synthetic observations from the stated SDE.

The original script is preserved at the workspace root and copied into `backups/baseline_1.py`. Direct SHA-256 verification gave the same digest for both:

`3b4fbba7f33dc84b46ee9db54dedce150c4d4d013fd5096168b4fa0abdad590e`.

The original images and README are also copied into `backups/`. New modules and outputs are written under `stochastic-pll-project/`, so the supplied script is available for exact replay. Source preservation must be checked again when the final manifest is written.

The parent audit confirmed the codebase graph project `stochastic-proj`, ready generation `2026-11-13T08:07:01Z`. The graph found the three functions `simulate_pll`, `plot_pll_dynamics`, and `main`; both call directions relevant to the simulator were inspected without truncation. Coverage reported no recorded gaps for the candidate Python and README paths. This is a coverage record, not a mathematical proof of discovery completeness. The entire 230-line supplied Python file was also read directly, including external NumPy and Matplotlib calls that are incompletely represented in the graph. The two supplied figures were not treated as code coverage.

## What the existing program computes

`simulate_pll(a,b,detuning,noise,T,dt,phi0,v0,seed)` integrates

\[
d\phi=v\,dt,\qquad
dv=\frac{\Delta-v-ab\sin\phi}{a}\,dt-\frac{\alpha}{a}\,dW_t.
\]

It checks positivity of `a`, `b`, `T`, and `dt` and nonnegativity of the noise amplitude. It creates `n_steps=round(T/dt)` time steps and uses NumPy's `default_rng(seed)`. Its independent increments are `sqrt(dt)*standard_normal(n_steps)`. Both updates use the state at the beginning of the step. Consequently, the supplied recurrence is a valid explicit Euler–Maruyama recurrence under its explicitly documented standard-Wiener assumption. There is no missing factor of `dt` in the stochastic increment, and the supplied negative sign agrees with textbook equation (10.4).

For `|Delta/(ab)|<1`, the script takes `phi_star=asin(Delta/(ab))`, starts at `phi_star+0.5` when no initial phase is specified, and initializes velocity to zero. The main program runs one deterministic trajectory and five stochastic paths, with seeds 0 through 4. Its figure shows those paths, their mean plus or minus one empirical standard deviation, and a phase portrait. The deterministic control and stochastic runs use the same initial condition. The deterministic random seed is immaterial because its diffusion amplitude is zero.

The figure is useful for illustrating noisy acquisition near one equilibrium. Five paths do not support a precise estimate of a rare-slip rate, a locking probability, a high-confidence noise comparison, or a convergence order. A visually narrow band in the supplied example does not establish permanent locking.

## Mathematical assumptions and textbook check

The supplied p.274 was inspected as an image. In Chapter 7, under “Communications”, equation (10.4) has the second-order form

\[
a\ddot\phi+\dot\phi+ab\sin\phi=\omega-\omega_0-\alpha\zeta_t.
\]

The paragraph defines `a=RC`, relates `b` to the product of the signal amplitudes, describes `alpha` as white-noise intensity, and identifies a cycle slip with a phase change by `2pi`. It says that the equation can be written as a two-dimensional vector Itô SDE. This confirms the model form, its coefficient `ab`, its sign, and its physical context. The specific convention `zeta_t dt=dW_t` with standard Brownian motion is made explicitly by this project; the supplied PLL paragraph does not separately state a covariance normalization for `zeta`.

The required conditions `a>0` and `b>0` describe the intended positive time constant and gain. Phase is an unwrapped real number in the solver, while phase modulo `2pi` is appropriate for circular observables. The equilibrium existence condition `|Delta|<ab` is also the local asymptotic stability range of the stable branch. It is not, by itself, a global capture or pull-in theorem for every initial velocity and phase of a second-order loop.

## Corrections and interpretive hazards

| Item | Evidence in supplied script | Required treatment |
|---|---|---|
| Unlocked “equilibrium” | Sets `phi_star=0` when `|Delta/(ab)|>=1`; plotting labels it a stable lock and attractor | Represent absent stable equilibria as not applicable. A numerical initialization at zero must have a different name. At equality the equilibrium is nonhyperbolic, not asymptotically stable by linearization. |
| Acquisition versus retention | Default phase starts 0.5 rad away from the stable branch | Preserve this acquisition experiment separately from ensembles initialized exactly at the stable phase. They answer different questions. |
| Plot clipping | Fixed phase limits 0.05 to 1.25 rad | Choose limits from the data and display unwrapped excursions or a clearly labeled wrapped view. Fixed limits can conceal slips and detuning effects. |
| Inconsistent time units | Time label says dimensionless while velocity says rad/s | Use model time and rad/model-time unless physical calibration is supplied. |
| Hard-coded annotation | Parameter box always prints alpha=0.15 and dt=10^-3 | Build annotations from the actual configuration. |
| Unsupported transient label | Shading labels the first 8 units “Transient Acquisition” for every run | Label this as an illustrative window or estimate acquisition under an explicitly stated rule. |
| Small ensemble | Only five paths | Execute the preregistered hundreds/thousands of independent paths and report denominators. |
| Uncertainty | Uses population standard deviation with `ddof=0` and labels mean±sigma | Keep path spread distinct from a confidence interval for the mean. Use unbiased sample variance where required; estimate uncertainty across independent trajectories. |
| No event detection | No slip or waiting-time code | Count completed well transitions with an operational detector and sensitivity checks, not every threshold crossing. |
| No convergence checks | One fixed step only | Couple Brownian increments across nested meshes; compare zero-noise results to a high-accuracy ODE solution. |
| Input boundaries | NaN/Inf inputs and nonintegral horizon are not explicitly handled | Reject invalid inputs; document effective horizon or require an integral step count. |
| Storage/provenance | Figure only; no per-path summary, config or environment archive | Preserve raw independent-path rows, seeds, event counts, sampled paths, source hashes and run metadata. |

These are additions and interpretive corrections to an otherwise correct Euler–Maruyama loop, not evidence that the original SDE discretization was mathematically wrong.

## Missing experiments and proposed architecture

The existing script does not investigate noise and detuning systematically, classify the critical threshold, examine long-horizon rare events, compare numerical resolutions, or vary inertia/gain. It has no unit tests, checkpoint system, survival analysis, synthetic network extension, or live LLM evaluation. None of these should be described as previously completed.

The implementation separates responsibilities:

* `pll_model.py` defines drift, diffusion, equilibria, eigenvalues, potential and parameters.
* `sde_solver.py` provides explicit deterministic/EM updates, reproducible increments and a compiled streaming kernel where used. Constant diffusion means its standard Milstein update must coincide with EM.
* `phase_slips.py` defines completed transitions and detector variants.
* `metrics.py` computes independent-trajectory summaries without timestep pseudoreplication.
* `statistical_analysis.py` computes trajectory-level uncertainty, Wilson intervals, bootstrap intervals, and right-censored first-slip summaries.
* `experiment_runner.py` executes configuration-defined conditions with validated checkpoints and seed provenance.
* `visualization.py` and `report_generator.py` derive figures/tables/report content from saved outputs.

The full noise–detuning Cartesian matrix, additional acquisition ensemble, long-horizon slip conditions, coupled convergence ensembles and a/b sensitivity experiments are specified before execution in `experiment_protocol.md`. E1–E4 slices reuse grid cells; their displayed counts must never be added as independent replications. The deterministic alpha=0 cells are controls, not stochastic samples with Monte Carlo confidence intervals.

Synthetic oscillator networks require their own model definition and protocol. An LLM coordination study requires an independently validated representation of disagreement, a full official benchmark evaluation, a cost estimate and explicit user approval before paid calls. Neither extension inherits a physical phase interpretation merely by analogy with the PLL.

## Audit boundaries

This note establishes the baseline implementation and the required corrections. Mathematical derivations are in `mathematical_model.md`; verified source metadata and the limits of source access are in `literature_review.md`. Numerical validation and the generated report must establish which planned experiments actually ran, whether every output is finite, and what statistical conclusions the saved data support. A test pass cannot substitute for an executed experiment, browser verification, or hardware calibration.
