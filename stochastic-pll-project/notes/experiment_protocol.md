# Preregistered local experiment protocol

Recorded before the full matrix, 2026-11-13. Source equation: supplied p.274, (10.4). Time units are model units; no hardware calibration is available. a=b=1 except E7. Master seed 20261113; SeedSequence([master, condition_id, trajectory_id]) generates independent PCG64 streams. Slices reuse conditions and are not independent replications.

## E1-E4
Question: how do noise and normalized detuning affect capture, phase variability and completed slips? Hypothesis: fluctuations and escape increase near the saddle-node; above threshold the observable is rotation.
Noise: [0,.025,.05,.10,.15,.20,.30,.40]. Delta/(ab): [0,.25,.50,.75,.90,.95,.99,1.01,1.05]. Full Cartesian grid, T=50, dt=.001. alpha=0 uses one deterministic path. All 63 stochastic cells have at least 500 independent paths. Use 1000 for delta=.5 with alpha=.1,.15,.4 and for delta=.9,.95,.99 with alpha=.1,.15,.3,.4. Total 39,009 paths including nine deterministic controls. Initial phase = stable phase, v=0; for |delta|>=1 phi=0, v=0. Baseline acquisition from original phi0=asin(.5)+.5, v0=0 is a separate 1000-path ensemble plus exact seed-42 and seed-0..4 original replay.

## E5
1000 paired Brownian paths, T=10, alpha=.15, delta=.5, initial asin(.5)+.5. Steps .01,.005,.0025,.00125,.000625; reference .00015625. Aggregate reference increments exactly. Endpoint state RMS and maximum sampled path error, paired phase/phase-square weak differences, and confidence intervals. Reference is numerical, not exact; fit rates only for resolved strong errors. Test alpha=.4,delta=.95 separately with 500 coupled paths for a nonlinear stress case.

## E6
Primary completed slip: enter the core of an adjacent stable well (distance <=pi/4 from its stable phase) and persist continuously for .5 time units. Direction is the stable-well index change; all intervening boundaries must have been crossed by continuity. Compare radius pi/2,dwell .25 and radius pi/6,dwell 1. Initial occupied well is k=0. Require direction of center displacement consistent with a completed transition, but no velocity threshold at commitment because capture approaches v=0. Cores and vertical saddle crossings are not exact two-dimensional basin boundaries. Approximate local stable manifold of saddle by backward ODE integration; do not use an unvalidated separatrix as detector. Above threshold all slip columns are not applicable. Six independent long ensembles: delta in [.75,.9,.95] crossed with alpha [.15,.3], T=200, n=1000, dt=.001. Long-horizon scope is selected, not a guarantee of resolving very rare events.

## E7
a,b in [.5,1,2] crossed with alpha [.15,.3], Delta=.5 fixed, 500 paths each, T=50,dt=.001. This changes both delta and damping; interpret mechanistically, not as isolated standardized causal effects. No stable phase is assigned when |Delta/(ab)|>=1.

## Metrics and inference
Per path: terminal phi/v, tail mean velocity (t>=.8T), unwrapped tail phase variance, circular deviation from stable phase, signed/absolute committed transitions, first committed time and right-censor indicator; terminal capture = |wrapped phase error|<pi/4 and |tail mean velocity|<.1; retained lock = terminal capture with no committed slips. These are finite-horizon operational outcomes. For unlocked cells report circular coherence and mean rotation, with locking/slips not applicable.
Ensemble phase mean and unbiased variance at sampled times; t-based pointwise 95% mean CIs (not path containment bands). Wilson 95% intervals for binary outcomes. Bootstrap independent path rows (2000 resamples) for rates/variance, explicitly degenerate when no events. Zero-event upper bounds use independent path event probability and a labeled Poisson exposure approximation for rates. First-slip Kaplan-Meier and restricted mean time to T include censoring; conditional observed-event means are not unconditional waiting times. No timestep-level pseudoreplication and no multiplicity-unadjusted significance declarations.

## Storage and execution
Preserve per-trajectory rows, seeds, sampled representative paths, online ensemble statistics, per-cell JSON/config, executable/source hashes, environment, wall times and logs. Skip only hash/config-validated completed checkpoints. Do not pool E2/E3 slices as extra trajectories. Preflight tests gate full runs. Synthetic protocol is separate and recorded before its execution.
