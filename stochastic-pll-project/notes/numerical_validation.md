# Numerical validation and execution record

The preflight suite was run before any full-matrix cells were executed. Initial tests failed because required modules were absent (results/run_metadata/preflight_red.txt). One C-interface signature mismatch was caught by Python-versus-compiled-path comparison and fixed before the full runs. The final preflight log records 25 passing tests. A later production audit adds compiled-versus-Python commitment-event agreement, unlocked metric handling, and all-row survival/statistics checks. Synthetic tests are recorded separately and included in the final combined suite.

## Mandatory checks
- Brownian increments: independent NumPy PCG64 Gaussian samples, variance dt; 200,000-increment variance test.
- Deterministic zero-noise: SciPy DOP853 rtol1e-11/atol1e-13; successive Euler refinements exhibit decreasing first-order deterministic endpoint error.
- Additive noise: Euler–Maruyama and the usual Milstein update are identical, verified on the same increments.
- Compiled integrator: agrees with the readable Python integrator on the same path to <2e-13; all reported outputs finite.
- Original initial condition: phi0=asin(.5)+.5, v0=0. Seeds0–4 and42 replay original code; maximum phase differences below1e-15, velocity differences below1.1e-15 (see original_replay.json). There was no mathematical solver bug to correct in the original unit-noise formulation.
- Equilibria: stable/saddle drift zeros and eigenvalue classifications, negative-detuning symmetry, threshold excluded from asymptotically stable classification.
- Slips: temporary boundary crossings are rejected; persistence and negative direction tested; compiled events checked against reference detector on a full recorded-resolution path.
- Statistics: Wilson nonzero upper limit with zero events; independent-row mean CI, bootstrap and Kaplan–Meier restricted mean checks.

## Numerical refinement
1000 baseline and500 stress paths each evaluated at five requested steps plus reference .00015625, T=10. Coarse increments are exact sums of fine increments. Endpoint strong RMS errors decrease monotonically. Empirical log-log slopes are about1.075 baseline and1.083 stress relative to that numerical reference; finite-reference bias can inflate the slopes. These fits support practical first-order behavior on this grid, not an exact-solution asymptotic theorem. Paired mean, second-moment and variance differences and uncertainty are preserved, including weak comparisons whose intervals include zero. All finite fractions equal1. Rare-event count bias at T=200 is not established by these T=10 checks.

## Full local execution
All72 Cartesian grid conditions executed, including9 deterministic controls and63 stochastic ensembles. Grid39009 paths; original-initial acquisition1000; six selected T200 ensembles6000; eighteen a/b/noise sensitivity ensembles9000; numerical convergence1500 independent Brownian paths. Total56509 independent PLL paths; coarse/reference realizations are paired and are not counted as extra independent paths. Base/long/sensitivity/acquisition wall time plus convergence and original replays: 200.13 seconds on this host. Time is observed for this run, not a hardware-independent promise.

A Python3.14 process-pool launch encountered the sandbox's prohibition on forkserver sockets before any cells ran. Execution used two bounded threads; ctypes releases the GIL while the C++ kernel runs and each path has separate state/NumPy seed. No experiment size was reduced. Matplotlib could not write its default user cache and used a temporary cache; chart reproduction sets MPLCONFIGDIR=/tmp/stochastic-pll-mpl. These were infrastructure issues, not failed scientific cohorts.

## Storage and limitations
Raw summaries, independent stream seeds, event records, selected sampled paths, terminal states, configurations, source hashes and wall times remain under results/raw_data/condition_*; convergence paired arrays remain under experiments/convergence/. All exact time points are regenerated from seeds, package versions and kernel source; the retained representative trajectories are sampled for plotting. Event decisions and tail metrics are calculated at every integration step. Native-library math and NumPy versions may change last-bit results across platforms. No calibration to real PLL hardware or inference about a physical oscillator's noise intensity is claimed.
