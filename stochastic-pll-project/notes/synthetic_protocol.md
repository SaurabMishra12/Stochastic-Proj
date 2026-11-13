# Synthetic network extension: protocol and exact numerical semantics

Recorded before full execution, 2026-11-13. This extension is a synthetic stochastic Kuramoto network. It is not derived from the PLL equation and establishes no LLM mechanism. The full matrix is run only after the core PLL experiments and their validation finish.

## Scientific question and design

How do fixed communication topology, differential noise, message loss, and delayed relative-state information change synchronization and disagreement? Hypothesis: dense coupling and reduced differential noise often favor coherence; topology and delay can produce persistent disagreement or nonmonotone behavior. These are hypotheses before seeing outputs, not guaranteed outcomes.

Complete Cartesian matrix:

- Agents N = 3 initially, followed by 8 and 20.
- Topology = complete, ring, star, sparse.
- Noise correlation rho = 0, 0.5, 1.
- Independent directed-message dropout probability p = 0 or 0.3.
- Relative-state message delay tau = 0 or 0.2 model time units.
- Fixed coupling K = 2 and marginal phase-noise intensity sigma = 0.35.
- 500 independent network trajectories per cell, T = 50, dt = 0.01.
- Total: 144 cells, 72,000 network trajectories, 360,000,000 Euler steps, and 3,720,000,000 node updates. Within each N, inputs are matched between conditions; the 72,000 condition-runs are not all statistically independent. There are 1,500 independent matched network-input blocks across the three sizes.

The fixed coupling and frequency range define one operating regime. This is a broad factorial experiment over communication and noise, not a study over coupling strengths or frequency distributions.

## Model and parameter meanings

For node i, original neighbors j and original degree d_i, the numerical target is

    d theta_i(t) = [omega_i + K/d_i sum_j A_ij B_ij(t)
                      sin(theta_j(t-tau) - theta_i(t-tau))] dt
                    + sigma [sqrt(1-rho) dW_i + sqrt(rho) dW_0].

A is a fixed undirected binary adjacency matrix with zero diagonal. B_ij is an independent Bernoulli(1-p) directed reception indicator regenerated at each integration step; this is a discrete communication model at dt=.01, not a timestep-independent continuous jump process. Lost messages contribute zero, and the denominator remains the original degree. Dropout lowers expected coupling to (1-p)K; the experiment therefore combines weakened mean coupling and random interruptions. It does not isolate interruptions at fixed effective coupling.

Delay applies to the complete observed relative error, corresponding to stale relative-state communication. A constant history theta(t)=theta(0) is imposed for t<=0. This differs from the commonly used sender-only delayed term sin(theta_j(t-tau)-theta_i(t)). Conclusions about common-noise cancellation and delay apply to the specified model only. The Brownian drivers are independent across separate networks, and independent W_i are independent of the common W_0. Their increment covariance is dt[(1-rho)I+rho 11^T].

At rho=1, all nodes receive exactly the same additive phase noise. Subtracting any pair of equations cancels the common Brownian term. Because delayed coupling also uses relative phases at the same delayed time, the relative dynamics are independent of sigma at rho=1, even with dropout. Shared noise can rotate the whole network; it does not intrinsically synchronize it. Increasing rho at fixed marginal sigma reduces the differential component sqrt(1-rho), which can improve coherence. This differs from noise-induced synchronization under state-dependent common forcing.

Intrinsic frequencies are one deterministic seed-shuffled evenly spaced set from -0.5 to 0.5 for each N and are fixed across topologies and conditions. Initial phases are independent uniform(-pi,pi) draws within each network. Every frequency vector, initial phase array and topology matrix is saved. One sparse graph is a seeded random spanning tree plus independently added missing undirected edges at probability .05. Graphs are fixed rather than averaged over graph realizations. These choices limit topology generalization.

For N=3 the ring is exactly the complete graph, so those outputs must agree exactly with matched streams; they do not supply independent topology evidence. Some N=3 sparse/star structures may also be graph-isomorphic; fixed node-frequency assignments can break equivalence under relabeling. The saved labeled adjacency matrices are authoritative.

## Discretization, streams and implementation

Euler--Maruyama uses Brownian increments scaled by sqrt(dt). The diffusion matrix is constant, so ordinary Milstein derivatives vanish. The C++ kernel evolves the phase mean separately from centered relative phases. This is algebraically the same update and makes common-noise cancellation exact in relative outputs. The delayed centered state is kept in a circular buffer, avoiding storage of the whole trajectory.

Each network has independent initial-phase, Brownian and dropout seed identities from NumPy SeedSequence([20261113,820,N,replicate,stream]). These seeds are shared across methods at the same N and replicate. Initial phases use NumPy PCG64. Brownian normals use a separate std::mt19937_64 and std::normal_distribution inside each network's own C++ loop; no RNG is shared between OpenMP threads. Exact replay depends on the saved compiler and C++ standard library implementation because the normal transform is not portable across standard libraries. Dropout is a counter-based SplitMix64 transform keyed by the per-network dropout seed, timestep and directed node pair. This gives the same reception event for a given edge across topologies where meaningful. Dropout does not perturb the Brownian stream.

C++ is compiled with g++ -O3 -std=c++17 -fPIC -shared -fopenmp. Source and artifact SHA-256 values are saved. Four OpenMP threads parallelize independent networks; trajectory outputs are invariant to scheduling because each network owns its state and RNG. No fast-math option is used.

## Metrics and uncertainty

The Kuramoto order parameter is R(t)=|N^-1 sum_j exp(i theta_j(t))|. R=1 means identical circular phases, and small R means weak first-harmonic phase coherence. R=0 does not identify a unique spatial pattern or prove statistical independence.

Tail metrics use all numerical states with t>=40, including both endpoints:

- tail_R_mean: time mean of R.
- tail_R_time_variance: population time variance of R inside a network; this is descriptive time variation, not an independent-sample variance estimate.
- tail_circular_variance: time mean of 1-R.
- tail_pairwise_disagreement: time mean over unordered pairs of 1-cos(theta_j-theta_i), equal to N/(N-1)(1-R^2). Values are dimensionless. This is phase disagreement, not semantic disagreement.
- tail_frequency_dispersion: population SD across agents of each agent's net phase displacement over t in [40,50] divided by 10. This is a finite-window average-frequency measure, not the nonexistent instantaneous derivative of Brownian phase.
- terminal_R: R(T).

Each cell saves all 500 independent network rows, including replicate id and all seed identities. Cell summaries report sample mean, median, between-network sample SD, and Student-t 95% confidence intervals over these independent network aggregates. Timesteps and individual agents are not treated as independent replications. Pointwise mean confidence bands on R(t) are not trajectory containment bands or simultaneous bands.

Paired communication contrasts subtract clean communication from each impaired condition using the matched network inputs and Brownian streams. They use independent network differences as the uncertainty unit. These intervals are descriptive, and no multiplicity-unadjusted significance claims are made. rho changes also reuse the same underlying independent/common streams, but correlation is a changed model condition, not a treatment on an independently sampled cohort.

## Output and resource plan

Each cell saves trajectory_rows.npy/csv, configuration.json (including adjacency and omega), checkpoint.json with configuration/source/artifact hashes, and sampled_paths.npz. The latter preserves R(t) for all networks every .1 time unit, complete initial phases, and absolute phase paths for the first five networks. Every integration step is used for tail metrics; every step is not stored. Disk estimate: approximately 350 MB before compression and metadata, under 1 GB including CSVs and figures. Per active network, the delay buffer stores at most 21*N doubles; the dominant cell output R array is 500*501 doubles (~2 MB). Compiled OpenMP avoids Python's per-edge overhead. A bounded throughput benchmark validates the implementation and estimates runtime; it is not substituted for the full factorial study.

Ten preflight tests cover topology/connectivity, covariance, degree normalization under dropout, delay history indexing, deterministic first-order error against solve_ivp, exact relative common-noise cancellation under delay/dropout, finite/reproducible outputs, independent-network confidence intervals and invalid inputs. A first native-binding attempt exposed a mismatched ctypes integer count and crashed before any full experiment; the ABI was corrected, then all initial eight tests passed; the two additional native-noise/delay checks were also run before full execution. No failed outputs enter the full study.

Execution commands:

    python -m pytest tests/test_synthetic.py -q
    python -m src.synthetic_agents --benchmark
    python -m src.synthetic_agents --full

Authoritative execution status is results/run_metadata/synthetic_run.json. A complete=true flag is set only after all 144 validated cell checkpoints, summary tables, paired contrasts and figures have been generated. Summary slices are not additional replications.

Measured warm throughput: 100 trajectories over T=5 took .0184 s for N3 complete, .0317 s for N8 ring, .1764 s for N20 complete and .1430 s for N20 sparse, all with rho=.5, dropout=.3 and delay=.2 on four threads. Projected full execution is approximately 6-9 minutes including storage, with a conservative dense-N20-for-every-cell bound of about 21 minutes. The matrix remains unchanged.

## Additional reproducible figure and diagnostic commands

    python experiments/synthetic_agents/make_factorial_figure.py
    python experiments/synthetic_agents/make_multistability_figure.py

The second command selects the first high-coherence and low-coherence rows of the executed N8 clean rho=1 ring cell and replays their deterministic relative states with sigma=0. This is valid because common additive noise cancels exactly in this model. These are diagnostic replays of existing network inputs, not new independent replications. It measures the signed ring winding as the sum of principal wrapped neighbor gaps divided by2pi and writes ring_state_diagnostic.json.

## Executed results and validation

All 144 cells and 72,000 condition-level network trajectories finished in 270.45 seconds including the primary tables and figures. Every cell retained 500 rows, T=50 and dt=.01. All 576 cell artifacts were hash-verified. Every reported mean, standard deviation and confidence interval was recomputed exactly from its network rows, and the N=3 complete/ring outputs are bitwise identical. The pairwise disagreement identity has maximum absolute error 7.88e-14. Five generated figures were rendered as PNG previews and visually inspected for titles, labels, legends and clipping. Validation is saved in experiments/synthetic_agents/validation.json.

A nonmonotone outcome occurred in the N=8 clean ring: mean tail R is .94512 at rho=0, .89720 at rho=.5 and .78460 at rho=1. At rho=1, the mean tail frequency dispersion is 1.13e-7, so low global order does not imply lack of frequency locking. Selected low-R replicate 6 has R=.0792492 and winding -1; high-R replicate 0 has R=.989622 and winding 0. Deterministic relative-state replays reproduce both original tail R values exactly. This verifies a selected twisted configuration rather than classifying every trajectory.

Dropout is not uniformly harmful to global R. In the N=8, rho=1 ring, dropout .3 increases mean tail R by .19356 relative to clean communication (paired descriptive 95% CI [.16075, .22637]). The model changes both mean coupling and random interruptions, so this comparison does not isolate a causal mechanism. Changes in basin selection are a plausible interpretation requiring further controlled experiments.
