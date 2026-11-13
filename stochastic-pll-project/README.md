# Stochastic phase-locked loop: complete local research project

Open `report/stochastic_pll_complete_report.html` directly in a browser. It embeds mathematics, charts, source-page images, code, tables and JavaScript; no server or network is needed. `report/main.tex`, `report/references.bib` and the PDF figures are the course-report sources. `report/main.pdf` is the compiled report when the complete workflow has finished.

## Results and scope

The original textbook equation (10.4), supplied p.274, is preserved. White noise is explicitly normalized by zeta(t)dt=dW(t); the textbook PLL paragraph does not independently calibrate this normalization to hardware. Code uses model time units, radians and radians per model time.

The full requested72-condition noise/detuning grid uses39,009 trajectories/control paths; nine cells are deterministic, stochastic cells use500 or1000 paths. Separate acquisition1000, long-horizon6000, a/b/noise9000 and coupled convergence1500 give56,509 independent PLL paths/control paths. Five coarse levels and the fine convergence reference share the same Brownian forcing; they are not extra independent paths. E1/E2/E3 are reusable slices of E4 where settings coincide, and are not pooled as independent replications.

The separate Kuramoto extension executes144 cells with500 networks each, N=3,8,20. Its72,000 network condition-runs share inputs across conditions at a fixed N, giving1500 independent matched input blocks. Degree normalization, discrete dropout and stale-relative-error delay are specified in `notes/synthetic_protocol.md`. Synthetic phase synchronization is not evidence of better LLM reasoning.

## Reproduce

Python3.14.8, NumPy/SciPy and other versions are saved in `requirements.txt` and `results/run_metadata/environment.json`. A C++17 compiler (`g++`) and OpenMP are required; no Numba or GPU is required. Tectonic0.17.0 is used to compile the formal PDF; it can fetch missing public LaTeX packages on first use. MathJax3.2.2 is vendored with its license. Exact RNG/math-library replay is platform/version dependent.

From this directory:

```bash
python -m pytest -q
MPLCONFIGDIR=/tmp/stochastic-pll-mpl python -m src.reproduce
```

The first command runs tests; the second reruns/resumes the entire full matrix and generates both reports. Completed valid checkpoints are reused; all original seeds and configurations are preserved. PLL checkpoint fingerprints cover numerical/analysis source and condition settings; synthetic checkpoints cover their source, native library and configuration. `src.experiment_runner.design` is the executable experiment design, also recorded in `configs/full_experiments.yaml` and the realized JSON. Changes to YAML alone do not silently change an executed design: update/validate the design function as well and keep the protocol consistent.

To rebuild reports from finished results:

```bash
MPLCONFIGDIR=/tmp/stochastic-pll-mpl python -m src.reproduce --reports-only
```

To regenerate individual outputs:

```bash
python -m src.analysis_supplement
MPLCONFIGDIR=/tmp/stochastic-pll-mpl python -m src.visualization
python -m src.synthetic_agents --full
python -m src.latex_report
python -m src.report_generator
python -m src.verify_outputs
```

## Evidence locations

- `notes/`: source audit, full mathematical derivation, preregistered experiment design, numerical validation, verified literature, synthetic protocol, supplied source images and original user specification.
- `backups/`: byte-identical copies of original Python/README/figures; originals remain in the parent directory.
- `results/raw_data/condition_*/`: per-trajectory rows with seeds, every primary committed event, sampled ensemble statistics, representative paths, terminal states, source/configuration fingerprints and wall times.
- `experiments/convergence/`: paired endpoint/path-error arrays, seeds and mesh sizes.
- `experiments/synthetic_agents/`: per-cell rows, seeds, topology/frequency/initial-state data, sampled R, selected phase paths, native source/library provenance and execution logs.
- `results/tables/`: condition summaries, raw-data-derived means/medians/SDs/CIs, descriptive effect sizes, network contrasts and interactive baseline data.
- `results/figures/`: reproducible SVGs with scientific captions; `report/figures/`: matching PDF figures for LaTeX.
- `results/run_metadata/`: complete actual counts, test logs, environment, source/original/raw hashes, integrity audit and browser verification.

The raw output-quality audit recomputes counts/means/variance from independent rows, reconciles event records, checks all9 deterministic controls and63 stochastic cells, verifies unique PLL seed identities, convergence finiteness and original preservation. Browser verification separately tests the actual offline document; static HTML checks do not imply rendering proof.

## LLM extension

No paid API calls were made. The official GSM8K JSONL files were downloaded and counted (7473 training,1319 test); hashes are saved in `experiments/llm_extension/dataset_manifest.json`. The conditional full-split design and token-budget formula are in `experiments/llm_extension/assessment.md`. A language-agent phase representation has not passed the scientific validation gate. Azure deployment/pricing/rate limits and explicit cost approval are required before any paid inference; no credentials are stored in this project.

## Limits

Finite-time operational capture is not global pull-in or asymptotic stochastic lock. The core-and-residence slip detector is compared with two alternatives, not asserted to be an exact separatrix detector. No observed events does not prove a zero rate. Convergence relative to a finite numerical reference at T10 does not establish long-horizon rare-event count accuracy. The parameter and topology designs have stated normalization, graph-realization and operating-regime limits. No hardware data, benchmark accuracy or novelty is fabricated.
