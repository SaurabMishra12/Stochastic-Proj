# Full stochastic PLL implementation plan

Goal: complete the user specification with actual local computations and standalone teaching report.
Architecture: preserve the original; Python model/analysis orchestration and a compiled C++ trajectory kernel using NumPy-generated Brownian increments. Results are checkpointed by condition. Synthetic oscillators form a separate experiment family. No paid API calls are authorized.
Spec: notes/user_specification.txt and supplied textbook images.

- [ ] Audit and hash originals; derive the model and record the noise convention.
- [ ] Define complete matrix before execution: 72 grid conditions (9 deterministic); 500 stochastic paths minimum, 1000 baseline and selected comparisons. T=50, dt=.001. Reuse grid slices for E1/E2/E3 without claiming them as independent experiments.
- [ ] Write numerical tests, observe initial failure, implement and pass preflight before matrix.
- [ ] Execute checkpointed full grid, selected six T=200 ensembles, coupled 1000-path convergence study, and 18 a/b/noise sensitivity ensembles.
- [ ] Analyze independent-trajectory uncertainty, event-rate bootstrap intervals, Wilson intervals, right-censored first-passage outcomes and detector sensitivity.
- [ ] Execute synthetic oscillator topology/noise/dropout/delay matrix after core validation.
- [ ] Verify primary literature and assess LLM extension without paid inference or arbitrary embedding phases.
- [ ] Generate embedded HTML, concise seven-section LaTeX, charts and reproducibility manifest.
- [ ] Browser-test offline interactions and layout; compile LaTeX; verify source preservation and all outputs.

Resource design: grid about 2 billion path-steps; six 1000-path extensions add 1.2 billion; 18*500 sensitivity adds .45 billion. Streaming one path of 50k/200k Brownian increments uses .4/1.6 MB; sampled ensembles at 1001 points use ~16 MB per 1000-path condition. Two concurrent condition workers bound memory. Exact timing is measured from mandatory validation runs, never inferred as completed science.
Review focus: threshold |delta|=1; negative detuning symmetry; initial capture versus global lock; completed transitions versus boundary recrossings; right censoring; deterministic ensembles have no Monte Carlo CI.
