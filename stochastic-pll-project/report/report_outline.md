# Standalone report outline

The report is a complete teaching document generated only after executed output exists. The mathematical source is the supplied Kloeden and Platen Chapter 7 pages 273–275, especially equation (10.4), p.274. All figures are embedded; MathJax 3.2.2 SVG rendering is embedded without CDN requests.

A. **Project overview**: purpose, research questions, complete execution scope, key findings computed from real rows, contributions and limits.
B. **Mathematical foundations**: source provenance, physical architecture, second-order to Itô derivation, every symbol, Brownian scaling, drift/diffusion, stable and unstable equilibria, Jacobian/eigenvalues, tilted potential, stochastic escape, lock/capture/rotation distinctions.
C. **Visual mathematical explanations**: thirteen numbered explanatory SVG diagrams, each with an explanatory caption and scope statement. These are labeled conceptual and never presented as numerical evidence.
D. **Complete code walkthrough**: every Python module and its role/input/output/relationships; every defined function including arguments, returns, docstrings, AST-derived execution steps, pertinent equations, implementation choices and common errors. Full syntax-highlighted sources in collapsibles; detailed actual numerical-loop analysis.
E. **Experiment methodology**: preregistered E1–E7 matrix, independent trajectory counts, controlled variables, coupled numerical comparisons, detector sensitivity, uncertainty methods, synthetic extension and approval-gated LLM proposal.
F. **Actual experimental results**: baseline controls and exact original replay, noise/detuning grid, long-horizon slips, parameter sensitivity, coupled convergence, synthetic topologies. Data-backed summary tables with uncertainty. Every saved plot receives a specific explanation, interpretation and limitation; raw tables remain available inline and downloadable. Two offline interactive figures operate on executed series and grid rows.
G. **Literature review**: verified primary sources, textbook source pages, source-to-claim scope, distinction between established results and proposed extensions.
H. **Discussion and limitations**: numerical/reference error, finite-horizon operational capture, phase-space separatrix caveat, censoring, zero-event uncertainty, multiplicity, synthetic scope, lack of LLM evidence.
I. **Conclusion and future work**: demonstrated outcomes, uncertainty, discriminating follow-up experiments and extension gates.
J. **Reproducibility**: actual environment/manifest, run configuration, command sequence, seeds/stream semantics, all figure-generation paths and hashes, embedded notes and raw-source inspection.

Navigation: sidebar A–J anchors, whole-section search with count and no-results/reset state, theme toggle, mobile contents toggle, expand/collapse explanations, figure controls with labels and an accessible readout. All numerical result claims are derived from loaded CSV/JSON and completed-run metadata.

Generation gate: missing required outputs, incomplete run metadata, absent chart captions or missing vendor runtime cause a clear error. No placeholder datasets, externally fetched runtime files or fabricated tables are allowed.
