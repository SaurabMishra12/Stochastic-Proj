# Three-dataset MAS pilot

Current stage: RUNNING after the user's explicit approval of a US$25 hard cap. See `run_status.json` and `approval_record.json` for current execution and authorization evidence. The first seven-call condition used one of the selected150study examples; it was not an additional paid smoke-test question. User scope supersedes the earlier full-scale proposal: exactly50examples each from3real datasets,150total, with fixed cohort/order/controller seeds. No additional development cohort, larger test split or robustness sweep is authorized by this pilot plan.

## Frozen experiment

- GSM8K:50uniformly sampled without replacement from1319official test rows. Seed20261113.
- HotpotQA:50from7405distractor validation rows, including all supplied10context paragraphs and excluding reference-answer/supporting-fact metadata. Seed20261114. OriginalCMUhost no longer resolved; the maintained HuggingFace distribution is preserved with a checksum.
- BIG-BenchHard:50from250`logical_deduction_three_objects` rows. Seed20261115. This is one identified BBH task, not an estimate of the full BBH benchmark.
- Five conditions: independent, full communication, fixed directed ring, answer-disagreement-triggered communication, stochastic disagreement controller.
- Every condition uses3independent drafts,3refinements from the same frozen initial round, and1common-format adjudication.7logical calls per item/method,5250planned calls. Initial calls are separately generated within every method; no cross-method response cache is used.
- Fixed `gpt-5-mini`, low reasoning effort,4096total output tokens including hidden reasoning, store=false, no tools, no temperature/top_p or unsupported provider seed. Brief external summaries are capped at600characters for communication. Prompt/config/code/cohort hashes are preserved.
- Sampling and controller streams are reproducible. Exact model outputs cannot be promised reproducible: this API has no verified supported deterministic seed parameter.
- The stochastic controller is a disclosed OU-style message gate driven by answer disagreement. It is not a derived physical PLL and has no validated embedding-to-phase interpretation. This pilot is exploratory; its50-question per-dataset confidence intervals may be wide.

## Provenance and files

`cohort.jsonl`: all150fixed selected examples with reference labels for local scoring only. `model_view()` is the explicit label-free boundary into payload construction. QA passages naturally contain evidence about the answer; reference answer fields and annotated supporting facts are never injected.

`cohort_manifest.json`: source checksums, all selected indices, task/split, seeds and counts. Rawsource data remain in `source_data/`; GSM8Koriginal files remain one directory above.

`provider_metadata.json`: read-only authenticated GET results (200), OpenAIpublisher, gpt-5-mini version2025-08-07, GlobalStandard capacity250 and KoreaCentral headers. This is model/deployment metadata, not generated-answer validation. No runtime quota is inferred from capacity alone.

`run_manifest.json`: exact frozen inference design and hash. `offline_validation.json`: validation only, not empirical LLM results. `logs/`: preflight logs. No fake answers are placed in research-results directories.

## Spending and approval

Verified public retail rates for the matching KoreaCentralGlobalStandard meters are $.25uncached input and $2output per million tokens. Cached discounts are ignored for a conservative estimate. Account contract terms, tax and final invoice may differ.

Planning assumption:2000input and800billed completion tokens per call,5250calls ->10.5Minput +4.2Moutput ->$11.025. These are assumptions, not measured usage. The full output limit may produce more than800tokens; hidden reasoning is billed output. The global32768input-bound/4096output-bound ceiling would be$86.016if every call used both bounds once; up to3attempts per slot is bounded separately. Actual prompts are checked before dispatch, with serializedUTF8bytes+512 as a conservative input-token upper bound. Offline long-message scenario min/max bounds are1159/13233; responses can change subsequent actual bounds.

Approved hard list-price spending cap:$25. It can stop the matrix before completion if usage/retries are higher. SQLite reserves each attempt's upper-bound cost before sending. Unknown/missing-usage outcomes retain the full reservation across crashes and resume. Only429/500/502/503/504 may retry, at most3HTTPattempts; content filters, incomplete outputs, parsing errors and ambiguous timeouts are not automatically repaired or resent. HTTP attempts are counted separately from logical calls. Existing logical IDs cannot be silently redispatched. No paid judges or hidden retries.

Your original specification requires a cost estimate and approval before billable calls. The runner refuses to start without an explicit user-approval JSON matching the frozen manifest and a stated maximum USD amount. The key is held in a mode0600external temporary file or an environment variable; it is not saved in this project, prompts or request logs.

## Commands

From the project directory:

```bash
python -m pytest -q
python -m src.llm_mas.pilot prepare
python -m src.llm_mas.pilot run --approval /tmp/stochastic-pll-pilot-approval.json --workers 12
```

The third command is gated; preparing the runner does not authorize spending. Rate settings200RPM/200000TPM are conservative planning settings, not verified quotas. Rate retries/backoff and real usage are recorded. Runs may take substantial time because750method/item cells require5250generations.

## Planned analysis

Preserve all750method/item cells, all5250logical slots, rawresponses, attempt/request logs, token usage, costs, errors, stage answers, sent messages and controller draws. Compute numeric-exact GSM8K matching, conventional normalized HotpotQA EM/tokenF1, and BBH option accuracy. Treat the question as the uncertainty unit, preserve all50questions per dataset even when responses fail, and use paired question bootstrap comparisons/Wilson intervals with exploratory multiple-comparison controls. Do not pool the three tasks into a broad benchmark claim. No inference accuracy is reported until actual calls complete.

Read-only analysis/report commands (no model calls):

```bash
python -m src.llm_mas.analysis
MPLCONFIGDIR=/tmp/stochastic-mpl-cache python -m src.llm_mas.report
```

These can produce explicitly marked partial snapshots while inference runs. The `--integrate` report option requires a complete, audited matrix before appending the pilot to the original complete research HTML; it preserves a byte-identical backup of the pre-pilot report. Analytical modules are excluded from the frozen inference-code hash and cannot alter model payloads or the budget policy. Token counting uses the pinned project-local `.analysis_deps/tiktoken`0.12.0 with the cached `o200k_base` vocabulary. Physical request payloads and final scoring are reconstructed and compared to raw evidence in the final audit.
