# Azure GPT-5 mini pilot pricing evidence

Verified on **2026-11-13** using official Microsoft sources. This research made public pricing GET requests only. The parent agent separately reported an authenticated metadata GET with HTTP 200 and zero inference requests; see [provider metadata](pilot/provider_metadata.json). The deployment reports `gpt-5-mini`, version `2025-08-07`, `GlobalStandard`, and `KoreaCentral`. Its matching catalog region is `koreacentral`.

## Matched pay-as-you-go rates

| Token class | USD per 1,000,000 tokens | Korea Central meter ID |
| --- | ---: | --- |
| Uncached input | $0.25 | `fabef4c7-194c-52d7-914f-0b3998b002ac` |
| Cached input | $0.025 | `4c5ee70b-19c5-5041-bace-ce319aa143c1` |
| Output, including reasoning | $2.00 | `f32d456a-4151-5a49-96af-503e93707054` |

The exact meter names are `GPT 5 Mini Inpt Glbl 1M Tokens`, `GPT 5 Mini cchd Inpt Glbl 1M Tokens`, and `GPT 5 Mini outpt Glbl 1M Tokens`. Each has `unitOfMeasure: "1M"`, `currencyCode: "USD"`, `type: "Consumption"`, `isPrimaryMeterRegion: true`, and `effectiveStartDate: "2025-08-01T00:00:00Z"`. These are prices per **one million**, not one thousand tokens. The product is `Azure OpenAI GPT5`; its current retail service name is `Foundry Models`.

The [raw public catalog response](pricing/azure_gpt5_mini_usd_page_01.json) contains 234 exact GPT 5 Mini standard/batch records across 28 regions. `NextPageLink` is null. The [fetch manifest](pricing/azure_gpt5_mini_usd_fetch_manifest.json) records the exact URL, retrieval time, HTTP status and SHA-256. [Azure's API documentation](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices) describes this unauthenticated catalog as undiscounted retail pricing and identifies USD as Microsoft's pricing currency. Account discounts, credits, currency conversion, taxes and contract terms remain unverified. The dynamic [Azure pricing page](https://azure.microsoft.com/en-us/pricing/details/azure-openai/) timed out in the web reader; the verified rates above come from the official retail API.

## Deployment types

For the returned GPT 5 Mini meters, Global Standard is $0.25/$0.025/$2.00 for input/cached input/output; Data Zone Standard is $0.275/$0.0275/$2.20 per million. Prices are consistent across each meter's returned regions. Data Zone prices must not be substituted into this Global Standard deployment's estimate.

[Microsoft's deployment documentation](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/deployment-types) distinguishes `GlobalStandard` (global processing), `DataZoneStandard` (processing within a Microsoft data zone), and `Standard` (processing within the selected Azure geography). All are separate pay-per-token deployment SKUs. A resource region alone does not imply geography-based Standard processing. No regional Standard GPT-5 mini price was verified in this bounded catalog query; no availability conclusion is inferred from that absence. Batch, priority and provisioned processing are separate offerings and are excluded from the pilot estimate.

## Confirmed pilot and proposed cap

The scope is **50 samples each from GSM8K, HotpotQA and BBH reasoning: 150 total**, with five methods and seven logical calls per sample-method pair. That is **5,250 logical calls**. No extra development cohort or preliminary inference subset is included.

| Scenario | Input tokens | Billed output tokens | Global Standard cost |
| --- | ---: | ---: | ---: |
| Expected, 2,000 input + 800 output per logical call, no retry | 10,500,000 | 4,200,000 | $11.025 |
| One attempt per logical call at 12,288 input + 4,096 output ceiling | 64,512,000 | 21,504,000 | $59.136 |
| Every logical call uses three full-ceiling HTTP attempts | 193,536,000 | 64,512,000 | $177.408 |

Expected cost per dataset is $3.675. These are arithmetic scenarios, not measured usage or guaranteed total charges. The corresponding expected Data Zone scenario is $12.1275; it is included only to document the SKU distinction.

The proposed **$25 USD execution cap requires explicit user approval before billable calls**. It covers the expected scenario with room for some variation, but cannot guarantee all 5,250 calls finish. Reserve each HTTP attempt's worst-case cost before sending; account conservatively for retries and ambiguous failures, and stop before the cap. HotpotQA context, agent-message history and hidden reasoning can increase usage. No cached-token savings are assumed.

[Azure's reasoning documentation](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning#control-costs) states that reasoning tokens are billed at the output rate. `max_completion_tokens` covers reasoning, visible output and formatting tokens together. Use provider-reported total completion tokens once, rather than adding reasoning tokens to that total again.

## Throughput uncertainty

The metadata capacity value is `250`; this research does not convert it into a verified runtime quota. [Current quota documentation](https://learn.microsoft.com/en-us/azure/foundry/openai/quotas-limits) lists tier-dependent values. For example, Tier 1 GPT-5 mini Global Standard is 1,000 RPM / 1,000,000 TPM and Data Zone Standard is 300 RPM / 300,000 TPM. These are published tier examples, not this account's verified allocation. The proposed runner throttle of 200 RPM / 200,000 TPM is a planning setting. Actual account tier, allocation, runtime rate-limit headers, latency and sustained throughput remain unverified; handle 429 responses and reduce concurrency after the approved run begins.
