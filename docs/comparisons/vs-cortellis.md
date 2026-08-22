# OpenBio-Intel vs. Cortellis / Citeline

An honest comparison. Clarivate Cortellis and Citeline (Pharmaprojects, Trialtrove, Sitetrove) are mature enterprise platforms with decades of curated data and analyst teams behind them. OpenBio-Intel is an open-source project you can read and self-host. These are different things — here is where each is genuinely stronger.

## Where the commercial platforms win

- **Curated depth and history** — decades of hand-curated pipeline records, deal terms, and epidemiology that no automated pipeline reconstructs.
- **Forecasts and success-rate models** — consensus revenue forecasts and probability-of-approval scores built on licensed sell-side data and analyst labor. OpenBio-Intel deliberately does not fake these.
- **KOL/site intelligence at scale** — investigator databases with disambiguated identities.
- **Support, SLAs, compliance reviews** — you're buying an organization, not just software.

## Where OpenBio-Intel wins

- **You can read the methodology.** Every retrieval decision, grounding gate, and extraction prompt is in the repo. When a number looks wrong you can trace exactly how it was produced — and every table row carries clickable citations to primary records.
- **Self-hosted: your queries never leave your infrastructure.** For competitive-intelligence work, what you *ask* is itself sensitive. SaaS platforms see every query your team runs.
- **Sources the platforms are slow on** — FDA Complete Response Letters (published only since mid-2025) are fully indexed and searchable here; live FAERS disproportionality screening is a tool call away.
- **Extensible in an afternoon** — new data source, different LLM provider, custom export: fork and change it.
- **Cost** — your infrastructure (a few small cloud instances) plus LLM API usage, versus five-figure-per-seat subscriptions.
- **Agent-native** — an MCP server exposes everything to Claude and other agents; the platforms are human-UI-first.

## The substrate is the same

Trial registries, FDA databases, SEC filings, press releases — most of what both serve comes from public data. The commercial moat is extraction, entity resolution, and monitoring. OpenBio-Intel's position is that this layer is better built in the open, where its accuracy can be audited ([benchmarks](../benchmarks.md)) rather than taken on faith.

**Honest bottom line:** if you need consensus forecasts, curated deal economics, or enterprise support, buy the commercial product. If you need auditable, self-hosted, extensible clinical trial + regulatory intelligence — or a foundation to build your own internal platform on — that's what this is.
