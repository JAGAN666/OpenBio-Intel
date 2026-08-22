# Data sources

Everything OpenBio-Intel serves comes from public primary sources. That's the point: commercial intelligence vendors build mostly on the same substrate — the moat is extraction and monitoring, not access.

| Source | What it provides | How it's ingested |
|---|---|---|
| **ClinicalTrials.gov** (v2 API) | ~600K trial records: design, phase, sponsors, interventions, conditions, dates | Bulk seed + daily delta; hybrid-indexed in Qdrant; also queried LIVE for catalyst date enrichment and watchlist diffs |
| **openFDA drugsfda** | Drug approvals, application numbers, products, ingredients | Bulk JSON stream (`ijson`), Qdrant collection |
| **openFDA transparency/crl** | FDA **Complete Response Letters** — full-text rejection letters, published since mid-2025 | `fetch_fda_crls.py`; 459 letters → 3,410 hybrid chunks. Note: the officially-announced `other/approved_CRLs` path 404s; `transparency/crl` is the real endpoint |
| **PubMed Central OA** | Peer-reviewed full text | NCBI Entrez → JATS parse → chunk → embed |
| **SEC EDGAR** | 10-K/8-K pipeline/R&D sections | `sec-edgar-downloader` → section isolation → chunk |
| **Corporate news** | Press releases (FDA/J&J/AbbVie RSS) + earnings-call transcripts | RSS + transcript fetch; where PDUFA dates surface first |
| **FAERS** (openFDA drug/event) | Post-market adverse events | Queried **live** at answer time; reporting odds ratios computed with 95% CI |
| **Orange Book** | Small-molecule patents + exclusivity | Monthly FDA zip → 10,869 brand products → Neo4j `RegulatoryProduct` nodes |
| **Purple Book** | Biologic exclusivity | Monthly change CSVs stacked (FDA publishes **no full extract** — verified) → 938 BLAs reconstructed |
| **Conference-poster PDFs** | Efficacy/safety tables not in registries | LlamaParse vision-mode → Markdown chunks |
| **AACT** (optional) | Exact SQL over all of ClinicalTrials.gov | Text-to-SQL tool, enabled when you register (free) and set `AACT_DB_URL` |

## Honest limitations

- **Purple Book coverage is partial** — reconstructed from monthly change files because FDA publishes no complete extract; biologics that haven't changed recently may be missing. Labeled as such in results.
- **Conference abstracts (ASCO/ESMO/AACR) are not ingested** — their text-and-data-mining licenses forbid redistribution. The catalyst pipeline covers conference readouts indirectly via the press releases and 8-Ks companies publish alongside.
- **PDUFA dates are extracted, not official** — FDA never publishes forward action dates (21 CFR 314.430); like every commercial calendar, ours are mined from company disclosures, and each event carries its source citation.
- **FAERS shows association, not causation** — the tool says so in every payload, and so should you.
