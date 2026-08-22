# Show HN draft (not published — post this yourself)

**Timing per the launch research:** Tuesday–Thursday, 8–9am ET. Link the
GitHub REPO, not the docs site. Stay in the comments all day. Publish a
"how I built it" technical post 1–2 days earlier and link it from a
comment, not the submission.

---

**Title:**

Show HN: OpenBio-Intel – open-source clinical trial market intelligence (LangGraph, Qdrant, Neo4j)

**URL:** https://github.com/JAGAN666/OpenBio-Intel

**First comment (post immediately after submitting):**

Hi HN — I built an open-source alternative to biopharma intelligence platforms like Cortellis and AlphaSense, for the slice of their value that's actually built on public data: clinical trials, FDA approvals *and rejection letters*, PubMed, SEC filings, adverse-event signals, and patent/exclusivity data.

You ask an analyst question ("which trials combine pembrolizumab with what?") and a LangGraph agent federates across ten tools, fans out ~100 parallel extraction workers over the retrieved trials, and returns a comparison table where every cell links to the primary record. Nothing is answered from model memory — the schema forbids it, and per-corpus grounding gates reject answers whose evidence doesn't share vocabulary with the question.

Things I learned building it that might interest this crowd:

- Dense embeddings are genuinely bad at biopharma queries: on our committed 50-query golden set, dense-only retrieval returned *zero* relevant results for 26% of queries — anything naming a development code like "BBO-10203". Qdrant's server-side BM25+dense RRF fusion took mean recall@20 from 0.44 to 0.59; a cross-encoder rerank got 0.61. Harness and baselines are in the repo (eval/).
- FDA started publishing full-text rejection letters (Complete Response Letters) in mid-2025 and almost nobody has indexed them. The officially-announced API path 404s; the real endpoint is transparency/crl. 459 letters, searchable.
- FDA never publishes forward PDUFA dates — every commercial "FDA calendar" mines them from press releases and 8-Ks. So does mine, with citations.
- Long agent runs (2–10 min) inside HTTP requests are a reliability disaster. Everything now runs as jobs on a Postgres SKIP LOCKED queue with LangGraph checkpointing — I SIGKILLed a worker mid-run and the job resumed from its checkpoint and finished in 33s instead of re-spending 150s of LLM calls.
- There's an MCP server, so all ten tools work inside Claude directly.

Honest limitations: no auth in the reference deploy (put your own in front), Purple Book biologics coverage is partial because FDA publishes no full extract, and the things commercial vendors are genuinely better at (consensus forecasts, curated deal terms, KOL databases) are documented as out of scope rather than faked.

Stack: FastAPI + LangGraph + Qdrant + Neo4j + Postgres + Next.js, deployed on ECS via Terraform with OIDC CI/CD. MIT licensed. Docs: https://jagan666.github.io/OpenBio-Intel/

Would love feedback — especially from anyone who's worked with the commercial platforms and knows what analysts actually need.

---

**Launch-week cadence (from the OpenBB playbook):**

1. 1–2 days before HN: post the technical deep-dive (benchmarks page content works) to dev.to; soft-share in r/biotech, r/SecurityAnalysis if rules allow.
2. Launch day: Show HN + first comment above; answer everything.
3. Week after: one feature demo post (catalyst tracker or CRL search) per week on X/LinkedIn; monthly "update" Show HN only when there's something real.
