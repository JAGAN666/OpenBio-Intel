# medical-rag — enterprise life sciences RAG platform

Unified **Land-Then-Archive** ingestion pipeline. ClinicalTrials.gov v2 → S3 raw
data lake → FastEmbed (local embeddings) → Qdrant, with high-selectivity payload
filtering on trial metadata.

## Architecture

```
ClinicalTrials.gov API v2
        │  Phase 2/3, cond=Oncology, sort=LastUpdatePostDate:desc, 50 studies
        ▼
   1. LAND      data/raw/payload_<ts>.json           untouched payload
        │
        ▼
   2. ARCHIVE   s3://medical-rag-raw-data-lake-jn-9043/
        │         raw/clinicaltrials_gov/v2/ingest_date=YYYY-MM-DD/payload_<ts>.json
        │       ── size + ETag VERIFIED before anything is parsed ──
        │
        ▼  (gate: parsing never begins unless the archive is durable)
   3. INDEX     FastEmbed nomic-embed-text-v1.5 (768-dim, 8192-token, local ONNX)
        │       vectorizes BriefSummary
        ▼
   Qdrant :6333   collection `clinical_trials`
                  payload indexes: Phase, OverallStatus, LeadSponsorName, NCTId
                  every point carries SourceS3Key lineage
```

The raw payload is archived **before** parsing, so the entire transform is
replayable from S3 — you can re-embed with a better model later without
re-hitting the upstream API. The Hive-style `ingest_date=` partition means
Athena or Glue can register this prefix as a partitioned external table with no
reorganization.

## Files

| File | Purpose |
|---|---|
| `fetch_and_embed_trials.py` | The pipeline: fetch → land → archive → embed → index |
| `verify_qdrant.py` | Hybrid query: semantic + strict `Phase` filter |
| `research_agent.py` | LangGraph ReACT agent: reason → Qdrant search → cited answer |
| `api.py` | FastAPI wrapper — `POST /api/research` for the frontend |
| `evaluate_agent.py` | Ragas evaluation: faithfulness + answer_relevancy vs. an independent judge |
| `provision_aws.sh` | Creates + hardens the S3 bucket via AWS CLI (idempotent) |

## Intelligence layer — `research_agent.py`

**Deterministic Orchestrator** architecture — three guardrails wrap the
ReACT loop so the graph fails closed instead of hallucinating:

```
START ─▶ [IntentClassifier] ──in domain?──▶ [OutOfDomain] ─▶ END
                │ yes                        deterministic refusal,
                ▼                             no LLM call, no Qdrant hit
           [Agent] ──tool_calls?──▶ [Tool Node]
              │  no                      │
              ▼                          ├──has_results?──▶ [NoResultsFallback] ─▶ END
         [Synthesis] ◀───────────────────┘  no                deterministic, no LLM call
          │      ▲  loop back (ReACT)   yes
   valid  │      │                       │
  output  │      └── retry (max 2) ──────┘
          ▼           on Pydantic ValidationError
         END
```

```bash
.venv/bin/python research_agent.py                       # the AC-4 question
.venv/bin/python research_agent.py --question "..."      # your own
.venv/bin/python research_agent.py --quiet               # answer only, no trace
```

Needs `ANTHROPIC_API_KEY` in `.env`. Generator defaults to `claude-opus-5`;
the `IntentClassifier` gate uses `claude-haiku-4-5` (cheap/fast, same
provider — no second API key to manage).

Design notes:

- **`Tool Node` loops back to `Agent`, not straight to `Synthesis`.** That is
  what makes this ReACT rather than a fixed chain — the agent observes results
  and issues refined follow-up searches. `MAX_TOOL_ROUNDS` caps the loop by
  unbinding the tools, forcing the agent to conclude.
- **`has_results` is a lexical-overlap check, NOT a cosine-similarity
  threshold.** Measured directly before building this: score magnitude in
  this embedding space tracks query *phrasing style*, not factual grounding —
  the fictional phrase `"trials of XYZ-Fake-Drug-999 in oncology"` scored
  **0.636**, higher than the real drug query `"brentuximab vedotin"` at
  **0.555** (brentuximab is genuinely in the corpus). Any threshold strict
  enough to reject the fictional query would also reject the real one — i.e.
  it would break the happy path to "fix" the edge case. Instead, `has_results`
  checks whether any retrieved trial's own curated fields (title, conditions,
  intervention names — exact strings, not a lossy vector) share distinctive
  vocabulary with the query. kNN search always returns the *k nearest*
  points, however irrelevant; lexical overlap against ground-truth fields is
  what actually signals "nothing here is relevant."
- **The stopword list is domain-specific and was expanded after a live
  failure.** `"XYZ-Fake-Drug-999 investigational compound efficacy and
  safety"` first slipped through as `has_results=True`, because a real
  trial's title happens to be *"A Study to Evaluate the Safety and
  Tolerability of..."* — generic ClinicalTrials.gov regulatory phrasing
  overlapping the query by accident. A corpus-wide document-frequency filter
  (exclude tokens appearing in ≥30% of trials) is layered on top as
  defense-in-depth, but at only 50 trials it has too little data to do the
  separating alone — "safety" (6%) and "efficacy" (4%) are currently *rarer*
  than the real drug "pembrolizumab" (16%). The static list does the real
  work today; the frequency filter is what takes over as the corpus scales
  into the thousands, where boilerplate title templates dominate their true
  frequency.
- **`has_results` is sticky-OR across ReACT rounds**, not re-evaluated fresh
  each round. Once any round finds grounded results, a later unlucky
  refinement search doesn't kill an already-successful earlier one. An
  ungrounded *first* round (nothing to inherit) still correctly triggers
  `NoResultsFallback` — see AC 3 above.
- **The Synthesis retry loop uses `include_raw=True`**, not a bare
  `try/except`. `with_structured_output(SmartTableResponse, include_raw=True)`
  returns `{"raw", "parsed", "parsing_error"}` instead of raising — the graph
  can inspect a failure without an exception unwinding the request. Verified
  in isolation with a fake LLM that fails on command (`0`, `1`,
  exactly-`MAX_SYNTHESIS_RETRIES`, and over-budget cases), since Anthropic
  enforces the tool schema server-side and a live `ValidationError` is
  unlikely to occur organically.
- **`Synthesis` is a separate LLM call** that sees only the retrieved records
  plus the question, under a strict citation contract — so citations are
  grounded in tool output rather than model memory.
- **Every exit path returns the same `SmartTableResponse` type** —
  `OutOfDomain` and `NoResultsFallback` are deterministic (no LLM call) but
  still populate the full contract (empty `table_data`, fixed message), so
  the FastAPI layer and frontend never special-case a guardrail outcome.
- **No `temperature` / `top_p`.** `claude-opus-5` rejects them with a 400;
  behaviour is steered by prompt. Thinking is on by default and `max_tokens`
  caps thinking + text together, hence the generous `max_tokens=8000`.
- **The tool uses `qdrant-client` directly**, not `langchain-qdrant`'s
  `QdrantVectorStore`. The collection stores metadata flat (`NCTId`, `Phase`, …)
  with text under `document` on a named vector `fast-all-minilm-l6-v2`, whereas
  the vector store expects `page_content`/`metadata`. Reusing the same
  `set_model()` path that indexed the data guarantees query and stored vectors
  are comparable.
- **Citations are verified, not trusted.** The run regexes every `NCT\d{8}` out
  of the answer and cross-checks it against the IDs actually returned by tool
  calls; a fabricated identifier exits non-zero.
| `iam-policy-medical-rag.json` | Least-privilege IAM policy for the bucket |
| `.env` | AWS region + bucket name (credentials come from `~/.aws`) |
| `requirements.txt` | Pinned deps — see the version warning below |

## Run

```bash
# Qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest

# Deps
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Pipeline  (first run downloads ~80MB of ONNX weights, then cached)
.venv/bin/python fetch_and_embed_trials.py --limit 50 --recreate

# Verify
.venv/bin/python verify_qdrant.py
```

Flags: `--limit N`, `--condition`, `--batch-size`, `--recreate`,
`--skip-s3` (local dev only — breaks replayability).

## AWS provisioning

```bash
./provision_aws.sh --dry-run
./provision_aws.sh
```

Creates the bucket, blocks all public access, enables versioning and SSE-S3,
applies lifecycle tiering (`STANDARD_IA` at 30d → `GLACIER_IR` at 90d,
noncurrent versions reaped at 180d), tags it, then verifies by round-tripping a
real object.

## ⚠️ Pinned client — do not bump blindly

`qdrant-client` is pinned to **1.18.0**. The FastEmbed convenience API this
pipeline uses — `client.set_model()`, `client.add()`, `client.query()` — was
**removed in 1.19.0**. 1.18.0 is the last release that ships it, and it already
emits a `DeprecationWarning`.

To move to 1.19+, port to the explicit API: build vectors with
`fastembed.TextEmbedding` (or wrap text in `models.Document`) and write via
`client.upsert(points=[PointStruct(...)])`.

## Smart Table output (structured)

`research_agent.py` returns a validated `SmartTableResponse` — `narrative_summary`
(prose, NCTId-cited) plus `table_data` (`list[TrialRow]`) for a frontend grid.
LangChain's `.with_structured_output()` binds the Pydantic model as a tool
schema, so a malformed row raises a `ValidationError` server-side instead of
rendering an empty cell.

**Strict corpus grounding.** The schema and synthesis prompt forbid supplying a
mechanism from the model's own pharmacological knowledge: if a record names an
agent without describing how it works, the row must say so rather than fill the
gap. This was added after a real defect — the model asserted rilvegostomig was a
"PD-1/TIGIT bispecific" and pumitamig a "PD-L1×VEGF-A bispecific" with bracketed
NCT citations, when `TIGIT` and `VEGF` appear in **zero** stored documents. Both
claims are true in the wider world; neither was supported by the cited record.
*A true fact with a false citation is still a citation defect.*

**Flags are decoupled from prose.** `TrialRow` carries both
`mechanism_described: bool` (the UI contract — branch on it) and
`mechanism_or_findings: str` (analyst context — render it). A record that names
an agent without explaining how it works yields `false` *plus* whatever the
record does give: population, biomarker, comparator, endpoint. The frontend gets
a reliable signal without the backend discarding clinical detail.

`mechanism_described` is declared **after** `mechanism_or_findings` on purpose —
Pydantic field order is JSON-schema property order, which is generation order in
the tool call, so the model writes what it found first and then flags what it
just wrote.

Frontend notes:

- `phase` is `str`, never an enum — trials legitimately return `"Phase 2/Phase 3"`.
- Branch on `mechanism_described`, never on the prose. Rows flagged `false`
  still carry useful text and must not be rendered as empty.
- The flag is a genuine judgment, not a keyword match: a trial whose record
  names `EGFR` only as a *patient-selection biomarker* is correctly flagged
  `false`, while one describing an agent as "an oral HER2-targeted treatment"
  is flagged `true`.

## Running the full stack

```bash
# 1. Qdrant
docker start $(docker ps -aq --filter "publish=6333") 2>/dev/null || \
  docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest

# 2. API  (http://127.0.0.1:8000 — /docs for OpenAPI, /api/health for readiness)
.venv/bin/python -m uvicorn api:app --reload --port 8000

# 3. Frontend  (http://localhost:3000)
cd frontend && npm run dev
```

> **Port 8000 note.** `localhost` resolves to IPv6 first on macOS. If something
> else holds the IPv6 wildcard on :8000 (e.g. a stray `python -m http.server
> 8000`), `localhost:8000` reaches *that* server while uvicorn sits on IPv4.
> The frontend therefore targets `127.0.0.1:8000` explicitly.

## Type sync — Pydantic is the source of truth

`frontend/types/trial.ts` is **generated, never hand-edited**. After changing
`SmartTableResponse` or `TrialRow`:

```bash
.venv/bin/python export_schema.py     # -> frontend/schema.json
cd frontend && npm run sync-types      # -> frontend/types/trial.ts
```

A backend field rename now becomes a frontend *compile error* instead of an
`undefined` in a table cell. `export_schema.py` also injects
`additionalProperties: false` before writing — without it,
json-schema-to-typescript emits an `[k: string]: unknown` index signature that
silently permits `row.typo`, making the generated types weaker than the
hand-written ones they replaced.

## Evaluation — `evaluate_agent.py`

Scores the agent's live output against an **independent judge** so
`faithfulness` measures actual grounding, not the generator agreeing with
itself:

```bash
.venv/bin/python evaluate_agent.py
# --from-trace <path>   re-score a saved trace, no new agent/generator call
# --dump <path>          save the captured trace for later re-scoring
# --judge-timeout N      raise if faithfulness returns FAILED (default 900s)
```

Needs `NVIDIA_API_KEY` in `.env` — free tier at
[build.nvidia.com](https://build.nvidia.com).

Pipeline: run the agent → capture `(question, retrieved_contexts, answer)` →
shape into a HF `Dataset` → score with `ragas.evaluate()` → print a table.

Three non-obvious fixes were needed to get a real score out of this stack:

- **ragas 0.4.x won't import.** It hard-imports
  `langchain_community.chat_models.vertexai`, which the sunsetting
  `langchain-community` no longer ships. The symbol is used in exactly one
  place — an `isinstance()` check list our judge is never a member of — so
  `evaluate_agent.py` installs an inert stub module before the `ragas` import
  rather than downgrading `langchain-community` (which would drag
  `langchain-core` below 1.0 and break `langgraph`/the agent itself).
- **Column names changed.** Ragas 0.4 requires `user_input` /
  `retrieved_contexts` / `response`, not the `question`/`contexts`/`answer`
  trio from 0.1.x.
- **`ChatNVIDIA`'s judge calls die mid-flight.** Its `aiohttp` client has no
  configurable socket-read timeout, and reliably hit
  `SocketTimeoutError` on real faithfulness jobs (statement generation + a
  verification call per retrieved context). The judge LLM goes through
  `langchain-openai` against NVIDIA's OpenAI-compatible endpoint instead
  (`https://integrate.api.nvidia.com/v1`), whose `httpx` client does take a
  `timeout=`. Embeddings stay on `NVIDIAEmbeddings` — the OpenAI-compatible
  embeddings client can't send the `input_type` field NVIDIA's asymmetric
  embedding model requires, and 400s without it.
- **`asyncio.wait_for` wraps the *entire* metric, not one attempt.**
  `--judge-timeout` bounds statement generation plus every per-context
  verification call combined. A longer answer means more statements, means
  more calls, means a real risk of running past a short ceiling even though
  no single call is slow. `evaluate_agent.py` also caps `max_retries=2`
  (down from ragas' default of 10) — a slow-but-working judge can lose more
  wall-clock to retries than to one unhurried attempt.
- **Judge model choice matters more than the parameter count implies.**
  `nvidia/llama-3.3-nemotron-super-49b-v1.5` (the spec's originally intended
  judge) is a *reasoning* model that spends most of its latency on
  chain-of-thought before emitting the JSON ragas asks for — benchmarked at
  27.1s vs. 4.5s against `nvidia/nemotron-3-super-120b-a12b` on an identical
  extraction prompt. The larger model is MoE with ~12B active parameters and
  returns JSON as its first token, so it's the current default despite the
  bigger nominal size.

## Analysis — `analyze_trace.py`

Explains a faithfulness score by re-running the judge's two internal calls
(statement decomposition, then NLI verdicts) and printing only the claims
that scored 0, with the judge's own `reason`:

```bash
.venv/bin/python analyze_trace.py                              # eval_traces/eval_trace_final.json
.venv/bin/python analyze_trace.py --trace path/to/trace.json
.venv/bin/python analyze_trace.py --chunk-size 4                # if you see 504s
```

Two things worth knowing before reading its output:

- **NCTId attribution is a heuristic, not judge output.** Ragas' Faithfulness
  metric concatenates *all* retrieved contexts into one blob and judges every
  statement against that whole blob — it never records which context
  supported which claim. `LIKELY SOURCE` is our own lexical-overlap guess,
  labeled as such, not a citation.
- **The NLI-verdict call is chunked.** Sent unchunked (all statements in one
  request), it reliably 504'd against NVIDIA's gateway — a ~40-statement
  answer means several thousand tokens of expected JSON output (verdict +
  reason per statement) in a single response. Chunking caps output size per
  call without changing what the judge is allowed to see.
- **Faithfulness has real run-to-run variance.** Re-scoring the *identical*
  saved answer twice produced 40 statements → 0.77 and 25 statements → 0.88 —
  the judge's own statement segmentation isn't fully deterministic even at
  `temperature=0.0`. Treat a single score as a point estimate, not a fixed
  property of the answer; average several runs before gating on it.

## Frontend (`frontend/`)

Next.js 16 (App Router) + Tailwind 4 + TanStack Table v9, rendering the Smart
Table contract. Runs off `frontend/data/mockResponse.json` — a captured
`research_agent.py` response — so UI work costs no LLM calls.

```bash
cd frontend && npm install && npm run dev   # http://localhost:3000
```

| Path | Purpose |
|---|---|
| `types/trial.ts` | TS mirror of the Pydantic contract |
| `data/mockResponse.json` | Captured backend response (11 trials) |
| `components/BriefingCard.tsx` | `narrative_summary` → executive-summary card |
| `components/TrialsTable.tsx` | TanStack grid: linked NCT IDs, pills, badge rule |

**⚠️ TanStack Table v9, not v8.** `useReactTable` and the `get*RowModel()`
helpers no longer exist. v9 composes features explicitly:

```ts
const features = tableFeatures({ rowSortingFeature, sortedRowModel: createSortedRowModel(), sortFns: {...} })
const table = useTable({ features, columns, data })
```

Two v9 gotchas that cost build errors: `row.getVisibleCells()` requires
`columnVisibilityFeature` (core rows expose `getAllCells()`), and a
heterogeneous column array needs an explicit
`ColumnDef<typeof features, TrialRow, any>[]` annotation because each accessor
yields a different `TValue`.

Swapping the mock for the live backend is one line in `app/page.tsx` — replace
the JSON import with a `fetch`; nothing else changes.

## Design notes

- **The archive is a gate, not a side effect.** `archive_to_s3()` re-reads the
  object from S3 and compares byte length and MD5/ETag against what was sent.
  Any mismatch aborts before parsing — a silently truncated archive would
  poison every future replay.
- **Phase normalisation is load-bearing.** The API emits enum tokens
  (`"PHASE3"`), but queries filter on `Phase == 'Phase 3'`. Without normalising
  on write, that filter matches nothing and returns an empty result set with no
  error. `PHASE_LABELS` handles the mapping.
- **`Phase` is a list.** Trials can be registered as `["Phase 2", "Phase 3"]`.
  Qdrant's `MatchValue` matches if *any* array element equals the value, so a
  Phase 2/3 trial correctly satisfies a `Phase 3` filter.
- **Deterministic point ids.** `uuid5(NCT_NAMESPACE, nct_id)` makes re-runs an
  idempotent upsert rather than a duplicate. Qdrant point ids must be int or
  UUID, so the raw `NCT…` string cannot be used directly.
- **Raw pages are stored verbatim.** The archived envelope keeps each upstream
  response untouched under `pages`, adding only request metadata for audit.
- **Payload indexes** on the filtered fields — the reason Qdrant was chosen over
  a plain vector store; without them, filtering degrades to a full scan.
- **Empty summaries are skipped**, not embedded — a zero-content vector would
  still compete for search slots.
- **Payload enrichment.** Each point carries `conditions`, `interventions`
  (`[{type, name}]`), and `studyType` pulled from the v2 API, plus a flattened
  `interventionNames` list. The raw `interventions` array is the canonical
  shape, but a list of dicts cannot back a plain keyword index — the parallel
  flat list is what makes exact-match drug filtering a one-line `MatchValue`:

  ```python
  Filter(must=[
      FieldCondition(key="interventionNames", match=MatchValue(value="Pembrolizumab")),
      FieldCondition(key="Phase",             match=MatchValue(value="Phase 3")),
  ])
  ```

- **Embedding model: `nomic-embed-text-v1.5` (768-dim, 8192-token window).**
  Verify the tokenizer, not the model card — the previous `all-MiniLM-L6-v2`
  advertises 512 but ships `truncation.max_length = 128`, which silently
  dropped most of every enriched document. Nomic's packaged tokenizer really
  is 8192 (`tokenizer.truncation` reports it, and a 4000-word string encodes
  to 4002 tokens uncut). Current corpus: median doc 290 tokens, max 726 —
  **0/50 truncated**.

  Measured A/B on 37 queries drawn from text that sat *beyond* MiniLM's
  128-token cutoff:

  | model | hit@1 | hit@3 | mean rank |
  |---|---|---|---|
  | all-MiniLM-L6-v2 | 54% | 65% | 6.7 |
  | nomic-embed-text-v1.5 | **86%** | **86%** | **1.9** |

  **Changing this constant requires `--recreate`** — 384-dim and 768-dim
  collections are incompatible, and the vector name changes with the model.
  Keep `EMBEDDING_MODEL` identical across `fetch_and_embed_trials.py`,
  `research_agent.py`, and `verify_qdrant.py` or query vectors will not match
  what was indexed.

## Verified results

`verify_qdrant.py` runs the query twice to prove the filter does real work:

```
semantic query  : 'tumor shrinkage'
payload filter  : Phase == 'Phase 3'
eligible points : 14 of 50

unfiltered top-3 : ['NCT03181867', 'NCT07667400', 'NCT07756840']   ← Phase 2, 1/2, 2
filtered   top-3 : ['NCT06712316', 'NCT06956001', 'NCT06692738']   ← all Phase 3
PASS: all 3 returned results carry Phase == 'Phase 3'.
```

The two sets are fully disjoint — the filter, not ranking luck, produced them.
