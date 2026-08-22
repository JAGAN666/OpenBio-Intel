# API & MCP

The FastAPI backend publishes its full OpenAPI schema at `/docs` on any running instance. The essentials:

## Jobs (the primary surface)

Long-running agent queries execute as durable jobs — never inside the HTTP request.

```bash
# enqueue (returns in milliseconds)
curl -X POST $BASE/api/jobs -H "Content-Type: application/json" \
  -d '{"type": "research", "query": "Which trials use pembrolizumab?"}'
# -> {"job_id": "..."}

# follow progress: SSE that REPLAYS the full event log, then live events.
# GET, so a browser EventSource gets automatic reconnection for free.
curl -N $BASE/api/jobs/{job_id}/stream

# or poll
curl $BASE/api/jobs/{job_id}
```

`type` is `research` (Smart Table), `landscape`, or `catalysts`. Events are `status`, `progress`, and terminal `result`/`error` — a reconnecting client rebuilds exact progress state from the replay.

## Watchlist

```bash
curl $BASE/api/watchlist                          # list
curl -X POST $BASE/api/watchlist \
  -d '{"type": "company", "value": "Coherus BioSciences"}'
curl -X POST $BASE/api/watchlist/check            # diff both live sources now
curl $BASE/api/watchlist/digest                   # latest change digest
```

## Exports

`POST /api/export/excel` and `POST /api/export/pptx` take a `SmartTableResponse` body (i.e. a job's `result`) and return the file — no agent re-run.

## Synchronous endpoints

`POST /api/research`, `/api/landscape`, `/api/catalysts` still execute in-request — kept for CLI use and simple scripting. For anything user-facing, use jobs.

## MCP server

`mcp_server.py` exposes ten tools over stdio to any MCP client:

`search_clinical_trials` · `query_knowledge_graph` · `search_fda_records` · `search_fda_crls` · `search_pubmed_literature` · `search_sec_filings` · `search_corporate_news` · `get_safety_signals` · `get_exclusivity` · `run_smart_table`

Each returns the same JSON observations the in-app agent consumes, including the `has_results` grounding flag. Configuration is in the [Quickstart](quickstart.md#mcp-claude-desktop-claude-code).
