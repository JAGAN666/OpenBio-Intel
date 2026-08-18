# Contributing to medical-rag

Thanks for considering a contribution. This is a young open-source project — issues, PRs, and design discussion are all welcome.

## Development setup

```bash
git clone https://github.com/JAGAN666/medical-rag.git
cd medical-rag
cp .env.example .env   # fill in at least OPENAI_API_KEY and ANTHROPIC_API_KEY

# Infra (Qdrant + Neo4j) via Docker
docker compose up -d qdrant neo4j

# Python environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Backend
.venv/bin/python -m uvicorn api:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

`./quickstart.sh` covers the fully-containerized path (including a sample seed) if you'd rather not run Python/Node locally at all — see the [README](README.md#5-minute-quickstart).

### Knowledge-graph / ingestion work

If you're touching `build_kg.py`, `fetch_pubmed.py`'s entity-resolution step, or anything else that needs scispacy: that dependency chain (scispacy + its RxNorm linker + `nmslib-metabrainz`) has real platform constraints — see `Dockerfile.etl`'s own comments, and `docker-compose.yml`'s `seed` service if you're on Apple Silicon (it needs `platform: linux/amd64` — no arm64 wheel exists for `nmslib-metabrainz` upstream). The isolated `.venv-kg` pattern documented in `build_kg.py`'s module docstring exists specifically because the main project venv's Python version has no wheels for that dependency chain either.

## Before opening a PR

This project doesn't (yet) have a formal `pytest` suite — verification here has consistently meant checking real behavior against real infrastructure, not mocks. Please follow the same discipline:

- **Python**: run the same syntax/correctness check CI runs before pushing:
  ```bash
  pip install flake8
  flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv,.venv-kg,node_modules,frontend,.git
  ```
  (Scoped to genuine syntax errors and undefined names, not style — this repo doesn't enforce a formatter like `black`.)
- **Frontend**: `cd frontend && npm run lint && npx tsc --noEmit` must both pass clean.
- **If you changed the agent's tools, retrieval, or synthesis logic**: run it against something real before calling it done —
  ```bash
  .venv/bin/python research_agent.py --question "your test question"
  ```
  and read the actual output, not just "it didn't crash." If you changed a Qdrant collection's schema, verify the point count and a real query, not just that ingestion completed.
- **If you changed a Pydantic response model** (`SmartTableResponse`, `LandscapeMatrix`, `CatalystTimeline`): regenerate the frontend types rather than hand-editing them —
  ```bash
  .venv/bin/python export_schema.py
  cd frontend && npm run sync-types        # or sync-types:landscape / sync-types:catalysts
  ```

**A real, contribution-friendly `pytest` suite (unit tests for the deterministic pieces — phase normalization, date parsing, snapshot-based schema checks — that don't need a live LLM/Qdrant) would be a genuinely valuable contribution.** If you're looking for a substantial first PR, that's a good one.

## Commit / PR guidelines

- Keep commits focused — one logical change per commit, with a message that explains *why*, not just *what* (the diff already shows what changed).
- If you're fixing a bug, say what the actual failure was, not just "fix bug."
- Open the PR against `main`. CI (`.github/workflows/deploy.yml`'s `Test & Lint` job) must pass before merge.
- For anything beyond a small fix — a new data source, a new agent tool, a new page — consider opening an issue first to discuss the approach. Federating in a new retrieval source touches several places at once (the tool definition, `AgentState`, `_deduped_pools`, the extraction-stage fusion rules) and it's easier to agree on the shape before the diff exists than after.

## Reporting bugs / requesting features

Use the [issue templates](.github/ISSUE_TEMPLATE/) — they ask for the specific detail (repro steps, actual vs. expected, environment) that makes a report actionable.

## Code of conduct

Be respectful, be constructive, assume good faith. This is a technical project built by people who care about getting the details right — disagreement about approach is welcome; personal attacks are not.
