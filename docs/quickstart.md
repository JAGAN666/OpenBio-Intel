# Quickstart

Requires [Docker](https://docs.docker.com/get-docker/) + Docker Compose, an [OpenAI API key](https://platform.openai.com/api-keys) (embeddings — required), and an [Anthropic key](https://console.anthropic.com/) (orchestration — or switch `LLM_PROVIDER` to `kimi`/`nvidia`).

```bash
git clone https://github.com/JAGAN666/OpenBio-Intel.git
cd OpenBio-Intel
./quickstart.sh
```

The script checks Docker, copies `.env.example` → `.env` (pausing for you to add keys), starts Qdrant + Neo4j + Postgres + backend + frontend, and runs a lightweight seed (50 trials + 50 FDA records) so the UI is immediately interactive at **http://localhost:3000**.

## Full corpus

The demo seed proves the pipeline; the real product is the full corpus:

```bash
# ClinicalTrials.gov (~600K trials) + openFDA -- hours of runtime,
# real embedding API cost. Read the script header first.
uv run python seed_bulk_data.py --source all

# Knowledge graph (RxNorm entity resolution -- needs the py3.11 venv,
# see build_kg.py's docstring)
uv run python build_kg.py

# FDA Complete Response Letters (459 letters, minutes)
uv run python fetch_fda_crls.py

# Orange/Purple Book exclusivity -> Neo4j (minutes)
uv run python fetch_exclusivity.py
```

## Local development

```bash
uv sync --locked                      # Python env from the committed lockfile
docker compose up -d qdrant neo4j postgres
uv run uvicorn api:app --reload --port 8000
uv run python worker.py               # job-queue worker (second terminal)
cd frontend && npm ci && npm run dev  # http://localhost:3000
```

## MCP (Claude Desktop / Claude Code)

```json
{
  "mcpServers": {
    "openbio-intel": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/OpenBio-Intel", "python", "mcp_server.py"]
    }
  }
}
```

All ten intelligence tools plus the full Smart Table pipeline become available inside your agent, served from your own stack.
