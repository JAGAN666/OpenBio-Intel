#!/usr/bin/env bash
# 5-minute quickstart: boots the full stack (Qdrant, Neo4j, FastAPI backend,
# Next.js frontend) via Docker Compose and seeds a small sample corpus (50
# clinical trials + 50 FDA approval records) so http://localhost:3000 is
# immediately interactive.
#
# This is a smoke-test-scale seed, not a research-grade corpus -- see
# seed_bulk_data.py directly for a full ingestion run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

log()  { printf "${BOLD}==>${RESET} %s\n" "$1"; }
warn() { printf "${YELLOW}==>${RESET} %s\n" "$1"; }
fail() { printf "${RED}==>${RESET} %s\n" "$1" >&2; exit 1; }

# --- 1. Check for Docker and Docker Compose ----------------------------------
command -v docker >/dev/null 2>&1 || fail "Docker is not installed. Get it at https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is not available (docker compose, not docker-compose). It ships with modern Docker Desktop / Docker Engine."
docker info >/dev/null 2>&1 || fail "Docker daemon isn't running -- start Docker Desktop (or dockerd) and try again."
log "Docker and Docker Compose found."

# --- 2. Copy .env.example to .env if missing ---------------------------------
if [ ! -f .env ]; then
    cp .env.example .env
    warn "Created .env from .env.example."
    warn "Open .env now and set at least OPENAI_API_KEY and ANTHROPIC_API_KEY, then re-run ./quickstart.sh."
    exit 0
fi

# Required keys must be genuinely set (not just present-but-blank) before
# continuing -- a blank OPENAI_API_KEY would fail loudly and confusingly
# much later (mid-embedding-call), not here where it's actually diagnosable.
missing=()
grep -qE '^OPENAI_API_KEY=.+' .env || missing+=("OPENAI_API_KEY")
grep -qE '^ANTHROPIC_API_KEY=.+' .env || missing+=("ANTHROPIC_API_KEY")
if [ "${#missing[@]}" -gt 0 ]; then
    fail "Missing required value(s) in .env: ${missing[*]}. Edit .env and re-run."
fi
log ".env found with required keys set."

# --- 3. Boot Qdrant, Neo4j, backend, frontend ---------------------------------
log "Building and starting the stack (first build compiles knowledge-graph deps -- can take several minutes; cached after)..."
docker compose up -d --build qdrant neo4j backend frontend

log "Waiting for the backend to report healthy..."
attempts=0
until [ "$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q backend)" 2>/dev/null)" = "healthy" ]; do
    attempts=$((attempts + 1))
    if [ "$attempts" -gt 60 ]; then
        fail "Backend did not become healthy in time. Check logs: docker compose logs backend"
    fi
    sleep 5
done
log "Backend is healthy."

# --- 4. Lightweight sample seed ------------------------------------------------
log "Seeding a sample corpus (50 clinical trials + 50 FDA approval records)..."
log "This uses --skip-s3 (no AWS needed) and --skip-neo4j (knowledge-graph entity"
log "resolution is skipped for this quick seed -- see seed_bulk_data.py to include it)."
docker compose --profile seed run --rm seed \
    python seed_bulk_data.py --source all --endpoint drugsfda --limit 50 --skip-s3 --skip-neo4j

echo
printf "${GREEN}${BOLD}Ready.${RESET} Open ${BOLD}http://localhost:3000${RESET} and try one of the example queries.\n"
echo "API docs: http://127.0.0.1:8000/docs"
echo "Logs:     docker compose logs -f backend"
echo "Stop:     docker compose down   (add -v to also delete the seeded data)"
