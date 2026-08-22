"""Postgres-backed job queue -- the async backbone for long-running agent runs.

Why this exists: every research/landscape/catalyst request used to execute
INSIDE the HTTP request. A 2-10 minute synchronous request dies with any
ALB hiccup, ECS deploy, or closed browser tab -- and takes all its spent
LLM tokens with it. Jobs decouple acceptance from execution:

    POST /api/jobs           -> {job_id}   (returns in milliseconds)
    worker.py                -> claims queued jobs, runs the graph,
                                streams progress into job_events
    GET  /api/jobs/{id}/stream -> SSE that REPLAYS buffered events then
                                follows live ones -- a refreshed tab
                                reattaches mid-run losing nothing

Design choices (deliberate deviations from the usual Redis+Celery stack):
- ONE Postgres for queue + events + LangGraph checkpoints. SELECT ... FOR
  UPDATE SKIP LOCKED is the battle-tested relational job queue; pg_notify
  covers the pub/sub need; adding Redis would be a second stateful service
  for nothing this workload needs.
- Events are rows, not ephemeral pub/sub messages: replayability is the
  feature (reconnect, post-hoc inspection), and a query emits tens of
  events, not thousands.
- Claiming sets status=running + a lease (claimed_at); a worker that dies
  mid-run leaves the row visible to reclaim_stale(), and the LangGraph
  checkpointer (worker.py) makes the retry resume rather than restart.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    # local docker-compose default (host port 5433 -- 5432 is commonly taken)
    "postgresql://openbio:openbio@localhost:5433/openbio",
)

JOB_TYPES = ("research", "landscape", "catalysts")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    payload     JSONB NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',  -- queued|running|done|failed
    attempts    INT  NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 3,
    result      JSONB,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status, created_at);

CREATE TABLE IF NOT EXISTS job_events (
    id       BIGSERIAL PRIMARY KEY,
    job_id   TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    event    TEXT NOT NULL,     -- status|progress|result|error (SSE event name)
    data     JSONB NOT NULL,
    ts       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS job_events_job_idx ON job_events (job_id, id);
"""


def connect(autocommit: bool = True) -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=autocommit, row_factory=dict_row)


def init_schema(conn: Optional[psycopg.Connection] = None) -> None:
    own = conn is None
    conn = conn or connect()
    conn.execute(SCHEMA)
    if own:
        conn.close()


# --- producer side (api.py) ---------------------------------------------------
def enqueue(job_type: str, payload: dict) -> str:
    assert job_type in JOB_TYPES, f"unknown job type {job_type}"
    job_id = uuid.uuid4().hex[:16]
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, type, payload) VALUES (%s, %s, %s)",
            (job_id, job_type, json.dumps(payload)),
        )
        conn.execute("SELECT pg_notify('jobs_queued', %s)", (job_id,))
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, type, payload, status, attempts, result, error, "
            "created_at, claimed_at, finished_at FROM jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
    return row


def events_since(job_id: str, after_id: int = 0) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "SELECT id, event, data FROM job_events "
            "WHERE job_id = %s AND id > %s ORDER BY id",
            (job_id, after_id),
        ).fetchall()


# --- worker side ---------------------------------------------------------------
def claim_next(conn: psycopg.Connection) -> Optional[dict]:
    """Claim one queued job. SKIP LOCKED means N workers never double-claim;
    the transaction commits the status flip atomically with the claim."""
    with conn.transaction():
        row = conn.execute(
            "SELECT id, type, payload, attempts FROM jobs "
            "WHERE status = 'queued' ORDER BY created_at "
            "FOR UPDATE SKIP LOCKED LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE jobs SET status = 'running', attempts = attempts + 1, "
            "claimed_at = now() WHERE id = %s",
            (row["id"],),
        )
    # Post-increment view: the returned attempts is THIS attempt's number
    # (1 = first execution), matching what logs and resume logic expect.
    row["attempts"] += 1
    return row


def reclaim_stale(conn: psycopg.Connection, lease_minutes: int = 20) -> int:
    """Return crashed-worker jobs to the queue (or fail them out of
    attempts). A running job whose lease expired means its worker died --
    ECS task recycled, deploy, OOM -- because live workers finish or fail
    jobs explicitly."""
    with conn.transaction():
        requeued = conn.execute(
            "UPDATE jobs SET status = 'queued' "
            "WHERE status = 'running' "
            "  AND claimed_at < now() - make_interval(mins => %s) "
            "  AND attempts < max_attempts "
            "RETURNING id",
            (lease_minutes,),
        ).fetchall()
        conn.execute(
            "UPDATE jobs SET status = 'failed', "
            "  error = 'worker died and attempts exhausted', finished_at = now() "
            "WHERE status = 'running' "
            "  AND claimed_at < now() - make_interval(mins => %s) "
            "  AND attempts >= max_attempts",
            (lease_minutes,),
        )
    return len(requeued)


def emit(conn: psycopg.Connection, job_id: str, event: str, data: Any) -> None:
    conn.execute(
        "INSERT INTO job_events (job_id, event, data) VALUES (%s, %s, %s)",
        (job_id, event, json.dumps(data)),
    )
    conn.execute("SELECT pg_notify(%s, %s)", (f"job_{job_id}", event))


def finish(conn: psycopg.Connection, job_id: str, result: Any) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'done', result = %s, finished_at = now() "
        "WHERE id = %s",
        (json.dumps(result), job_id),
    )
    emit(conn, job_id, "result", result)


def fail(conn: psycopg.Connection, job_id: str, error: str, requeue: bool) -> None:
    if requeue:
        conn.execute("UPDATE jobs SET status = 'queued' WHERE id = %s", (job_id,))
        emit(conn, job_id, "status", {"node": "Worker", "phase": "retrying",
                                      "error": error[:500]})
    else:
        conn.execute(
            "UPDATE jobs SET status = 'failed', error = %s, finished_at = now() "
            "WHERE id = %s",
            (error[:2000], job_id),
        )
        emit(conn, job_id, "error", {"message": error[:2000]})
