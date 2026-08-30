"""Integration tests for the Postgres-backed job queue accounting."""

import uuid

import jobs
import pytest


@pytest.fixture()
def db():
    """Use the throwaway Postgres service from docker-compose.yml."""
    try:
        conn = jobs.connect()
        conn.execute("SELECT 1")
        jobs.init_schema(conn)
    except Exception as exc:
        pytest.fail(
            "Postgres is required for jobs tests; start it with "
            f"`docker compose up -d postgres`: {exc}"
        )

    job_ids = []
    try:
        yield conn, job_ids
    finally:
        for job_id in job_ids:
            conn.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
        conn.close()


def _insert_job(conn, job_ids, *, status="queued", attempts=0, max_attempts=3,
                stale=False):
    job_id = uuid.uuid4().hex
    claimed_at = "now() - interval '30 minutes'" if stale else "NULL"
    conn.execute(
        "INSERT INTO jobs (id, type, payload, status, attempts, max_attempts, "
        f"claimed_at) VALUES (%s, 'research', '{{}}'::jsonb, %s, %s, %s, {claimed_at})",
        (job_id, status, attempts, max_attempts),
    )
    job_ids.append(job_id)
    return job_id


def _row(conn, job_id):
    return conn.execute(
        "SELECT status, attempts, result, error, claimed_at, finished_at "
        "FROM jobs WHERE id = %s",
        (job_id,),
    ).fetchone()


def test_claim_retry_and_finish_preserve_attempt_accounting(db):
    conn, job_ids = db
    job_id = _insert_job(conn, job_ids)

    first = jobs.claim_next(conn)
    assert first["id"] == job_id
    assert first["attempts"] == 1
    assert _row(conn, job_id)["status"] == "running"
    assert _row(conn, job_id)["attempts"] == 1

    jobs.fail(conn, job_id, "temporary worker error", requeue=True)
    assert _row(conn, job_id)["status"] == "queued"
    assert _row(conn, job_id)["attempts"] == 1

    second = jobs.claim_next(conn)
    assert second["id"] == job_id
    assert second["attempts"] == 2
    assert _row(conn, job_id)["attempts"] == 2

    jobs.finish(conn, job_id, {"answer": 42})
    row = _row(conn, job_id)
    assert row["status"] == "done"
    assert row["attempts"] == 2
    assert row["result"] == {"answer": 42}
    assert row["finished_at"] is not None


def test_reclaim_stale_requeues_jobs_with_attempts_remaining(db):
    conn, job_ids = db
    job_id = _insert_job(conn, job_ids, status="running", attempts=1, stale=True)

    assert jobs.reclaim_stale(conn, lease_minutes=20) == 1
    row = _row(conn, job_id)
    assert row["status"] == "queued"
    assert row["attempts"] == 1
    assert row["error"] is None


def test_reclaim_stale_fails_jobs_after_max_attempts(db):
    conn, job_ids = db
    job_id = _insert_job(
        conn, job_ids, status="running", attempts=3, max_attempts=3, stale=True
    )

    assert jobs.reclaim_stale(conn, lease_minutes=20) == 0
    row = _row(conn, job_id)
    assert row["status"] == "failed"
    assert row["attempts"] == 3
    assert row["error"] == "worker died and attempts exhausted"
    assert row["finished_at"] is not None
