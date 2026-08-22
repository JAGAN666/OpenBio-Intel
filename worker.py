"""Job-queue worker: claims jobs from Postgres and runs the agent pipelines.

Run as its own ECS service (same image as the backend, command override
["python", "worker.py"] -- see terraform's aws_ecs_task_definition.worker)
or locally with `uv run python worker.py`.

RESUME SEMANTICS (the reason this file exists at all): the research graph
is compiled with a LangGraph PostgresSaver checkpointer, thread_id = the
job id. When a worker dies mid-run (deploy, OOM, task recycle),
reclaim_stale() requeues the job and the next attempt calls the graph with
input=None on the SAME thread -- LangGraph replays from the last completed
super-step instead of restarting, so completed tool rounds and extraction
batches are not re-bought from the LLM APIs. Checkpoint granularity is the
super-step: a crash mid-Map loses only that Map batch's partial work.

Landscape/catalyst jobs run through their existing blocking helpers with
coarse progress events (start -> synthesizing -> result): they are single
retrieve+synthesize shapes with nothing meaningful to checkpoint between.
"""
from __future__ import annotations

import json
import os
import time
import traceback

import psycopg
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

import jobs  # noqa: E402
from jobs import claim_next, emit, fail, finish, init_schema, reclaim_stale  # noqa: E402

POLL_SECONDS = 2
STALE_CHECK_EVERY = 30  # poll loops between reclaim_stale sweeps
# How long a 'running' claim stays valid before a dead worker's job is
# requeued. Production default 20min (longer than any legitimate run);
# tests set it to 0 to reclaim immediately.
LEASE_MINUTES = int(os.getenv("WORKER_LEASE_MINUTES", "20"))

# Mirrors api.py's node-label map -- the frontend's ProgressPanel keys off
# these display names, so worker-emitted events must match what the old
# in-request SSE stream emitted.
NODE_STATUS_LABELS = {
    "intent_classifier": "IntentClassifier",
    "agent": "Agent",
    "tools": "ToolNode",
    "out_of_domain": "OutOfDomain",
    "no_results_fallback": "NoResultsFallback",
    "synthesize_table": "SynthesizeTable",
}


def _initial_research_state(query: str) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "tenant_id": None,
        "tool_rounds": 0, "result": None,
        "is_in_domain": None, "has_results": None,
        "synthesis_retries": 0, "synthesis_error": None,
        "retrieved_trials": [], "extracted_rows": [], "retrieved_literature": [],
        "retrieved_fda": [], "retrieved_crls": [], "retrieved_safety": [],
        "retrieved_exclusivity": [],
    }


def run_research(conn: psycopg.Connection, job: dict, graph) -> None:
    """Stream the research graph, translating LangGraph debug events into
    job_events rows -- the sync twin of api.py's _stream_events."""
    job_id = job["id"]
    query = job["payload"]["query"]
    config = {"recursion_limit": 25, "configurable": {"thread_id": job_id}}

    # Attempt 1 starts fresh; attempt >1 resumes the checkpointed thread by
    # passing None as input. If the first attempt died before its first
    # checkpoint there is nothing to resume -- fall back to a fresh start.
    resume = job["attempts"] > 1
    graph_input = None if resume else _initial_research_state(query)
    if resume:
        try:
            if graph.get_state(config).next == ():
                graph_input = _initial_research_state(query)
        except Exception:  # noqa: BLE001 -- no checkpoint at all
            graph_input = _initial_research_state(query)
        if graph_input is None:
            emit(conn, job_id, "status",
                 {"node": "Worker", "phase": "resumed-from-checkpoint"})

    map_total = 0
    map_done = 0
    map_announced = False
    final_result = None

    for event in graph.stream(graph_input, config=config, stream_mode="debug"):
        etype = event.get("type")
        payload = event.get("payload") or {}
        name = payload.get("name")

        if etype == "task":
            if name == "extract_trial":
                map_total += 1
                continue
            emit(conn, job_id, "status",
                 {"node": NODE_STATUS_LABELS.get(name, name), "phase": "start"})
            continue
        if etype != "task_result":
            continue
        if name == "extract_trial":
            map_done += 1
            if not map_announced:
                emit(conn, job_id, "status",
                     {"node": "MapWorkers", "phase": "start", "total": map_total})
                map_announced = True
            emit(conn, job_id, "progress",
                 {"node": "MapWorkers", "completed": map_done, "total": map_total})
            continue

        label = NODE_STATUS_LABELS.get(name, name)
        if payload.get("error"):
            emit(conn, job_id, "status",
                 {"node": label, "phase": "error", "error": str(payload["error"])})
            continue
        emit(conn, job_id, "status", {"node": label, "phase": "done"})

        result_update = (payload.get("result") or {})
        maybe = result_update.get("result")
        if maybe is not None:
            final_result = maybe

    if final_result is None:
        # Resume edge case: the graph already finished before the worker
        # died (result computed, jobs row never updated) -- pull it from
        # the checkpointed state instead of calling it a failure.
        state = graph.get_state(config)
        maybe = (state.values or {}).get("result")
        if maybe is None:
            raise RuntimeError("graph ended without producing a result")
        final_result = maybe

    finish(conn, job_id, json.loads(final_result.model_dump_json()))


def run_landscape(conn: psycopg.Connection, job: dict, graph) -> None:
    from research_agent import run_landscape_query
    emit(conn, job["id"], "status", {"node": "Retrieve+Synthesize", "phase": "start"})
    result = run_landscape_query(graph, job["payload"]["therapeutic_area"])
    if result is None:
        raise RuntimeError("landscape graph produced no structured result")
    finish(conn, job["id"], json.loads(result.model_dump_json()))


def run_catalysts(conn: psycopg.Connection, job: dict, graph) -> None:
    from research_agent import run_catalyst_query
    emit(conn, job["id"], "status", {"node": "Retrieve+Synthesize", "phase": "start"})
    result = run_catalyst_query(graph, job["payload"]["query"])
    if result is None:
        raise RuntimeError("catalyst graph produced no structured result")
    finish(conn, job["id"], json.loads(result.model_dump_json()))


def main() -> None:
    init_schema()
    print(f"[worker] schema ready; connecting graphs "
          f"(pid={os.getpid()})", flush=True)

    # Checkpointer gets its OWN long-lived autocommit connection (LangGraph
    # manages its transactions itself).
    from langgraph.checkpoint.postgres import PostgresSaver
    ckpt_conn = psycopg.connect(jobs.DATABASE_URL, autocommit=True)
    checkpointer = PostgresSaver(ckpt_conn)
    checkpointer.setup()

    from research_agent import (
        DEFAULT_MODEL, make_catalyst_graph, make_graph, make_landscape_graph,
    )
    research_graph = make_graph(DEFAULT_MODEL, verbose=False,
                                checkpointer=checkpointer)
    landscape_graph = make_landscape_graph(DEFAULT_MODEL, verbose=False)
    catalyst_graph = make_catalyst_graph(verbose=False)
    runners = {
        "research": lambda c, j: run_research(c, j, research_graph),
        "landscape": lambda c, j: run_landscape(c, j, landscape_graph),
        "catalysts": lambda c, j: run_catalysts(c, j, catalyst_graph),
    }
    print("[worker] graphs compiled; polling for jobs", flush=True)

    conn = jobs.connect()
    loops = 0
    while True:
        loops += 1
        if loops % STALE_CHECK_EVERY == 1:
            n = reclaim_stale(conn, lease_minutes=LEASE_MINUTES)
            if n:
                print(f"[worker] requeued {n} stale job(s)", flush=True)

        job = claim_next(conn)
        if job is None:
            time.sleep(POLL_SECONDS)
            continue

        print(f"[worker] claimed {job['id']} ({job['type']}, "
              f"attempt {job['attempts']})", flush=True)
        started = time.time()
        try:
            runners[job["type"]](conn, job)
            print(f"[worker] done {job['id']} in {time.time()-started:.1f}s",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 -- job isolation boundary
            requeue = job["attempts"] < 3
            print(f"[worker] job {job['id']} failed "
                  f"(attempt {job['attempts']}, requeue={requeue}): {exc}",
                  flush=True)
            traceback.print_exc()
            try:
                fail(conn, job["id"], str(exc), requeue=requeue)
            except Exception:  # noqa: BLE001 -- conn itself may be broken
                conn = jobs.connect()
                fail(conn, job["id"], str(exc), requeue=requeue)


if __name__ == "__main__":
    main()
