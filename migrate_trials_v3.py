"""Re-ingest the ClinicalTrials.gov corpus with the v3 record schema.

Why a re-crawl: the live collection stores nine CT.gov fields; the
exactness + mechanism work needs ~25 (intervention descriptions and
otherNames, arm groups, detailed description, dates, design, outcomes,
eligibility, countries, collaborators -- see ct_schema.API_FIELDS). The
registry is the only source for them, so every study is fetched again.

Why NOT a full re-embed: dense vectors cost money and hours; the old
embedding text is a strict subset of the new one, so the existing vector
is a sound approximation. Points whose NCT id already exists in the live
collection COPY their dense vector (deterministic uuid5 point ids make
this a direct lookup); only new studies are embedded. BM25 sparse vectors
and payload indexes are rebuilt locally from the new payload. Pass
--reembed to refresh every dense vector instead.

Neo4j: Trial nodes gain the v3 properties and every INVESTIGATES edge
gets a `role` (studied / comparator / placebo), without re-running
scispacy linking -- existing Drug/Concept nodes are kept, and drug-like
interventions that have no Drug node yet (BIOLOGICALs were never linked)
get a name-only node via MERGE, exactly like build_kg's unlinked path.

Online-safe: builds `clinical_trials_v3` beside the live alias, then
--swap repoints the `clinical_trials` alias (same pattern as
migrate_hybrid.py). Resumable: --resume-from <pageToken> continues a
crawl; already-upserted points are idempotent upserts.

Usage (local):
    uv run python migrate_trials_v3.py                 # crawl + build + KG refresh
    uv run python migrate_trials_v3.py --swap          # cut the alias over
    uv run python migrate_trials_v3.py --limit 2000    # smoke test
AWS: one-off ECS run-task on the backend task definition with the same
command override (the image ships this file).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models as qmodels

import ct_schema
from embeddings import embed_documents, vector_params
from sparse_embeddings import SPARSE_VECTOR_NAME, embed_docs, sparse_vector_params, trial_sparse_text

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

ALIAS = "clinical_trials"
TARGET = "clinical_trials_v3"
NCT_NAMESPACE = uuid.UUID("6f3a1d4c-9b2e-4c7a-8f11-0d5e2a7c4b93")  # = fetch_and_embed_trials
CT_API_URL = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE = 1000
UPSERT_BATCH = 256
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw" / "v3_migration"

KG_TRIAL_UPDATE = """
UNWIND $rows AS r
MERGE (t:Trial {id: r.nct_id})
SET t.title = r.title, t.phase = r.phase, t.status = r.status,
    t.sponsor = r.sponsor, t.collaborators = r.collaborators,
    t.study_type = r.study_type, t.conditions = r.conditions,
    t.summary = r.summary, t.intervention_names = r.intervention_names,
    t.studied_intervention_names = r.studied_intervention_names,
    t.interventions_json = r.interventions_json,
    t.arm_groups_json = r.arm_groups_json,
    t.start_date = r.start_date, t.start_year = r.start_year,
    t.primary_completion_date = r.primary_completion_date,
    t.enrollment = r.enrollment, t.countries = r.countries,
    t.design_json = r.design_json
WITH t, r
UNWIND r.drugs AS dr
MERGE (d:Drug {name: dr.name})
ON CREATE SET d.other_names = dr.other_names
MERGE (t)-[i:INVESTIGATES]->(d)
SET i.role = dr.role
"""


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "openbio-intel-v3-migration/1.0",
                      "Accept": "application/json"})
    return s


def _page(session: requests.Session, token: str | None) -> dict:
    params = {"pageSize": PAGE_SIZE, "fields": ",".join(ct_schema.API_FIELDS),
              "countTotal": "true"}
    if token:
        params["pageToken"] = token
    for attempt in range(6):
        try:
            r = session.get(CT_API_URL, params=params, timeout=120)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            wait = min(60, 2 ** attempt)
            print(f"[fetch] {exc} -- retry in {wait}s")
            time.sleep(wait)
    raise SystemExit("[fetch] CT.gov unreachable after retries")


def ensure_target(client: QdrantClient) -> None:
    if not client.collection_exists(TARGET):
        client.create_collection(
            TARGET, vectors_config=vector_params(),
            sparse_vectors_config=sparse_vector_params(), on_disk_payload=True)
        print(f"[qdrant] created {TARGET}")
    from fetch_and_embed_trials import ensure_payload_indexes
    ensure_payload_indexes(client, TARGET)


def _existing_vectors(client: QdrantClient, source: str, ids: list[str]) -> dict:
    """Dense vectors for the given point ids from the live collection."""
    out = {}
    for i in range(0, len(ids), UPSERT_BATCH):
        chunk = ids[i:i + UPSERT_BATCH]
        try:
            pts = client.retrieve(source, ids=chunk, with_payload=False, with_vectors=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[qdrant] retrieve failed ({exc}); embedding this chunk fresh")
            continue
        for p in pts:
            v = p.vector if not isinstance(p.vector, dict) else p.vector.get("")
            if v:
                out[str(p.id)] = v
    return out


def kg_refresh(driver, payloads: list[dict]) -> int:
    rows = []
    for pl in payloads:
        props = ct_schema.trial_kg_props(pl)
        roles = pl.get("interventionRoles") or {}
        props["drugs"] = [
            {"name": iv["name"], "other_names": iv.get("otherNames") or [],
             "role": roles.get(iv["name"], "studied")}
            for iv in pl.get("interventions") or []
            if iv.get("type") in ct_schema.STUDIED_TYPES
        ]
        rows.append(props)
    if not rows:
        return 0
    with driver.session() as s:
        s.run(KG_TRIAL_UPDATE, rows=rows).consume()
    return len(rows)


def build(args) -> None:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    ensure_target(client)
    source = ALIAS if client.collection_exists(ALIAS) else None
    driver = None
    if not args.skip_neo4j:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        with driver.session() as s:
            s.run("CREATE INDEX trial_id IF NOT EXISTS FOR (t:Trial) ON (t.id)").consume()
            s.run("CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)").consume()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = _session()
    token = args.resume_from
    total_seen = total_indexed = embedded = reused = kg_rows = 0
    pages = 0
    started = time.perf_counter()
    while True:
        page = _page(session, token)
        studies = page.get("studies") or []
        if not studies:
            break
        pages += 1
        total_seen += len(studies)
        if pages == 1:
            print(f"[fetch] upstream totalCount={page.get('totalCount')}")
        (RAW_DIR / f"page_{pages:05d}.json").write_text(
            __import__("json").dumps(page), encoding="utf-8")

        records = []
        for st in studies:
            pl = ct_schema.build_trial_payload(st, None)
            if pl is None:
                continue
            records.append({
                "id": str(uuid.uuid5(NCT_NAMESPACE, pl["NCTId"])),
                "payload": pl,
                "document": ct_schema.build_embedding_text(pl),
            })
            if args.limit and total_indexed + len(records) >= args.limit:
                break

        dense: dict[str, list[float]] = {}
        if source and not args.reembed:
            dense = _existing_vectors(client, source, [r["id"] for r in records])
        missing = [r for r in records if r["id"] not in dense]
        if missing:
            vecs = embed_documents([r["document"] for r in missing])
            for r, v in zip(missing, vecs):
                dense[r["id"]] = v
            embedded += len(missing)
        reused += len(records) - len(missing)

        sparse = embed_docs([trial_sparse_text(r["payload"]) for r in records])
        points = [qmodels.PointStruct(
            id=r["id"], vector={"": dense[r["id"]], SPARSE_VECTOR_NAME: sv},
            payload=r["payload"]) for r, sv in zip(records, sparse)]
        for i in range(0, len(points), UPSERT_BATCH):
            client.upsert(TARGET, points=points[i:i + UPSERT_BATCH], wait=False)
        total_indexed += len(records)

        if driver is not None:
            kg_rows += kg_refresh(driver, [r["payload"] for r in records])

        rate = total_indexed / max(1e-6, time.perf_counter() - started)
        print(f"[build] page {pages}: +{len(records)} (total {total_indexed}, "
              f"reused {reused}, embedded {embedded}, kg {kg_rows}) "
              f"{rate:.0f} pts/s  nextPageToken={page.get('nextPageToken')}")

        token = page.get("nextPageToken")
        if not token or (args.limit and total_indexed >= args.limit):
            break

    time.sleep(3)
    print(f"[build] done: {client.count(TARGET).count} points in {TARGET} "
          f"({(time.perf_counter() - started) / 60:.1f} min)")
    if driver is not None:
        driver.close()


def swap() -> None:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    target_count = client.count(TARGET).count
    live = None
    for a in client.get_aliases().aliases:
        if a.alias_name == ALIAS:
            live = a.collection_name
    live_count = client.count(ALIAS).count if live else 0
    print(f"[swap] {ALIAS} -> {live} ({live_count} pts); {TARGET} has {target_count} pts")
    if target_count < 0.98 * live_count:
        sys.exit("[swap] REFUSING: v3 has fewer than 98% of live points")
    ops = []
    if live:
        ops.append(qmodels.DeleteAliasOperation(
            delete_alias=qmodels.DeleteAlias(alias_name=ALIAS)))
    ops.append(qmodels.CreateAliasOperation(
        create_alias=qmodels.CreateAlias(collection_name=TARGET, alias_name=ALIAS)))
    client.update_collection_aliases(change_aliases_operations=ops)
    print(f"[swap] alias {ALIAS} -> {TARGET}; count({ALIAS}) = {client.count(ALIAS).count}")
    print(f"[swap] old collection {live} left in place -- delete manually once verified")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swap", action="store_true", help="repoint the alias to v3")
    ap.add_argument("--limit", type=int, default=None, help="stop after N studies (smoke test)")
    ap.add_argument("--resume-from", default=None, help="CT.gov pageToken to resume a crawl")
    ap.add_argument("--reembed", action="store_true", help="embed every study fresh")
    ap.add_argument("--skip-neo4j", action="store_true")
    args = ap.parse_args()
    if args.swap:
        swap()
    else:
        build(args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    # fastembed/onnxruntime aborts in its atexit teardown on macOS
    # ("recursive_mutex lock failed") AFTER all work is done; skip it.
    os._exit(rc)
