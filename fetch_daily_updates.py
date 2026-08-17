"""Daily Delta ETL: fetch only records updated/submitted in the last N days
from ClinicalTrials.gov and openFDA drug/drugsfda, embed via the same OpenAI
pipeline as the Historical Seed, upsert into Qdrant, and run build_kg.py's
entity resolution over the ClinicalTrials.gov delta.

This is the "Feed" half of the Seed-and-Feed architecture (seed_bulk_data.py
is the "Seed" half -- a one-time/on-demand full hydration that pages through
the ENTIRE corpus). This script instead asks each source's own API to
pre-filter to only what changed recently, so a routine day's run touches
tens to low-thousands of records, not hundreds of thousands.

    python fetch_daily_updates.py                        # yesterday's deltas, both sources
    python fetch_daily_updates.py --source clinicaltrials --days 1
    python fetch_daily_updates.py --source openfda --days 7 --limit 50   # smoke test
    python fetch_daily_updates.py --skip-s3 --skip-neo4j  # local dev only

RESEARCH NOTE -- ClinicalTrials.gov delta filter, verified live before
writing a line of this script (same discipline as seed_bulk_data.py's own
module docstring):
    filter.advanced=AREA[LastUpdatePostDate]RANGE[from,to] IS the correct,
    documented Essie advanced-filter syntax for the v2 API -- confirmed
    three ways: (1) a deliberately malformed AREA name ("NotARealField")
    returns a genuine HTTP 400 "Unknown area name", proving the API
    validates this syntax rather than silently no-op'ing on a wrong
    parameter; (2) a wide 30-day RANGE returned 17,136 real studies with
    LastUpdatePostDate values genuinely inside that window; (3) a narrow
    single-day RANGE legitimately returned 0 on the day this was verified --
    registry update volume is not guaranteed to be nonzero on any given
    calendar day. That last point is WHY --days is a tunable lookback
    (default 1, i.e. "yesterday" per the spec) rather than hardcoded with no
    recourse for a quiet day, a missed cron run, or a backfill.

RESEARCH NOTE -- openFDA delta filter, verified live and CORRECTED from the
spec's wording:
    The spec's `search=effective_time:[...]` does NOT work against
    drug/drugsfda -- verified directly: a real drugsfda record (fetched
    live) has NO top-level `effective_time` field at all (its top-level
    keys are application_number/openfda/products/sponsor_name/submissions),
    and querying `search=effective_time:...` against this endpoint returns
    HTTP 404 "No matches found!" every time, not a parsing error -- openFDA's
    search does not validate field names against a schema, so this is easy
    to mistake for "correct syntax that just found nothing" rather than
    "wrong field for this endpoint." `effective_time` IS a real openFDA
    field, but it belongs to a DIFFERENT endpoint (drug/label -- an FDA
    label document's effective date); the spec's wording appears to have
    conflated the two. The correct field on drug/drugsfda for "when was
    this application's filing last updated" is the nested
    `submissions.submission_status_date` (format YYYYMMDD, one date per
    submission -- an application can have many) -- confirmed live: a range
    query on it returns real, correctly-dated results (488 matches across
    the last 45 days at verification time).
    Also verified live: a query with genuinely ZERO matches returns HTTP
    404, not an empty 200 -- handled explicitly below rather than treated
    as a crash, since a quiet day is an expected, non-error outcome for a
    delta job.

SCOPE NOTE -- Neo4j / entity resolution:
    Same boundary as seed_bulk_data.py, for the same reason (see that
    script's own SCOPE NOTE): entity resolution runs ONLY for
    ClinicalTrials.gov deltas. openFDA's drugsfda records never had a
    Trial-centric graph schema to attach to, and only `drugsfda` has a live
    Qdrant collection here at all (`openfda_drug_events` was never seeded);
    --endpoint only accepts "drugsfda" for that reason.

SCOPE NOTE -- entity-resolution interpreter, adapted for containerization:
    Locally, build_kg.py runs in its own .venv-kg (Python 3.11) because the
    MAIN project venv is pinned to Python 3.14, which scispacy's dependency
    chain has no wheels for (see build_kg.py's own docstring). Inside
    Dockerfile.etl's python:3.11-slim, that constraint does not exist -- the
    container's single interpreter IS already 3.11, so scispacy installs
    directly into it with no nested venv needed. _resolve_kg_python() below
    picks the right interpreter for whichever context it's running in: an
    explicit KG_PYTHON env var wins if set, else .venv-kg/bin/python if that
    path exists (local dev), else the current interpreter (the container).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from qdrant_client import QdrantClient

from embeddings import EMBEDDING_MODEL, vector_params
from fetch_and_embed_trials import (
    QDRANT_HOST,
    QDRANT_PORT,
    archive_to_s3,
    ensure_collection,
    index_records,
)
from seed_bulk_data import (
    CT_API_FIELDS,
    CT_API_URL,
    CT_PAGE_SIZE,
    OPENFDA_COLLECTIONS,
    _ct_record,
    _ct_session,
    _drugsfda_record,
    ensure_openfda_collection,
)

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_KG_SCRIPT = PROJECT_ROOT / "build_kg.py"

BATCH_SIZE = 1000  # same unit as seed_bulk_data.py -- Qdrant/S3/Neo4j flush size

CT_DELTA_S3_PREFIX = "raw/clinical_trials/daily_delta"
OPENFDA_DELTA_S3_PREFIX = "raw/openfda/daily_delta"

# Verified live against the real endpoint -- see module docstring.
OPENFDA_SEARCH_URL = "https://api.fda.gov/drug/drugsfda.json"
OPENFDA_SEARCH_PAGE_LIMIT = 1000  # openFDA's documented max per-request `limit`


# =============================================================================
# DATE WINDOW
# =============================================================================
def _date_window(days: int) -> tuple[str, str]:
    """(from_date, to_date) as YYYY-MM-DD, UTC -- CT.gov's RANGE[] format."""
    now = datetime.now(timezone.utc).date()
    return (now - timedelta(days=days)).isoformat(), now.isoformat()


# =============================================================================
# ENTITY RESOLUTION -- interpreter resolution, see module SCOPE NOTE above
# =============================================================================
def _resolve_kg_python() -> Path:
    override = os.getenv("KG_PYTHON")
    if override:
        return Path(override)
    local_venv = PROJECT_ROOT / ".venv-kg" / "bin" / "python"
    if local_venv.exists():
        return local_venv
    return Path(sys.executable)


def _run_build_kg(batch_path: Path, verbose: bool) -> None:
    kg_python = _resolve_kg_python()
    if not kg_python.exists() and kg_python != Path(sys.executable):
        print(f"[neo4j]   SKIPPED -- {kg_python} not found (see build_kg.py's "
              f"docstring to set up .venv-kg, or set KG_PYTHON to an "
              f"interpreter with scispacy installed)", file=sys.stderr)
        return
    cmd = [str(kg_python), str(BUILD_KG_SCRIPT), "--input", str(batch_path)]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=not verbose, text=True)
    if result.returncode != 0:
        print(f"[neo4j]   build_kg.py failed (exit {result.returncode}) for {batch_path.name}",
              file=sys.stderr)
        if not verbose and result.stdout:
            print(result.stdout[-2000:], file=sys.stderr)
        if not verbose and result.stderr:
            print(result.stderr[-2000:], file=sys.stderr)
    else:
        print(f"[neo4j]   build_kg.py OK for {batch_path.name}")


# =============================================================================
# CLINICALTRIALS.GOV DELTA
# =============================================================================
def _fetch_ct_delta_page(session: requests.Session, from_date: str, to_date: str,
                         page_token: str | None) -> dict:
    params = {
        "filter.advanced": f"AREA[LastUpdatePostDate]RANGE[{from_date},{to_date}]",
        "pageSize": CT_PAGE_SIZE,
        "fields": ",".join(CT_API_FIELDS),
        "countTotal": "true",
    }
    if page_token:
        params["pageToken"] = page_token
    resp = session.get(CT_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_ct_delta(client: QdrantClient, from_date: str, to_date: str, limit: int | None,
                 skip_s3: bool, skip_neo4j: bool, batch_size: int, verbose: bool) -> dict:
    print("=" * 74)
    print(f"medical-rag :: Daily Delta -- ClinicalTrials.gov  [{from_date} .. {to_date}]")
    print("=" * 74)
    ensure_collection(client, recreate=False)

    session = _ct_session()
    page_token: str | None = None
    total_indexed = 0
    total_pages = 0
    stop = False

    while not stop:
        page = _fetch_ct_delta_page(session, from_date, to_date, page_token)
        studies = page.get("studies", [])
        if not studies:
            break
        total_pages += 1
        print(f"[fetch]   page {total_pages}: {len(studies)} studies "
              f"(totalCount upstream: {page.get('totalCount')})")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        blob = json.dumps(page, indent=2, ensure_ascii=False).encode("utf-8")
        local_dir = PROJECT_ROOT / "data" / "raw" / "daily_delta"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / f"clinical_trials_delta_{stamp}.json"
        local_path.write_bytes(blob)

        s3_key = None
        if not skip_s3:
            s3_key = archive_to_s3(local_path, blob, key_prefix=CT_DELTA_S3_PREFIX,
                                   content_type="application/json")

        records = []
        for study in studies:
            if limit is not None and total_indexed + len(records) >= limit:
                stop = True
                break
            rec = _ct_record(study, s3_key)
            if rec:
                records.append(rec)

        if records:
            index_records(client, records, batch_size)
            total_indexed += len(records)
            print(f"[index]   +{len(records)} points (running total: {total_indexed})")

            if not skip_neo4j:
                kg_payload = {"pages": [{"studies": studies[:len(records)]}]}
                kg_path = local_dir / f"clinical_trials_delta_kg_batch_{stamp}.json"
                kg_path.write_text(json.dumps(kg_payload), encoding="utf-8")
                _run_build_kg(kg_path, verbose)

        page_token = page.get("nextPageToken")
        if not page_token or stop:
            break

    print(f"[done]    {total_indexed} trial(s) indexed across {total_pages} page(s)")
    return {"indexed": total_indexed, "pages": total_pages}


# =============================================================================
# OPENFDA DELTA (drugsfda only -- see module SCOPE NOTE)
# =============================================================================
def _fetch_openfda_delta_page(from_date_c: str, to_date_c: str, skip: int) -> dict:
    params = {
        "search": f"submissions.submission_status_date:[{from_date_c} TO {to_date_c}]",
        "limit": OPENFDA_SEARCH_PAGE_LIMIT,
        "skip": skip,
    }
    resp = requests.get(OPENFDA_SEARCH_URL, params=params, timeout=60)
    if resp.status_code == 404:
        # Verified live -- a genuinely empty result set returns 404, not an
        # empty 200. A quiet day is expected, not a failure; see module
        # docstring.
        return {"meta": {"results": {"total": 0}}, "results": []}
    resp.raise_for_status()
    return resp.json()


def run_openfda_delta(client: QdrantClient, endpoint: str, from_date_c: str, to_date_c: str,
                      limit: int | None, skip_s3: bool, batch_size: int) -> dict:
    print("=" * 74)
    print(f"medical-rag :: Daily Delta -- openFDA drug/{endpoint}  [{from_date_c} .. {to_date_c}]")
    print("=" * 74)

    collection_name = OPENFDA_COLLECTIONS[endpoint]
    ensure_openfda_collection(client, collection_name, recreate=False)

    total_indexed = 0
    total_seen = 0
    skip = 0
    stop = False

    while not stop:
        page = _fetch_openfda_delta_page(from_date_c, to_date_c, skip)
        results = page.get("results", [])
        if not results:
            break
        total = page.get("meta", {}).get("results", {}).get("total", 0)
        print(f"[fetch]   skip={skip}: {len(results)} record(s) (total matching: {total})")

        local_dir = PROJECT_ROOT / "data" / "raw" / "daily_delta"
        local_dir.mkdir(parents=True, exist_ok=True)
        if not skip_s3:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            blob = json.dumps(page, indent=2, ensure_ascii=False).encode("utf-8")
            local_path = local_dir / f"openfda_{endpoint}_delta_{stamp}.json"
            local_path.write_bytes(blob)
            archive_to_s3(local_path, blob, key_prefix=f"{OPENFDA_DELTA_S3_PREFIX}/{endpoint}",
                          content_type="application/json")

        records = []
        for r in results:
            total_seen += 1
            if limit is not None and total_indexed + len(records) >= limit:
                stop = True
                break
            rec = _drugsfda_record(r)
            if rec:
                records.append(rec)

        if records:
            index_records(client, records, batch_size, collection_name=collection_name)
            total_indexed += len(records)
            print(f"[index]   +{len(records)} points (running total: {total_indexed})")

        skip += len(results)
        if skip >= total or stop:
            break

    print(f"[done]    {total_indexed} record(s) indexed into '{collection_name}' "
          f"({total_seen} scanned)")
    return {"indexed": total_indexed, "collection": collection_name}


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["clinicaltrials", "openfda", "all"],
                        default="all")
    parser.add_argument("--endpoint", choices=["drugsfda"], default="drugsfda",
                        help="openFDA endpoint -- only drugsfda has a live "
                             "collection to update (see module SCOPE NOTE)")
    parser.add_argument("--days", type=int, default=1,
                        help="lookback window in days (default 1 = 'yesterday', "
                             "matching the spec's daily-delta cadence). Larger "
                             "values are for backfill/verification after a "
                             "missed cron run, not routine use.")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N records PER SOURCE -- for a smoke "
                             "test, not routine use")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--skip-s3", action="store_true",
                        help="skip archiving (local dev only -- breaks replayability)")
    parser.add_argument("--skip-neo4j", action="store_true",
                        help="skip entity resolution (clinicaltrials only -- "
                             "openFDA never runs it, see module SCOPE NOTE)")
    parser.add_argument("--verbose-kg", action="store_true",
                        help="stream build_kg.py's own output instead of capturing it")
    args = parser.parse_args()

    started = time.time()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        client.get_collections()
    except Exception as exc:
        print(f"[qdrant]  cannot reach Qdrant at {QDRANT_HOST}:{QDRANT_PORT} -> {exc}",
              file=sys.stderr)
        return 1
    vp = vector_params()
    print(f"[qdrant]  {QDRANT_HOST}:{QDRANT_PORT} | model={EMBEDDING_MODEL} "
          f"dim={vp.size} distance={vp.distance}")

    from_date, to_date = _date_window(args.days)
    from_date_c, to_date_c = from_date.replace("-", ""), to_date.replace("-", "")
    print(f"[window]  lookback={args.days}d  {from_date} .. {to_date}")

    results = {}
    if args.source in ("clinicaltrials", "all"):
        results["clinicaltrials"] = run_ct_delta(
            client, from_date, to_date, args.limit, args.skip_s3, args.skip_neo4j,
            args.batch_size, args.verbose_kg)

    if args.source in ("openfda", "all"):
        results[f"openfda_{args.endpoint}"] = run_openfda_delta(
            client, args.endpoint, from_date_c, to_date_c, args.limit,
            args.skip_s3, args.batch_size)

    elapsed = time.time() - started
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"  done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
