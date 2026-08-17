"""Historical Seed: bulk-hydrate S3 + Qdrant (+ Neo4j) from ClinicalTrials.gov
and openFDA, instead of paginating the live APIs one page at a time.

    python seed_bulk_data.py --source clinicaltrials --limit 500
    python seed_bulk_data.py --source openfda --endpoint drugsfda --limit 500
    python seed_bulk_data.py --source openfda --endpoint event --limit 500
    python seed_bulk_data.py --source all --limit 500

RESEARCH NOTE -- "bulk JSON archive" for ClinicalTrials.gov, verified before
writing a line of this script (not assumed from the spec's wording):
    There is no single server-side URL that returns one zip of all 598,690+
    studies. Verified directly: (1) fetched clinicaltrials.gov's own
    documentation pages -- no such endpoint is documented; (2) drove the
    real "Download" feature on the site's search UI (File Format: JSON,
    "put each study in a separate file... as a zip archive", "All 598,690
    studies") and captured the actual downloaded zip -- it IS real and
    produces exactly what this spec describes (one JSON file per study,
    same protocolSection/derivedSection shape the v2 API already returns),
    but no distinct backend "bulk" URL was observable, and the UI's own
    pageSize behavior matches the same v2 API below -- strongly indicating
    the "download all" button paginates this same endpoint client-side and
    zips the result for convenience, not a separate bulk data product.
    Also verified along the way: fetch_and_embed_trials.py's own comment
    ("pageSize... API v2 hard cap" = 100) was WRONG -- pageSize=1000 is
    accepted and genuinely returns 1000 studies (pageSize=10000 is ALSO
    accepted but silently capped at 1000, confirming 1000 is the true
    ceiling). This script uses that verified, corrected pageSize -- a real
    10x throughput improvement over the earlier milestone's assumption --
    which is what actually makes "hydrate 598,690 studies without hammering
    the API one-at-a-100" tractable (599 requests instead of 5,987), even
    though it is API pagination rather than a literal one-shot archive.

RESEARCH NOTE -- openFDA, verified the same way:
    https://api.fda.gov/download.json IS real and DOES expose
    results.drug.{drugsfda,event}.partitions[].{file,size_mb,records} --
    fetched live and inspected directly. drug/drugsfda is one partition
    (29,267 records, 8.94MB zipped). drug/event is 1,767 partitions
    totaling 20.69 MILLION records -- confirming the batching requirement
    (AC 2) is not theoretical. Each partition, unzipped, is ONE JSON file
    holding ALL its records in a single top-level "results" array (verified
    by downloading and unzipping the drugsfda partition: 124.7MB
    uncompressed for 29,267 records) -- NOT pre-chunked, NOT newline-
    delimited JSON. A plain json.load() of a drug/event partition (or
    worse, of the concatenation of all 1,767) would defeat the entire
    purpose of AC 2, so this script streams the "results" array with ijson
    instead of ever materializing it as one Python list/dict.

SCOPE NOTE -- Neo4j / entity resolution:
    Only run for ClinicalTrials.gov data, by calling build_kg.py (which
    already implements it) as a subprocess against .venv-kg's interpreter --
    see build_kg.py's own module docstring for why scispacy needs an
    isolated Python 3.11 venv on this machine. openFDA's drug/drugsfda and
    drug/event records are NOT pushed through this pipeline: the
    (Trial)-[:INVESTIGATES]->(Drug)-[:MAPPED_TO_RXNORM]->(Concept) schema is
    trial-centric, and neither openFDA endpoint has a "Trial" -- forcing an
    FDA approval record or an adverse-event report through a schema built
    around clinical trials would be a genuine data-modeling error, not a
    simplification. (openFDA's OWN records already carry `rxcui` directly
    in their `openfda` block, verified in a live drugsfda record -- a
    reasonable foundation for a future openFDA-specific graph schema, not
    attempted here.)
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import ijson
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from embeddings import vector_params
from fetch_and_embed_trials import (
    AWS_REGION,
    EMBEDDING_MODEL,
    NCT_NAMESPACE,
    PHASE_LABELS,
    QDRANT_HOST,
    QDRANT_PORT,
    S3_BUCKET,
    archive_to_s3,
    build_embedding_text,
    ensure_collection,
    extract_interventions,
    index_records,
)

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_KG_PYTHON = PROJECT_ROOT / ".venv-kg" / "bin" / "python"
BUILD_KG_SCRIPT = PROJECT_ROOT / "build_kg.py"

# Verified against the live API -- see module docstring. Requesting more is
# silently capped at 1000, not rejected, so this genuinely is the ceiling,
# not a conservative guess.
CT_PAGE_SIZE = 1000
CT_API_URL = "https://clinicaltrials.gov/api/v2/studies"
CT_API_FIELDS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.designModule.phases",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
    "protocolSection.descriptionModule.briefSummary",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.designModule.studyType",
]
CT_S3_PREFIX = "raw/clinical_trials/bulk_seed"

OPENFDA_MANIFEST_URL = "https://api.fda.gov/download.json"
OPENFDA_S3_PREFIX = "raw/openfda/bulk_seed"
OPENFDA_COLLECTIONS = {
    "drugsfda": "openfda_drugsfda",
    "event": "openfda_drug_events",
}
# Namespaces distinct from NCT_NAMESPACE so ids can never collide across
# corpora even by coincidence.
OPENFDA_NAMESPACE = uuid.UUID("d3b8f0a1-6c4e-4f9a-9d2b-1e7a5c3f8b6d")

BATCH_SIZE = 1000  # spec's "batches of 1,000" -- the unit for Qdrant/S3/Neo4j flushes


# =============================================================================
# CLINICALTRIALS.GOV
# =============================================================================
def _ct_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "medical-rag-bulk-seed/1.0",
                         "Accept": "application/json"})
    return sess


def _fetch_ct_page(session: requests.Session, page_token: str | None) -> dict:
    params = {
        "pageSize": CT_PAGE_SIZE,
        "fields": ",".join(CT_API_FIELDS),
        "countTotal": "true",
    }
    if page_token:
        params["pageToken"] = page_token
    resp = session.get(CT_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _ct_record(study: dict, s3_key: str | None) -> dict | None:
    """Build the same {document, id, payload} shape
    fetch_and_embed_trials.py's build_records() produces, for exactly ONE
    study -- reused here instead of re-derived so a bulk-seeded point is
    byte-for-byte indistinguishable from one the incremental pipeline wrote."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    nct_id = ident.get("nctId")
    if not nct_id:
        return None
    summary = (proto.get("descriptionModule", {}).get("briefSummary") or "").strip()
    if not summary:
        return None

    raw_phases = proto.get("designModule", {}).get("phases", []) or []
    conditions = proto.get("conditionsModule", {}).get("conditions", []) or []
    interventions = extract_interventions(proto)
    study_type = proto.get("designModule", {}).get("studyType")

    return {
        "document": build_embedding_text(conditions, interventions, study_type, summary),
        "id": str(uuid.uuid5(NCT_NAMESPACE, nct_id)),
        "payload": {
            "NCTId": nct_id,
            "BriefTitle": ident.get("briefTitle") or "(no title)",
            "Phase": [PHASE_LABELS.get(p, p) for p in raw_phases],
            "OverallStatus": proto.get("statusModule", {}).get("overallStatus"),
            "LeadSponsorName": proto.get("sponsorCollaboratorsModule", {})
                                     .get("leadSponsor", {}).get("name"),
            "conditions": conditions,
            "interventions": interventions,
            "studyType": study_type,
            "interventionNames": [iv["name"] for iv in interventions],
            "BriefSummary": summary,
            "SourceURL": f"https://clinicaltrials.gov/study/{nct_id}",
            "SourceS3Key": s3_key,
        },
    }


def _run_build_kg(batch_path: Path, verbose: bool) -> None:
    """Entity resolution + Neo4j ingestion for one batch, via build_kg.py in
    its own isolated venv -- see this module's SCOPE NOTE docstring."""
    if not VENV_KG_PYTHON.exists():
        print(f"[neo4j]   SKIPPED -- {VENV_KG_PYTHON} not found "
              f"(see build_kg.py's docstring to set up .venv-kg)", file=sys.stderr)
        return
    cmd = [str(VENV_KG_PYTHON), str(BUILD_KG_SCRIPT), "--input", str(batch_path)]
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


def seed_clinicaltrials(client: QdrantClient, limit: int | None, skip_s3: bool,
                        skip_neo4j: bool, batch_size: int, verbose: bool) -> dict:
    print("=" * 74)
    print("medical-rag :: Historical Seed -- ClinicalTrials.gov")
    print("=" * 74)
    ensure_collection(client, recreate=False)

    session = _ct_session()
    page_token: str | None = None
    total_indexed = 0
    total_pages = 0
    stop = False

    while not stop:
        page = _fetch_ct_page(session, page_token)
        studies = page.get("studies", [])
        if not studies:
            break
        total_pages += 1
        print(f"[fetch]   page {total_pages}: {len(studies)} studies "
              f"(totalCount upstream: {page.get('totalCount')})")

        # --- LAND + ARCHIVE this page verbatim, same discipline as
        # fetch_and_embed_trials.py: nothing is parsed for Qdrant until the
        # raw page is durably archived. ------------------------------------
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        blob = json.dumps(page, indent=2, ensure_ascii=False).encode("utf-8")
        local_dir = PROJECT_ROOT / "data" / "raw" / "bulk_seed"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / f"clinical_trials_page_{stamp}.json"
        local_path.write_bytes(blob)

        s3_key = None
        if not skip_s3:
            s3_key = archive_to_s3(local_path, blob, key_prefix=CT_S3_PREFIX,
                                   content_type="application/json")

        # --- BATCH into Qdrant-ready records, respecting --limit -----------
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

            # --- Neo4j: same batch, in the SAME raw shape build_kg.py's
            # load_studies() already expects ({"pages": [{"studies": [...]}]})
            if not skip_neo4j:
                kg_payload = {"pages": [{"studies": studies[:len(records)]}]}
                kg_path = local_dir / f"clinical_trials_kg_batch_{stamp}.json"
                kg_path.write_text(json.dumps(kg_payload), encoding="utf-8")
                _run_build_kg(kg_path, verbose)

        page_token = page.get("nextPageToken")
        if not page_token or stop:
            break

    print(f"[done]    {total_indexed} trial(s) indexed across {total_pages} page(s)")
    return {"indexed": total_indexed, "pages": total_pages}


# =============================================================================
# OPENFDA
# =============================================================================
def _fetch_openfda_manifest() -> dict:
    resp = requests.get(OPENFDA_MANIFEST_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _openfda_partitions(endpoint: str) -> list[dict]:
    manifest = _fetch_openfda_manifest()
    node = manifest["results"]["drug"][endpoint]
    print(f"[fetch]   drug/{endpoint}: {node['total_records']:,} total records "
          f"across {len(node['partitions'])} partition(s) (export_date={node['export_date']})")
    return node["partitions"]


def _stream_openfda_records(zip_bytes: bytes):
    """Extract the ZIP in memory (per spec), then STREAM the "results" array
    out of the single large JSON file inside it via ijson, one record at a
    time -- never materializing the full array. See module docstring."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        inner_name = zf.namelist()[0]
        with zf.open(inner_name) as f:
            yield from ijson.items(f, "results.item")


def _drugsfda_record(r: dict) -> dict | None:
    app_no = r.get("application_number")
    if not app_no:
        return None
    products = r.get("products", []) or []
    brand_names = [p.get("brand_name") for p in products if p.get("brand_name")]
    ingredients = sorted({
        ai.get("name") for p in products for ai in (p.get("active_ingredients") or [])
        if ai.get("name")
    })
    dosage_forms = sorted({p.get("dosage_form") for p in products if p.get("dosage_form")})
    openfda = r.get("openfda", {}) or {}

    document = (
        f"Application: {app_no}\n"
        f"Sponsor: {r.get('sponsor_name') or 'Unknown'}\n"
        f"Brand names: {', '.join(brand_names) or 'Not specified'}\n"
        f"Active ingredients: {', '.join(ingredients) or 'Not specified'}\n"
        f"Dosage forms: {', '.join(dosage_forms) or 'Not specified'}"
    )
    return {
        "document": document,
        "id": str(uuid.uuid5(OPENFDA_NAMESPACE, f"drugsfda:{app_no}")),
        "payload": {
            "ApplicationNumber": app_no,
            "SponsorName": r.get("sponsor_name"),
            "BrandNames": brand_names,
            "ActiveIngredients": ingredients,
            "DosageForms": dosage_forms,
            "Rxcui": openfda.get("rxcui", []),
            "ProductType": openfda.get("product_type", []),
            "SourceURL": f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm"
                        f"?event=overview.process&ApplNo={app_no.replace('NDA', '').replace('ANDA', '').replace('BLA', '')}",
        },
    }


def _drug_event_record(r: dict) -> dict | None:
    report_id = r.get("safetyreportid")
    if not report_id:
        return None
    patient = r.get("patient", {}) or {}
    drugs = patient.get("drug", []) or []
    drug_names = sorted({d.get("medicinalproduct") for d in drugs if d.get("medicinalproduct")})
    indications = sorted({d.get("drugindication") for d in drugs if d.get("drugindication")})
    reactions = [rx.get("reactionmeddrapt") for rx in (patient.get("reaction") or [])
                if rx.get("reactionmeddrapt")]

    document = (
        f"Drugs: {', '.join(drug_names) or 'Not specified'}\n"
        f"Indications: {', '.join(indications) or 'Not specified'}\n"
        f"Reactions: {', '.join(reactions) or 'Not specified'}\n"
        f"Serious: {'Yes' if r.get('serious') == '1' else 'No'}"
    )
    return {
        "document": document,
        "id": str(uuid.uuid5(OPENFDA_NAMESPACE, f"event:{report_id}")),
        "payload": {
            "SafetyReportId": report_id,
            "ReceiveDate": r.get("receivedate"),
            "Serious": r.get("serious") == "1",
            "DrugNames": drug_names,
            "Indications": indications,
            "Reactions": reactions,
        },
    }


OPENFDA_BUILDERS = {"drugsfda": _drugsfda_record, "event": _drug_event_record}


def ensure_openfda_collection(client: QdrantClient, collection_name: str, recreate: bool) -> None:
    exists = client.collection_exists(collection_name)

    # Same dimension-mismatch guard as fetch_and_embed_trials.ensure_collection
    # -- a collection built under the old 768-dim Nomic model cannot accept
    # 1536-dim OpenAI vectors; force a recreate instead of leaving a
    # permanently-dead collection around.
    if exists and not recreate:
        current = client.get_collection(collection_name).config.params.vectors
        if getattr(current, "size", None) != vector_params().size:
            print(f"[index]   collection '{collection_name}' has the wrong vector "
                  f"size for {EMBEDDING_MODEL} -- forcing recreate")
            recreate = True

    if exists and recreate:
        client.delete_collection(collection_name)
        print(f"[index]   dropped existing collection '{collection_name}'")
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params(),
        )
        print(f"[index]   created collection '{collection_name}'")
    else:
        print(f"[index]   collection '{collection_name}' exists (upserting)")


def seed_openfda(client: QdrantClient, endpoint: str, limit: int | None,
                 skip_s3: bool, batch_size: int) -> dict:
    print("=" * 74)
    print(f"medical-rag :: Historical Seed -- openFDA drug/{endpoint}")
    print("=" * 74)
    collection_name = OPENFDA_COLLECTIONS[endpoint]
    builder = OPENFDA_BUILDERS[endpoint]
    ensure_openfda_collection(client, collection_name, recreate=False)

    partitions = _openfda_partitions(endpoint)
    total_indexed = 0
    total_seen = 0

    for part in partitions:
        if limit is not None and total_indexed >= limit:
            break
        url = part["file"]
        print(f"[fetch]   {part.get('display_name', url)}  "
              f"({part.get('size_mb')} MB, {part.get('records')} records)")

        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        zip_bytes = resp.content  # ONE partition (tens of MB) -- bounded, per spec's
                                  # "extract the ZIP in memory"; see docstring for why
                                  # this is the deliberate boundary of what's memory-safe

        if not skip_s3:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            local_dir = PROJECT_ROOT / "data" / "raw" / "bulk_seed"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / f"openfda_{endpoint}_{stamp}.json.zip"
            local_path.write_bytes(zip_bytes)
            archive_to_s3(local_path, zip_bytes,
                          key_prefix=f"{OPENFDA_S3_PREFIX}/{endpoint}",
                          content_type="application/zip")

        batch: list[dict] = []
        for record in _stream_openfda_records(zip_bytes):
            total_seen += 1
            rec = builder(record)
            if rec:
                batch.append(rec)

            if len(batch) >= batch_size or (limit is not None and total_indexed + len(batch) >= limit):
                index_records(client, batch, batch_size, collection_name=collection_name)
                total_indexed += len(batch)
                print(f"[index]   +{len(batch)} points (running total: {total_indexed})")
                batch = []
                if limit is not None and total_indexed >= limit:
                    break

        if batch:
            index_records(client, batch, batch_size, collection_name=collection_name)
            total_indexed += len(batch)
            print(f"[index]   +{len(batch)} points (running total: {total_indexed})")

    print(f"[done]    {total_indexed} record(s) indexed into '{collection_name}' "
          f"({total_seen} scanned)")
    return {"indexed": total_indexed, "collection": collection_name}


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["clinicaltrials", "openfda", "all"],
                        default="all")
    parser.add_argument("--endpoint", choices=["drugsfda", "event", "both"],
                        default="both", help="openFDA endpoint (ignored for --source clinicaltrials)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N records PER SOURCE -- for a smoke test "
                             "(AC 3: e.g. --limit 500), not full ingestion")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--skip-s3", action="store_true",
                        help="skip archiving (local dev only -- breaks replayability)")
    parser.add_argument("--skip-neo4j", action="store_true",
                        help="skip entity resolution (clinicaltrials only -- openFDA "
                             "never runs it, see module docstring)")
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
    print(f"[qdrant]  model={EMBEDDING_MODEL} dim={vector_params().size}")

    results = {}
    if args.source in ("clinicaltrials", "all"):
        results["clinicaltrials"] = seed_clinicaltrials(
            client, args.limit, args.skip_s3, args.skip_neo4j, args.batch_size, args.verbose_kg)

    if args.source in ("openfda", "all"):
        endpoints = ["drugsfda", "event"] if args.endpoint == "both" else [args.endpoint]
        for ep in endpoints:
            results[f"openfda_{ep}"] = seed_openfda(
                client, ep, args.limit, args.skip_s3, args.batch_size)

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
