"""Unified Land-Then-Archive ingestion pipeline.

    ClinicalTrials.gov v2  ->  local landing  ->  S3 archive  ->  Qdrant

Sequential, fail-closed flow:

    1. FETCH    pull raw JSON from /api/v2/studies (oncology, Phase 2/3)
    2. LAND     write the untouched payload to data/raw/
    3. ARCHIVE  upload it to S3 and VERIFY the write (size + ETag) before
                anything is parsed -- the archive is the replay source of
                truth, so if it is not durable we must not proceed
    4. INDEX    parse, embed via OpenAI (see embeddings.py), upsert into Qdrant

Because the raw payload is archived before parsing, the entire downstream
transform can be replayed from S3 (e.g. to re-embed with a better model)
without re-hitting the upstream API.

    python fetch_and_embed_trials.py
    python fetch_and_embed_trials.py --limit 10 --recreate
    python fetch_and_embed_trials.py --skip-s3      # local dev only

NOTE ON EMBEDDING (migrated off local FastEmbed/Nomic)
    Embedding now goes through embeddings.py's OpenAI client
    (text-embedding-3-small, 1536-dim) instead of the FastEmbed convenience
    API (client.set_model() / client.add()) -- local CPU embedding could not
    keep pace with hydrating a 500,000+ record corpus (~2s/doc measured on
    this machine) without severe memory pressure. index_records() below now
    embeds explicitly and upserts raw PointStruct objects; ensure_collection()
    sizes the collection from embeddings.vector_params() instead of
    client.get_fastembed_vector_params(). qdrant-client stays pinned to
    1.18.0 (see requirements.txt) but no longer needs the `[fastembed]`
    extra -- none of its convenience methods are used anymore, only the raw
    upsert()/query_points()/scroll() API, which is unaffected by either
    change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from embeddings import EMBEDDING_MODEL, embed_documents, vector_params

try:  # optional: lets .env supply S3_BUCKET / AWS_* without exporting them
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:  # pragma: no cover
    pass

# --- source ------------------------------------------------------------------
API_URL = "https://clinicaltrials.gov/api/v2/studies"
MAX_PAGE_SIZE = 100  # API v2 hard cap

API_FIELDS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.designModule.phases",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
    "protocolSection.descriptionModule.briefSummary",
    # --- payload enrichment ---------------------------------------------
    "protocolSection.conditionsModule.conditions",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.designModule.studyType",
]

# --- sinks -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

S3_BUCKET = os.getenv("S3_BUCKET", "medical-rag-raw-data-lake-jn-9043")
S3_PREFIX = "raw/clinicaltrials_gov/v2"
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Overridable via env, matching research_agent.py's own QDRANT_HOST/PORT
# pattern -- "localhost" is correct for bare-metal/host dev, but this script
# also needs to run against non-local Qdrant (e.g. one-off ECS tasks
# populating the AWS-hosted instance at qdrant.clinical-rag.local), which a
# hardcoded constant can never reach without editing this file.
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "clinical_trials"

# EMBEDDING_MODEL is imported from embeddings.py (text-embedding-3-small,
# 1536-dim, via OpenAI) -- see that module's docstring for why local
# FastEmbed/Nomic CPU embedding was replaced. The collection's vector size
# is derived from embeddings.vector_params(), so it stays in sync
# automatically -- but the collection MUST be recreated whenever the
# embedding model changes (768 -> 1536 is incompatible, like any dimension
# change).

# Stable namespace so the same NCT id always maps to the same point id,
# making re-runs an idempotent upsert instead of creating duplicates.
NCT_NAMESPACE = uuid.UUID("6f3a1d4c-9b2e-4c7a-8f11-0d5e2a7c4b93")

# The API emits enum tokens ("PHASE3"); queries filter on the human form
# ("Phase 3"). Normalising on WRITE is what makes that filter match at all.
PHASE_LABELS = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "Not Applicable",
}


# =============================================================================
# 1. FETCH
# =============================================================================
def _session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    sess = requests.Session()
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.headers.update(
        {"User-Agent": "medical-rag-etl/0.3", "Accept": "application/json"}
    )
    return sess


def fetch_raw(target_count: int, condition: str) -> dict:
    """Page the v2 API and return the raw payload envelope.

    Each upstream response is kept VERBATIM under `pages`; nothing is
    reshaped here. The envelope only adds request metadata so the pull can
    be reproduced or audited later.
    """
    sess = _session()
    pages: list[dict] = []
    collected = 0
    page_token: str | None = None

    base = {
        "query.cond": condition,
        "aggFilters": "phase:2 3",  # v2 facet syntax: PHASE2 or PHASE3
        "fields": ",".join(API_FIELDS),
        "sort": "LastUpdatePostDate:desc",  # "most recent"
        "countTotal": "true",
    }

    print(f"[fetch]   GET {API_URL}")
    print(f"[fetch]   cond={condition!r} phases=PHASE2|PHASE3 target={target_count}")

    total = None
    while collected < target_count:
        params = dict(base)
        params["pageSize"] = min(MAX_PAGE_SIZE, target_count - collected)
        if page_token:
            params["pageToken"] = page_token

        resp = sess.get(API_URL, params=params, timeout=60)
        if resp.status_code != 200:
            raise SystemExit(
                f"[fetch]   HTTP {resp.status_code} from ClinicalTrials.gov\n"
                f"{resp.text[:400]}"
            )

        page = resp.json()
        pages.append(page)
        batch = page.get("studies", [])
        collected += len(batch)
        if total is None:
            total = page.get("totalCount")
        print(f"[fetch]     page {len(pages)}: +{len(batch)} (total {collected})")

        page_token = page.get("nextPageToken")
        if not page_token or not batch:
            break

    if total is not None:
        print(f"[fetch]   matching studies upstream: {total:,}")

    return {
        "source": "clinicaltrials.gov/api/v2/studies",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": {**base, "requested_count": target_count},
        "returned_count": collected,
        "total_available_upstream": total,
        "pages": pages,  # verbatim upstream responses
    }


def studies_from(payload: dict) -> list[dict]:
    """Flatten the verbatim pages back into a single list of studies."""
    return [s for page in payload.get("pages", []) for s in page.get("studies", [])]


# =============================================================================
# 2. LAND
# =============================================================================
def land_local(payload: dict) -> tuple[Path, bytes]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RAW_DATA_DIR / f"payload_{stamp}.json"

    blob = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    path.write_bytes(blob)

    print(f"[land]    {path.relative_to(PROJECT_ROOT)} ({len(blob) / 1024:,.1f} KB)")
    return path, blob


# =============================================================================
# 3. ARCHIVE  (must succeed before any parsing happens)
# =============================================================================
def archive_to_s3(
    local_path: Path,
    blob: bytes,
    *,
    key_prefix: str = S3_PREFIX,
    content_type: str = "application/json",
) -> str:
    """Upload the raw payload and verify durability. Returns the S3 key.

    Verification is deliberately strict: we re-read the object's metadata
    from S3 and compare byte length and MD5/ETag against what we sent. A
    silent truncation here would poison every future replay.

    `key_prefix`/`content_type` default to this module's own JSON pipeline
    values so every existing call site is unaffected; ingest_pipeline.py
    reuses this same verified upload-and-verify logic for parsed PDF
    Markdown with a different prefix (`raw/pdfs`) and content type
    (`text/markdown`) instead of duplicating it.
    """
    ingest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"{key_prefix}/ingest_date={ingest_date}/{local_path.name}"

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
    except (BotoCoreError, NoCredentialsError) as exc:
        raise SystemExit(f"[archive] could not create S3 client: {exc}")

    print(f"[archive] PUT s3://{S3_BUCKET}/{key}")
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=blob,
            ContentType=content_type,
        )
    except NoCredentialsError:
        raise SystemExit(
            "[archive] no AWS credentials found.\n"
            "          Configure ~/.aws/credentials or set AWS_* in .env.\n"
            "          (--skip-s3 bypasses archiving for local dev.)"
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        raise SystemExit(
            f"[archive] upload failed ({code}): "
            f"{exc.response.get('Error', {}).get('Message', exc)}"
        )
    except BotoCoreError as exc:
        raise SystemExit(f"[archive] upload failed: {exc}")

    # --- verify the write before we trust it -----------------------------
    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=key)
    except ClientError as exc:
        raise SystemExit(f"[archive] VERIFY FAILED -- object not readable back: {exc}")

    remote_size = head["ContentLength"]
    remote_etag = head["ETag"].strip('"')
    local_md5 = hashlib.md5(blob).hexdigest()

    if remote_size != len(blob):
        raise SystemExit(
            f"[archive] VERIFY FAILED -- size mismatch: "
            f"sent {len(blob)} bytes, S3 reports {remote_size}"
        )
    # Single-part uploads use MD5 as the ETag; multipart appends "-N" and
    # would not match, so only assert when the shape is comparable.
    if "-" not in remote_etag and remote_etag != local_md5:
        raise SystemExit(
            f"[archive] VERIFY FAILED -- checksum mismatch: "
            f"local md5 {local_md5}, S3 etag {remote_etag}"
        )

    print(f"[archive] VERIFIED {remote_size:,} bytes, etag {remote_etag}")
    print(f"[archive] encryption: {head.get('ServerSideEncryption', 'none')}")
    return key


# =============================================================================
# 4. TRANSFORM + INDEX
# =============================================================================
def extract_interventions(proto: dict) -> list[dict]:
    """protocolSection.armsInterventionsModule.interventions -> [{type, name}].

    Every level uses .get() with a default: armsInterventionsModule is absent
    on observational studies, and an individual intervention can lack `type`.
    """
    raw = proto.get("armsInterventionsModule", {}).get("interventions", []) or []
    out = []
    for iv in raw:
        if not isinstance(iv, dict):
            continue
        name = (iv.get("name") or "").strip()
        if not name:
            continue  # an unnamed intervention is not filterable or citable
        out.append({"type": (iv.get("type") or "UNKNOWN").strip(), "name": name})
    return out


def build_embedding_text(
    conditions: list[str],
    interventions: list[dict],
    study_type: str | None,
    summary: str,
) -> str:
    """The enriched string handed to the embedding model.

        Conditions: <c1>, <c2>
        Interventions: <TYPE>: <name>, <TYPE>: <name>
        Study Type: <studyType>
        Summary: <briefSummary>

    Structured lines lead, then the full narrative summary. Under the previous
    all-MiniLM-L6-v2 model that ordering was load-bearing -- its tokenizer
    truncated at 128 tokens, so only the leading text was ever vectorized.
    nomic-embed-text-v1.5 has an 8192-token window and the longest enriched
    document here is ~726 tokens, so nothing is truncated now and the ordering
    is merely a readability convention.
    """
    iv_str = ", ".join(f"{iv['type']}: {iv['name']}" for iv in interventions)
    return (
        f"Conditions: {', '.join(conditions) if conditions else 'Not specified'}\n"
        f"Interventions: {iv_str if iv_str else 'Not specified'}\n"
        f"Study Type: {study_type or 'Not specified'}\n"
        f"Summary: {summary}"
    )


def build_records(studies: list[dict], s3_key: str | None) -> list[dict]:
    records: dict[str, dict] = {}
    no_summary = 0
    enrich_stats = {"conditions": 0, "interventions": 0, "studyType": 0}

    for study in studies:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        nct_id = ident.get("nctId")
        if not nct_id:
            continue

        summary = (proto.get("descriptionModule", {}).get("briefSummary") or "").strip()
        if not summary:
            # Nothing to vectorize -- an empty-content vector would still
            # compete for search slots while carrying no meaning.
            no_summary += 1
            continue

        raw_phases = proto.get("designModule", {}).get("phases", []) or []

        # --- payload enrichment (all .get()-guarded) ----------------------
        conditions = proto.get("conditionsModule", {}).get("conditions", []) or []
        interventions = extract_interventions(proto)
        study_type = proto.get("designModule", {}).get("studyType")

        if conditions:
            enrich_stats["conditions"] += 1
        if interventions:
            enrich_stats["interventions"] += 1
        if study_type:
            enrich_stats["studyType"] += 1

        records[nct_id] = {
            # the enriched string is what gets vectorized
            "document": build_embedding_text(conditions, interventions, study_type, summary),
            "id": str(uuid.uuid5(NCT_NAMESPACE, nct_id)),
            "payload": {
                "NCTId": nct_id,
                "BriefTitle": ident.get("briefTitle") or "(no title)",
                "Phase": [PHASE_LABELS.get(p, p) for p in raw_phases],
                "OverallStatus": proto.get("statusModule", {}).get("overallStatus"),
                "LeadSponsorName": proto.get("sponsorCollaboratorsModule", {})
                .get("leadSponsor", {})
                .get("name"),

                # --- enriched structured metadata -------------------------
                "conditions": conditions,          # ["Breast Carcinoma", ...]
                "interventions": interventions,    # [{"type": "DRUG", "name": "..."}]
                "studyType": study_type,
                # Flattened drug names. The raw `interventions` array above is
                # the spec'd shape, but a list of dicts cannot back a plain
                # keyword index -- this parallel list is what makes exact-match
                # filtering on a drug name a one-line MatchValue condition.
                "interventionNames": [iv["name"] for iv in interventions],

                "BriefSummary": summary,           # unenriched text, for display
                "SourceURL": f"https://clinicaltrials.gov/study/{nct_id}",
                # lineage: which archived object this point was derived from
                "SourceS3Key": s3_key,
            },
        }

    if no_summary:
        print(f"[index]   skipped {no_summary} study/studies with no BriefSummary")
    n = len(records)
    print(f"[index]   {n} records ready to embed")
    print(f"[index]   enrichment coverage: "
          f"conditions {enrich_stats['conditions']}/{n}, "
          f"interventions {enrich_stats['interventions']}/{n}, "
          f"studyType {enrich_stats['studyType']}/{n}")
    return list(records.values())


def ensure_collection(client: QdrantClient, recreate: bool) -> None:
    exists = client.collection_exists(COLLECTION_NAME)

    # A collection built under the old 768-dim Nomic model can never accept
    # 1536-dim OpenAI vectors -- upserting into it would hard-fail on the
    # first point. Force a recreate rather than leaving a permanently-dead
    # collection sitting there for someone to hit later without --recreate.
    if exists and not recreate:
        current = client.get_collection(COLLECTION_NAME).config.params.vectors
        current_size = getattr(current, "size", None)
        if current_size != vector_params().size:
            print(f"[index]   collection '{COLLECTION_NAME}' has vector size "
                  f"{current_size}, expected {vector_params().size} -- forcing recreate")
            recreate = True

    if exists and recreate:
        client.delete_collection(COLLECTION_NAME)
        print(f"[index]   dropped existing collection '{COLLECTION_NAME}'")
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vector_params(),
        )
        print(f"[index]   created collection '{COLLECTION_NAME}'")
    else:
        print(f"[index]   collection '{COLLECTION_NAME}' exists (upserting)")

    # Payload indexes make high-selectivity metadata filtering fast -- the
    # reason Qdrant was chosen. Without them, filtering is a full scan.
    # `interventionNames` and `conditions` are what enable exact-match drug /
    # indication filtering alongside vector similarity.
    indexed = ("Phase", "OverallStatus", "LeadSponsorName", "NCTId",
               "conditions", "interventionNames", "studyType")
    for field in indexed:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    print(f"[index]   payload indexes: {', '.join(indexed)}")


def index_records(
    client: QdrantClient,
    records: list[dict],
    batch: int,
    *,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Embed every record's `document` in ONE bulk call (embeddings.py's
    embed_documents() does the actual sub-batching against the OpenAI API,
    and retries a whole exhausted batch via its outer tenacity layer), then
    upsert raw PointStruct objects -- no more client.add() convenience
    wrapper, since that FastEmbed-only method no longer applies once the
    embedding step happens out-of-process against a managed API.

    `batch` now controls the Qdrant upsert chunk size, not the embedding
    batch size (embeddings.py's own _CHUNK_SIZE governs that) -- upserting
    thousands of points in a single call is its own, separate cost from
    embedding them.
    """
    print(f"[index]   embedding {len(records)} docs (model={EMBEDDING_MODEL})")
    started = datetime.now(timezone.utc)

    vectors = embed_documents([r["document"] for r in records])

    points = [
        qmodels.PointStruct(id=r["id"], vector=vec, payload=r["payload"])
        for r, vec in zip(records, vectors)
    ]
    for i in range(0, len(points), batch):
        client.upsert(collection_name=collection_name, points=points[i:i + batch])

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"[index]   upserted {len(points)} points in {elapsed:.1f}s")


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50, help="trials to fetch (default 50)")
    parser.add_argument("--condition", default="Oncology", help="condition (default Oncology)")
    parser.add_argument("--batch-size", type=int, default=16, help="embed batch size")
    parser.add_argument("--recreate", action="store_true",
                        help="drop and rebuild the collection first")
    parser.add_argument("--skip-s3", action="store_true",
                        help="skip archiving (local dev only -- breaks replayability)")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    print("=" * 74)
    print(f"medical-rag :: Land-Then-Archive ETL  |  {started.isoformat()}")
    print("=" * 74)

    # --- Qdrant client + model (fail fast before doing any real work) -----
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        client.get_collections()
    except Exception as exc:
        print(
            f"[qdrant]  cannot reach Qdrant at {QDRANT_HOST}:{QDRANT_PORT} -> {exc}\n"
            "[qdrant]  start it with:\n"
            "  docker run -d -p 6333:6333 -p 6334:6334 \\\n"
            "    -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest",
            file=sys.stderr,
        )
        return 1
    vp = vector_params()
    print(f"[qdrant]  {QDRANT_HOST}:{QDRANT_PORT} | model={EMBEDDING_MODEL} "
          f"dim={vp.size} distance={vp.distance}")

    # --- 1. FETCH ---------------------------------------------------------
    payload = fetch_raw(args.limit, args.condition)
    if not payload["returned_count"]:
        print("[fetch]   no studies returned -- aborting", file=sys.stderr)
        return 1

    # --- 2. LAND ----------------------------------------------------------
    local_path, blob = land_local(payload)

    # --- 3. ARCHIVE (gate: nothing is parsed until this succeeds) ---------
    if args.skip_s3:
        print("[archive] SKIPPED (--skip-s3) -- payload is NOT replayable")
        s3_key = None
    else:
        s3_key = archive_to_s3(local_path, blob)

    # --- 4. TRANSFORM + INDEX --------------------------------------------
    records = build_records(studies_from(payload), s3_key)
    if not records:
        print("[index]   nothing to index -- aborting", file=sys.stderr)
        return 1

    ensure_collection(client, args.recreate)
    index_records(client, records, args.batch_size)

    # --- confirm what is actually stored ----------------------------------
    info = client.get_collection(COLLECTION_NAME)
    counts = {}
    for phase in ("Phase 2", "Phase 3"):
        counts[phase] = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="Phase", match=qmodels.MatchValue(value=phase)
                )]
            ),
            exact=True,
        ).count

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print("-" * 74)
    print(f"points_count : {info.points_count}   status: {info.status}")
    print(f"phase mix    : " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"archived to  : "
          + (f"s3://{S3_BUCKET}/{s3_key}" if s3_key else "(skipped)"))
    print(f"done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
