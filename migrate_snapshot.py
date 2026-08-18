"""Migrate the local `clinical_trials` Qdrant collection to the AWS-deployed
Qdrant instance via a snapshot -- not re-embedding. The local collection
already has the full 597,706-point corpus with the correct 1536-dim OpenAI
embeddings; the AWS collection was created much earlier by a different
ingestion path (FastEmbed's convenience API), which stores vectors under a
NAMED vector field instead of the unnamed one every current query uses --
confirmed live via a real "Wrong input: Not existing vector name error"
from Qdrant itself when the deployed backend tried to query it. A snapshot
restore is a pure binary data copy: no OpenAI embedding calls, no 10+ hour
re-fetch from ClinicalTrials.gov, and it recreates the collection with the
snapshot's own (correct) schema.

TWO PHASES, run in two different places -- the AWS Qdrant instance sits on
a private VPC network (qdrant.clinical-rag.local) with no route from
outside AWS, so this cannot run start-to-finish from a laptop:

  --stage --collection NAME
             Run LOCALLY. Snapshots the local collection, uploads the
             snapshot file to S3 (medical-rag-raw-data-lake-jn-9043,
             already used for other raw-data storage in this project), and
             prints a presigned GET URL.

  --recover URL --collection NAME
             Run INSIDE the VPC -- e.g. as a one-off `aws ecs run-task`
             using the already-deployed backend image/network config, which
             already has proven reachability to both Qdrant and the public
             internet (S3). Deletes the existing AWS collection of the same
             name if one exists and recovers it from the snapshot at URL,
             with priority="snapshot" so the snapshot's data and schema
             fully replace whatever was there. Verifies the resulting point
             count.

--collection accepts any collection name -- generalized after first proving
this against clinical_trials specifically (the one with the confirmed
schema mismatch); the same approach applies to every other collection this
project uses (sec_filings, pubmed_literature, corporate_news,
clinical_trials_pdf_extracts, openfda_drugsfda), none of which were ever
seeded onto AWS at all.

Usage:
    .venv/bin/python migrate_snapshot.py --stage --collection clinical_trials
    # then, from inside the VPC:
    python migrate_snapshot.py --recover "<presigned-url>" --collection clinical_trials
"""

from __future__ import annotations

import argparse
import os
import sys

import boto3
import requests
from qdrant_client import QdrantClient

S3_BUCKET = "medical-rag-raw-data-lake-jn-9043"
S3_PREFIX = "qdrant-migration"
PRESIGNED_URL_TTL_SECONDS = 3600


def stage(COLLECTION: str) -> str:
    # timeout=600, not the client's default (a handful of seconds) --
    # verified live: creating a snapshot of the full 597,706-point
    # collection takes long enough server-side that the default timeout
    # fires before Qdrant responds, even though the snapshot itself
    # eventually succeeds. wait=True already blocks for the real duration;
    # this just keeps the HTTP client from giving up first.
    local = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"),
                          port=int(os.getenv("QDRANT_PORT", "6333")),
                          timeout=600)

    before = local.get_collection(COLLECTION)
    print(f"[stage] local collection {COLLECTION!r}: "
          f"{before.points_count} points, status={before.status}")

    print(f"[stage] creating snapshot of {COLLECTION!r} (this blocks "
          f"until the snapshot is fully written)...")
    snap = local.create_snapshot(collection_name=COLLECTION, wait=True)
    if snap is None:
        sys.exit("[stage] create_snapshot returned None -- snapshot failed")
    print(f"[stage] snapshot created: {snap.name} ({snap.size} bytes)")

    download_url = (f"http://{os.getenv('QDRANT_HOST', 'localhost')}:"
                     f"{os.getenv('QDRANT_PORT', '6333')}/collections/"
                     f"{COLLECTION}/snapshots/{snap.name}")
    local_path = f"/tmp/{snap.name}"
    print(f"[stage] downloading snapshot from {download_url} ...")
    with requests.get(download_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
    downloaded_size = os.path.getsize(local_path)
    print(f"[stage] downloaded {downloaded_size} bytes to {local_path}")
    if downloaded_size != snap.size:
        sys.exit(f"[stage] size mismatch: snapshot={snap.size} "
                  f"downloaded={downloaded_size} -- aborting, do not upload")

    s3_key = f"{S3_PREFIX}/{snap.name}"
    print(f"[stage] uploading to s3://{S3_BUCKET}/{s3_key} ...")
    s3 = boto3.client("s3")
    s3.upload_file(local_path, S3_BUCKET, s3_key)

    presigned = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=PRESIGNED_URL_TTL_SECONDS,
    )
    print(f"\n[stage] DONE. Presigned URL (valid {PRESIGNED_URL_TTL_SECONDS}s):")
    print(presigned)
    print(f"\n[stage] Next: run this INSIDE the VPC (e.g. via a one-off ECS "
          f"task using the backend image/network config):")
    print(f'  python migrate_snapshot.py --recover "{presigned}" --collection {COLLECTION}')
    return presigned


def recover(location: str, COLLECTION: str) -> None:
    aws = QdrantClient(host=os.environ["QDRANT_HOST"],
                        port=int(os.getenv("QDRANT_PORT", "6333")),
                        timeout=600)

    if aws.collection_exists(COLLECTION):
        before = aws.get_collection(COLLECTION)
        print(f"[recover] existing AWS collection {COLLECTION!r}: "
              f"{before.points_count} points (mismatched schema -- "
              f"deleting before restore)")
        aws.delete_collection(COLLECTION)

    print(f"[recover] recovering {COLLECTION!r} from snapshot "
          f"(priority=snapshot) -- this blocks until done...")
    ok = aws.recover_snapshot(
        collection_name=COLLECTION,
        location=location,
        priority="snapshot",
        wait=True,
    )
    if not ok:
        sys.exit("[recover] recover_snapshot returned falsy -- recovery failed")

    after = aws.get_collection(COLLECTION)
    print(f"[recover] DONE. AWS collection {COLLECTION!r} now: "
          f"{after.points_count} points, status={after.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", action="store_true",
                        help="Phase 1: snapshot local collection, upload to S3.")
    group.add_argument("--recover", metavar="URL",
                        help="Phase 2: recover AWS collection from the "
                             "presigned URL --stage printed. Run inside the VPC.")
    parser.add_argument("--collection", required=True,
                         help="Qdrant collection name to migrate.")
    args = parser.parse_args()

    if args.stage:
        stage(args.collection)
    else:
        recover(args.recover, args.collection)
