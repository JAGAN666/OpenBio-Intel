"""Ingest FDA Complete Response Letters into the `fda_crls` Qdrant collection.

Source: openFDA's transparency/crl endpoint -- NOT the "other/approved_CRLs"
path some 2025 announcements mention (verified live: that returns 404; the
real endpoint is https://api.fda.gov/transparency/crl.json, 459 letters at
last check, WITH full letter text already extracted server-side, so no PDF
pipeline is needed at all).

Why this corpus matters: CRLs are FDA's rejection letters -- the highest-
value negative regulatory signal there is (why a drug was refused: CMC,
trial design, safety), published only since mid-2025 and barely indexed by
any aggregator yet. Text lengths run 5-20K chars per letter, chunked here
to the same ~1,500-char shape as the SEC corpus.

Usage:
    uv run python fetch_fda_crls.py            # full sync (upsert, idempotent)
    uv run python fetch_fda_crls.py --limit 20 # smoke test

Chunks carry application_number / company_name / letter_date / letter_type
payloads; IDs are uuid5(app_no + file_name + chunk_index) so re-runs are
idempotent upserts, matching the other ingestion scripts' discipline.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime

import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from embeddings import embed_documents, vector_params
from sparse_embeddings import (
    SPARSE_VECTOR_NAME,
    collection_has_bm25,
    embed_docs as embed_sparse_docs,
    sparse_text_builder_for,
    sparse_vector_params,
)

load_dotenv()

import os  # noqa: E402  (after load_dotenv so env wins)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = "fda_crls"
CRL_API = "https://api.fda.gov/transparency/crl.json"
CRL_NAMESPACE = uuid.UUID("d3b1f7a2-4c5e-4b8a-9f0d-2e6c7a8b9c0d")
# FDA's public landing page for the CRL database -- individual letter PDFs
# have no stable public URL in the API payload, so citations point here
# with the file_name as the reference.
CRL_SOURCE_PAGE = ("https://www.fda.gov/drugs/drug-approvals-and-databases/"
                   "complete-response-letters")

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)


def fetch_all(limit: int | None) -> list[dict]:
    out: list[dict] = []
    skip = 0
    while True:
        r = requests.get(CRL_API, params={"limit": 100, "skip": skip}, timeout=30)
        r.raise_for_status()
        batch = r.json().get("results", [])
        if not batch:
            break
        out.extend(batch)
        skip += len(batch)
        if limit and len(out) >= limit:
            return out[:limit]
        if len(batch) < 100:
            break
    return out


def build_records(letters: list[dict]) -> list[dict]:
    records = []
    for letter in letters:
        text = (letter.get("text") or "").strip()
        if not text:
            continue
        app_nos = letter.get("application_number") or []
        app_no = app_nos[0] if app_nos else "UNKNOWN"
        file_name = letter.get("file_name") or ""
        for i, chunk in enumerate(splitter.split_text(text)):
            records.append({
                "id": str(uuid.uuid5(CRL_NAMESPACE, f"{app_no}|{file_name}|{i}")),
                "document": chunk,
                "payload": {
                    "ApplicationNumbers": app_nos,
                    "CompanyName": letter.get("company_name"),
                    "LetterDate": letter.get("letter_date"),
                    "LetterType": letter.get("letter_type"),
                    "LetterYear": letter.get("letter_year"),
                    "ApprovalStatus": letter.get("approval_status"),
                    "FileName": file_name,
                    "ChunkIndex": i,
                    "Text": chunk,
                    "SourceURL": CRL_SOURCE_PAGE,
                },
            })
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="letters cap (smoke test)")
    args = ap.parse_args()

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    letters = fetch_all(args.limit)
    print(f"[crl] fetched {len(letters)} letters from openFDA transparency/crl")
    records = build_records(letters)
    print(f"[crl] {len(records)} chunks to embed")

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=vector_params(),
            sparse_vectors_config=sparse_vector_params(),
            on_disk_payload=True,
        )
        for field in ("ApplicationNumbers", "CompanyName", "LetterYear"):
            client.create_payload_index(COLLECTION, field_name=field,
                                        field_schema=qmodels.PayloadSchemaType.KEYWORD,
                                        wait=True)
        print(f"[crl] created collection '{COLLECTION}' (hybrid: dense + bm25)")

    vectors = embed_documents([r["document"] for r in records])

    builder = sparse_text_builder_for(COLLECTION)
    use_sparse = builder is not None and collection_has_bm25(client, COLLECTION)
    if use_sparse:
        sparse_vecs = embed_sparse_docs([builder(r["payload"]) for r in records])
        points = [qmodels.PointStruct(
            id=r["id"], vector={"": v, SPARSE_VECTOR_NAME: sv}, payload=r["payload"])
            for r, v, sv in zip(records, vectors, sparse_vecs)]
    else:
        points = [qmodels.PointStruct(id=r["id"], vector=v, payload=r["payload"])
                  for r, v in zip(records, vectors)]

    for i in range(0, len(points), 256):
        client.upsert(COLLECTION, points=points[i:i + 256])
    print(f"[crl] upserted {len(points)} points "
          f"({'hybrid' if use_sparse else 'dense-only'}); "
          f"collection now {client.count(COLLECTION).count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
