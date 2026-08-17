"""Unstructured-document ingestion: PDFs -> vision-parsed Markdown -> S3 -> Qdrant.

Same Land-Then-Archive discipline as fetch_and_embed_trials.py, applied to a
different source shape (freeform PDFs instead of clean API JSON):

    1. PARSE    vision-language Markdown extraction via parse_pdfs.py
                (LlamaParse, premium_mode -- see that module's docstring for
                why this beats OCR on tables)
    2. LAND     write the raw Markdown to data/raw/pdfs/
    3. ARCHIVE  upload to S3 and VERIFY the write, reusing
                fetch_and_embed_trials.archive_to_s3's size/ETag check
    4. INDEX    chunk with LangChain's MarkdownTextSplitter, embed via
                OpenAI (see embeddings.py), upsert into Qdrant

    python ingest_pipeline.py --pdf --dir path/to/pdfs/
    python ingest_pipeline.py --pdf --dir path/to/pdfs/ --skip-s3   # local dev only

WHY A SEPARATE QDRANT COLLECTION
    PDF chunks are indexed into `clinical_trials_pdf_extracts`, NOT the
    `clinical_trials` collection the research agent already searches. That
    collection's whole schema -- and the guardrails built on top of it
    (search_clinical_trials' payload shape, the Map-Reduce extract_trial
    worker's "nct_id MUST be copied exactly from this record's own NCTId
    field" instruction, the Smart Table contract) -- assumes every point is
    a structured ClinicalTrials.gov record with NCTId/Phase/LeadSponsorName
    fields. A freeform PDF chunk has none of those. Qdrant itself would not
    object to mixed payload shapes in one collection, but the agent's
    extraction and citation logic would: a PDF chunk surfacing in a kNN hit
    would report NCTId=None into a pipeline that promises every table row
    traces to a real trial id. Keeping the two corpora in separate
    collections is what this milestone actually needed (parsed PDF content
    landing in Qdrant, verifiably, without scrambled tables) without
    reopening research_agent.py's guardrail design. Wiring the agent to
    search this second collection (or to route by DocType) is a natural
    follow-up, out of scope here.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from embeddings import vector_params  # noqa: E402
from fetch_and_embed_trials import (  # noqa: E402
    AWS_REGION,
    EMBEDDING_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    S3_BUCKET,
    archive_to_s3,
    index_records,
)
from parse_pdfs import process_pdf, run_async  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"

PDF_S3_PREFIX = "raw/pdfs"
PDF_CONTENT_TYPE = "text/markdown"

# Deliberately NOT `clinical_trials` -- see the module docstring.
COLLECTION_NAME_PDF = "clinical_trials_pdf_extracts"

# Distinct namespace from fetch_and_embed_trials.NCT_NAMESPACE so a PDF
# chunk and a trial record can never collide on point id even by accident.
PDF_NAMESPACE = uuid.UUID("a17c9e2b-4f6d-4a3e-9c1b-2e8f5d7a6c40")

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200


# =============================================================================
# 1. PARSE
# =============================================================================
def parse_pdf_sync(path: Path) -> str:
    print(f"[parse]   {path.name}  (LlamaParse, premium vision-language mode)")
    markdown = run_async(process_pdf(str(path)))
    print(f"[parse]   {len(markdown):,} chars of Markdown extracted")
    return markdown


# =============================================================================
# 2. LAND
# =============================================================================
def land_local_markdown(pdf_path: Path, markdown: str) -> tuple[Path, bytes]:
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RAW_PDF_DIR / f"{pdf_path.stem}_{stamp}.md"

    blob = markdown.encode("utf-8")
    out_path.write_bytes(blob)

    print(f"[land]    {out_path.relative_to(PROJECT_ROOT)} ({len(blob) / 1024:,.1f} KB)")
    return out_path, blob


# =============================================================================
# 3. ARCHIVE  (reuses fetch_and_embed_trials.archive_to_s3 -- same
#    upload-then-verify logic, different prefix/content-type)
# =============================================================================
def archive_markdown_to_s3(local_path: Path, blob: bytes) -> str:
    return archive_to_s3(
        local_path, blob, key_prefix=PDF_S3_PREFIX, content_type=PDF_CONTENT_TYPE
    )


# =============================================================================
# 4. CHUNK + INDEX
# =============================================================================
def _table_blocks(markdown: str) -> list[str]:
    """Every contiguous run of markdown table-row lines (`| ... |`) in the
    source document, as exact multi-line strings. Used both to protect
    tables during chunking (chunk_markdown) and to verify none got split
    (verify_tables_intact)."""
    blocks: list[str] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def chunk_markdown(pdf_name: str, markdown: str, s3_key: str | None,
                    chunk_size: int, chunk_overlap: int) -> tuple[list[dict], list[str]]:
    """MarkdownTextSplitter walks a separator hierarchy (headings, code
    fences, thematic breaks, blank lines, then plain newlines/words) -- it is
    markdown-STRUCTURE-aware, not specifically table-boundary-aware; there is
    no dedicated "never split a table" rule in LangChain's separator list
    (verified directly against RecursiveCharacterTextSplitter's Markdown
    separators, then confirmed the hard way: the first live run of this
    pipeline against a real merged-header efficacy table DID split it across
    a chunk boundary -- the table's own markdown text, with each spanning
    header repeated per sub-column, was large enough that even isolating
    "heading + table" from the surrounding prose didn't fit under
    chunk_size, and MarkdownTextSplitter's separator hierarchy has nothing
    finer than a blank line or a bare newline to fall back to once headings
    are exhausted -- exactly a mid-table break.

    The fix is a table-atomic pre-pass, layered on top of (not instead of)
    MarkdownTextSplitter: every contiguous table block is swapped out for a
    short placeholder token BEFORE the document goes through the splitter,
    so the splitter's chunk_size accounting is driven by the surrounding
    prose, never by the table's own size. Each placeholder is then swapped
    back out for its table's full text as its own standalone chunk --
    intentionally atomic even if that one chunk exceeds chunk_size, because
    a slightly oversized but intact table is strictly better than a
    correctly-sized but scrambled one for a data table.
    """
    table_blocks = _table_blocks(markdown)
    working = markdown
    placeholder_of: list[str] = []
    for i, block in enumerate(table_blocks):
        token = f"\x00TABLE_{i}\x00"
        working = working.replace(block, token, 1)
        placeholder_of.append(token)

    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    prose_chunks = splitter.split_text(working)

    chunks: list[str] = []
    for piece in prose_chunks:
        remainder = piece
        for i, token in enumerate(placeholder_of):
            if token not in remainder:
                continue
            before, remainder = remainder.split(token, 1)
            if before.strip():
                chunks.append(before.strip())
            chunks.append(table_blocks[i])  # the table, verbatim, alone
        if remainder.strip():
            chunks.append(remainder.strip())

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "document": chunk,
            "id": str(uuid.uuid5(PDF_NAMESPACE, f"{pdf_name}:{i}")),
            "payload": {
                "SourceFile": pdf_name,
                "DocType": "pdf_extract",
                "ChunkIndex": i,
                "ChunkCount": len(chunks),
                "Text": chunk,
                "SourceS3Key": s3_key,
                "ParsedAtUtc": datetime.now(timezone.utc).isoformat(),
            },
        })
    print(f"[chunk]   {pdf_name}: {len(chunks)} chunk(s) "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")
    return records, chunks


def ensure_pdf_collection(client: QdrantClient, recreate: bool) -> None:
    exists = client.collection_exists(COLLECTION_NAME_PDF)

    # Same dimension-mismatch guard as fetch_and_embed_trials.ensure_collection
    # -- a collection built under the old 768-dim Nomic model cannot accept
    # 1536-dim OpenAI vectors; force a recreate instead of leaving a
    # permanently-dead collection around.
    if exists and not recreate:
        current = client.get_collection(COLLECTION_NAME_PDF).config.params.vectors
        if getattr(current, "size", None) != vector_params().size:
            print(f"[index]   collection '{COLLECTION_NAME_PDF}' has the wrong "
                  f"vector size for {EMBEDDING_MODEL} -- forcing recreate")
            recreate = True

    if exists and recreate:
        client.delete_collection(COLLECTION_NAME_PDF)
        print(f"[index]   dropped existing collection '{COLLECTION_NAME_PDF}'")
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME_PDF,
            vectors_config=vector_params(),
        )
        print(f"[index]   created collection '{COLLECTION_NAME_PDF}'")
    else:
        print(f"[index]   collection '{COLLECTION_NAME_PDF}' exists (upserting)")

    for field in ("SourceFile", "DocType"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME_PDF,
            field_name=field,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    print(f"[index]   payload indexes: SourceFile, DocType")


# =============================================================================
# AC4 verification: did any table get split across a chunk boundary?
# =============================================================================
def verify_tables_intact(markdown: str, chunks: list[str]) -> dict:
    """For each table block found in the ORIGINAL markdown, confirm it
    still appears verbatim, whole, inside a single chunk. A table that got
    cut by the splitter will not be a substring of any one chunk, even
    though its rows still exist somewhere in the union of all chunks."""
    tables = _table_blocks(markdown)
    intact = [t for t in tables if any(t in c for c in chunks)]
    split = [t for t in tables if t not in intact]
    return {
        "tables_found": len(tables),
        "tables_intact": len(intact),
        "tables_split": len(split),
        "split_previews": [t.splitlines()[0][:80] for t in split],
    }


# =============================================================================
def discover_pdfs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.pdf"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true",
                        help="ingest a directory of PDFs instead of the "
                             "ClinicalTrials.gov API")
    parser.add_argument("--dir", type=Path,
                        help="directory containing .pdf files (required with --pdf)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=16, help="embed batch size")
    parser.add_argument("--recreate", action="store_true",
                        help="drop and rebuild the PDF collection first")
    parser.add_argument("--skip-s3", action="store_true",
                        help="skip archiving (local dev only -- breaks replayability)")
    args = parser.parse_args()

    if not args.pdf:
        print("[ingest]  nothing to do -- pass --pdf --dir <path> "
              "(this entry point only handles PDF ingestion; use "
              "fetch_and_embed_trials.py for the ClinicalTrials.gov API)",
              file=sys.stderr)
        return 1
    if not args.dir or not args.dir.is_dir():
        print(f"[ingest]  --dir must point at an existing directory of PDFs "
              f"(got {args.dir!r})", file=sys.stderr)
        return 1

    pdfs = discover_pdfs(args.dir)
    if not pdfs:
        print(f"[ingest]  no *.pdf files found in {args.dir}", file=sys.stderr)
        return 1

    started = datetime.now(timezone.utc)
    print("=" * 74)
    print(f"medical-rag :: PDF ingestion (Land-Then-Archive)  |  {started.isoformat()}")
    print("=" * 74)
    print(f"[ingest]  {len(pdfs)} PDF(s) in {args.dir}: "
          + ", ".join(p.name for p in pdfs))

    # --- Qdrant client + model (fail fast before any parsing spend) --------
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        client.get_collections()
    except Exception as exc:
        print(f"[qdrant]  cannot reach Qdrant at {QDRANT_HOST}:{QDRANT_PORT} -> {exc}",
              file=sys.stderr)
        return 1
    print(f"[qdrant]  model={EMBEDDING_MODEL} dim={vector_params().size}")
    ensure_pdf_collection(client, args.recreate)

    total_records = 0
    total_tables = {"found": 0, "intact": 0, "split": 0}

    for pdf_path in pdfs:
        print("-" * 74)
        # --- 1. PARSE -------------------------------------------------------
        markdown = parse_pdf_sync(pdf_path)
        if not markdown.strip():
            print(f"[parse]   {pdf_path.name}: empty result -- skipping", file=sys.stderr)
            continue

        # --- 2. LAND ----------------------------------------------------------
        local_path, blob = land_local_markdown(pdf_path, markdown)

        # --- 3. ARCHIVE (gate: nothing is chunked/embedded until this succeeds)
        if args.skip_s3:
            print("[archive] SKIPPED (--skip-s3) -- payload is NOT replayable")
            s3_key = None
        else:
            s3_key = archive_markdown_to_s3(local_path, blob)

        # --- 4. CHUNK + INDEX -------------------------------------------------
        records, chunks = chunk_markdown(
            pdf_path.name, markdown, s3_key, args.chunk_size, args.chunk_overlap
        )
        if not records:
            print(f"[index]   {pdf_path.name}: no chunks produced -- skipping")
            continue

        index_records(client, records, args.batch_size,
                      collection_name=COLLECTION_NAME_PDF)
        total_records += len(records)

        report = verify_tables_intact(markdown, chunks)
        total_tables["found"] += report["tables_found"]
        total_tables["intact"] += report["tables_intact"]
        total_tables["split"] += report["tables_split"]
        print(f"[verify]  tables in source: {report['tables_found']}  "
              f"intact-in-one-chunk: {report['tables_intact']}  "
              f"split-across-chunks: {report['tables_split']}")
        for preview in report["split_previews"]:
            print(f"[verify]    ✗ split table starting: {preview!r}")

    info = client.get_collection(COLLECTION_NAME_PDF)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print("-" * 74)
    print(f"points_count       : {info.points_count}   status: {info.status}")
    print(f"chunks indexed run : {total_records}")
    print(f"table integrity    : {total_tables['intact']}/{total_tables['found']} "
          f"intact, {total_tables['split']} split across a chunk boundary")
    print(f"archived to        : "
          + ("(skipped)" if args.skip_s3 else f"s3://{S3_BUCKET}/{PDF_S3_PREFIX}/..."))
    print(f"done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
