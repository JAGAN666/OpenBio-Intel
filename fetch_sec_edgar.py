"""SEC EDGAR corporate filings ingestion (10-K / 8-K) for pipeline/R&D
disclosure.

    sec-edgar-downloader (download_details=True) -> primary-document.html
    -> BeautifulSoup HTML strip -> keyword-windowed section isolation
    (Pipeline / Clinical Trials / Research and Development) -> merge
    adjacent windows -> RecursiveCharacterTextSplitter -> embed
    (embeddings.py) -> Qdrant `sec_filings`.

    python fetch_sec_edgar.py                                  # PFE/MRK/JNJ/ABBV, 10-K+8-K, 1 each (AC2 default)
    python fetch_sec_edgar.py --tickers MRK --forms 10-K --limit 1
    python fetch_sec_edgar.py --tickers PFE,MRK,JNJ,ABBV --forms 10-K,8-K --limit 2

VERIFIED before writing this (not assumed from the spec's wording):

  - Downloader signature: Downloader(company_name, email_address,
    download_folder=None). SEC's fair-access policy requires an identifying
    company name + contact email on every request -- same fair-use spirit
    as Bio.Entrez.email in fetch_pubmed.py.
    .get(form, ticker_or_cik, *, limit=None, ..., download_details=False,
    ...) -> int (count of filings downloaded).

  - download_details=True (used here) additionally saves a pre-isolated
    primary-document.html per filing, alongside the default
    full-submission.txt (an SGML wrapper of EVERY exhibit -- verified live:
    29.5MB vs. 5.2MB for the same real MRK 10-K). Only primary-document.html
    is parsed for body content; full-submission.txt is read for its cheap
    ~4KB SGML header only (FILED AS OF DATE / COMPANY CONFORMED NAME), never
    its multi-exhibit body.

  - Directory layout, confirmed live: <download_folder>/sec-edgar-filings/
    <TICKER>/<FORM>/<ACCESSION_NUMBER>/{full-submission.txt,primary-document.html}

  - Accession numbers embed the filer's CIK as their first 10 digits
    (0000310158-26-000063 for a real MRK 10-K == MRK's actual CIK,
    confirmed against the SGML header's own CENTRAL INDEX KEY field) --
    used here to build a correct, resolvable EDGAR filing-index URL per
    record without a separate CIK lookup call.

  - Section isolation: verified live against a real MRK 10-K that
    "Pipeline"/"Research and Development"/"Clinical Trials" mentions
    cluster tightly together (Item 1 Business's R&D discussion), not
    scattered randomly across the ~795K-character extracted document -- so
    a keyword-windowed extraction (grab characters around each match, merge
    overlapping/adjacent windows) reliably isolates that section without
    depending on the filing's own heading markup, which iXBRL-tagged HTML
    10-Ks do not expose consistently enough to parse structurally.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sec_edgar_downloader import Downloader

from embeddings import EMBEDDING_MODEL, vector_params
from fetch_and_embed_trials import QDRANT_HOST, QDRANT_PORT, index_records

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

# Real iXBRL 10-Ks open with an XML declaration even though the body is
# genuine HTML -- bs4's own warning about this is cosmetic here (verified
# live: get_text() extraction is correct and complete regardless).
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "sec_edgar"

# SEC's fair-access policy requires an identifying company name + contact
# email on every request -- same spirit as Bio.Entrez.email in
# fetch_pubmed.py.
SEC_EDGAR_COMPANY = os.getenv("SEC_EDGAR_COMPANY", "medical-rag-research")
SEC_EDGAR_EMAIL = os.getenv("SEC_EDGAR_EMAIL", "jaganjagannath666@gmail.com")

COLLECTION_NAME = "sec_filings"

DEFAULT_TICKERS = ["PFE", "MRK", "JNJ", "ABBV"]  # per spec: "top biopharma tickers"
DEFAULT_FORMS = ["10-K", "8-K"]

SECTION_KEYWORDS = ["Pipeline", "Clinical Trials", "Research and Development"]
WINDOW_BEFORE = 500
WINDOW_AFTER = 2500
MERGE_GAP = 500  # windows within this many chars of each other merge into one span

SEC_NAMESPACE = uuid.UUID("f4c1a9d2-7e3b-4a5f-9c8d-2b6e1a4f7d93")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150


# =============================================================================
# FETCH
# =============================================================================
def download_filings(tickers: list[str], forms: list[str], limit: int) -> list[dict]:
    """Downloads via sec-edgar-downloader, returns one entry per filing that
    actually produced a primary-document.html: [{ticker, form, accession, path}]."""
    dl = Downloader(SEC_EDGAR_COMPANY, SEC_EDGAR_EMAIL, download_folder=str(DOWNLOAD_DIR))
    out = []
    for ticker in tickers:
        for form in forms:
            print(f"[fetch]   {ticker} {form} (limit={limit})")
            try:
                n = dl.get(form, ticker, limit=limit, download_details=True)
            except Exception as exc:
                print(f"[fetch]   {ticker} {form} FAILED: {exc}", file=sys.stderr)
                continue
            print(f"[fetch]   {ticker} {form}: {n} filing(s) downloaded")

            form_dir = DOWNLOAD_DIR / "sec-edgar-filings" / ticker / form
            if not form_dir.exists():
                continue
            for accession_dir in sorted(form_dir.iterdir()):
                primary = accession_dir / "primary-document.html"
                if primary.exists():
                    out.append({"ticker": ticker, "form": form,
                               "accession": accession_dir.name, "path": primary})
    return out


_HEADER_DATE_RE = re.compile(r"FILED AS OF DATE:\s*(\d{8})")
_HEADER_NAME_RE = re.compile(r"COMPANY CONFORMED NAME:\s*(.+)")


def filing_header(accession_dir: Path) -> dict:
    """Cheap read of full-submission.txt's SGML header ONLY -- first 4KB,
    never the whole multi-exhibit file -- for FILED AS OF DATE / company name."""
    full_sub = accession_dir / "full-submission.txt"
    if not full_sub.exists():
        return {}
    with full_sub.open("r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4096)
    date_m = _HEADER_DATE_RE.search(head)
    name_m = _HEADER_NAME_RE.search(head)
    filed = date_m.group(1) if date_m else None
    return {
        "filed_date": f"{filed[:4]}-{filed[4:6]}-{filed[6:]}" if filed else None,
        "company_name": name_m.group(1).strip() if name_m else None,
    }


def filing_source_url(accession: str) -> str:
    """The accession number's first 10 digits ARE the filer's CIK (verified
    live against the SGML header's own CENTRAL INDEX KEY) -- enough to build
    a real, resolvable EDGAR filing-index URL with no extra lookup call."""
    cik = str(int(accession.split("-")[0]))
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"


# =============================================================================
# PARSE + SECTION ISOLATION
# =============================================================================
def extract_text(html_path: Path) -> str:
    html = html_path.read_bytes()
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text


def isolate_sections(text: str) -> str:
    """Keyword-windowed extraction, see module docstring for why this works
    on real 10-Ks: grab WINDOW_BEFORE/WINDOW_AFTER characters around every
    Pipeline/Clinical Trials/Research and Development mention, merge windows
    within MERGE_GAP characters of each other into one contiguous span, and
    concatenate the merged spans."""
    hits = []
    for kw in SECTION_KEYWORDS:
        hits.extend(m.start() for m in re.finditer(re.escape(kw), text, re.IGNORECASE))
    if not hits:
        return ""
    hits.sort()

    spans: list[list[int]] = []
    for h in hits:
        start, end = max(0, h - WINDOW_BEFORE), min(len(text), h + WINDOW_AFTER)
        if spans and start <= spans[-1][1] + MERGE_GAP:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])

    return "\n\n".join(text[s:e] for s, e in spans)


# =============================================================================
# CHUNK + EMBED-READY RECORDS
# =============================================================================
def build_chunks(filings: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    records = []
    for f in filings:
        header = filing_header(f["path"].parent)
        section_text = isolate_sections(extract_text(f["path"]))
        if not section_text.strip():
            print(f"[parse]   {f['ticker']} {f['form']} {f['accession']}: no "
                  f"Pipeline/Clinical Trials/R&D mentions found -- skipped")
            continue

        company = header.get("company_name") or f["ticker"]
        filed_date = header.get("filed_date")
        source_url = filing_source_url(f["accession"])

        for idx, chunk in enumerate(splitter.split_text(section_text)):
            document = (
                f"Company: {company} ({f['ticker']})\n"
                f"Filing: {f['form']} filed {filed_date or 'unknown date'}\n\n"
                f"{chunk}"
            )
            records.append({
                "document": document,
                "id": str(uuid.uuid5(SEC_NAMESPACE, f"{f['accession']}:{idx}")),
                "payload": {
                    "Ticker": f["ticker"],
                    "Company": company,
                    "Form": f["form"],
                    "AccessionNumber": f["accession"],
                    "FiledDate": filed_date,
                    "ChunkIndex": idx,
                    "Text": chunk,
                    "SourceURL": source_url,
                },
            })
        print(f"[parse]   {f['ticker']} {f['form']} {f['accession']}: "
              f"{len(section_text):,} section chars isolated")
    return records


# =============================================================================
# QDRANT
# =============================================================================
def ensure_collection(client: QdrantClient, recreate: bool) -> None:
    exists = client.collection_exists(COLLECTION_NAME)
    if exists and not recreate:
        current = client.get_collection(COLLECTION_NAME).config.params.vectors
        if getattr(current, "size", None) != vector_params().size:
            print(f"[index]   collection '{COLLECTION_NAME}' has the wrong vector "
                  f"size for {EMBEDDING_MODEL} -- forcing recreate")
            recreate = True

    if exists and recreate:
        client.delete_collection(COLLECTION_NAME)
        print(f"[index]   dropped existing collection '{COLLECTION_NAME}'")
        exists = False

    if not exists:
        client.create_collection(collection_name=COLLECTION_NAME, vectors_config=vector_params())
        print(f"[index]   created collection '{COLLECTION_NAME}'")
    else:
        print(f"[index]   collection '{COLLECTION_NAME}' exists (upserting)")

    for field in ("Ticker", "Form", "FiledDate", "AccessionNumber"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    print(f"[index]   payload indexes: Ticker, Form, FiledDate, AccessionNumber")


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS),
                        help="comma-separated tickers (default: PFE,MRK,JNJ,ABBV)")
    parser.add_argument("--forms", default=",".join(DEFAULT_FORMS),
                        help="comma-separated forms (default: 10-K,8-K)")
    parser.add_argument("--limit", type=int, default=1,
                        help="most-recent filings per ticker per form (default 1 -- AC2's 'latest 10-K')")
    parser.add_argument("--batch-size", type=int, default=16, help="Qdrant upsert batch size")
    parser.add_argument("--recreate", action="store_true",
                        help="drop and rebuild the collection first")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    forms = [f.strip().upper() for f in args.forms.split(",") if f.strip()]

    started = time.time()
    print("=" * 74)
    print(f"medical-rag :: SEC EDGAR ingestion  |  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 74)
    print(f"[config]  tickers={tickers} forms={forms} limit={args.limit}")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        client.get_collections()
    except Exception as exc:
        print(f"[qdrant]  cannot reach Qdrant at {QDRANT_HOST}:{QDRANT_PORT} -> {exc}",
              file=sys.stderr)
        return 1
    print(f"[qdrant]  {QDRANT_HOST}:{QDRANT_PORT} | model={EMBEDDING_MODEL} "
          f"dim={vector_params().size}")

    filings = download_filings(tickers, forms, args.limit)
    if not filings:
        print("[fetch]   no filings downloaded -- aborting", file=sys.stderr)
        return 1

    records = build_chunks(filings)
    if not records:
        print("[parse]   nothing to index -- aborting", file=sys.stderr)
        return 1
    print(f"[chunk]   {len(records)} chunk(s) from {len(filings)} filing(s) "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    ensure_collection(client, args.recreate)
    index_records(client, records, args.batch_size, collection_name=COLLECTION_NAME)

    info = client.get_collection(COLLECTION_NAME)
    elapsed = time.time() - started
    print("-" * 74)
    print(f"points_count : {info.points_count}   status: {info.status}")
    print(f"filings      : {len(filings)}")
    print(f"done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
