"""PubMed Central (PMC) open-access literature ingestion.

    NCBI Entrez esearch/efetch (db=pmc)  ->  JATS XML parse  ->  chunk  ->
    embed (embeddings.py, OpenAI text-embedding-3-small)  ->  Qdrant
    `pubmed_literature`  ->  drug entity resolution into Neo4j via
    build_kg_pubmed.py (isolated .venv-kg subprocess, same pattern
    seed_bulk_data.py already uses for build_kg.py).

    python fetch_pubmed.py --limit 50
    python fetch_pubmed.py --limit 10 --query "pembrolizumab[Title/Abstract] AND open access[filter]"
    python fetch_pubmed.py --limit 50 --skip-neo4j    # Qdrant only, faster iteration

VERIFIED before writing this (not assumed from the spec's wording):

  - SSL: Bio.Entrez uses raw urllib, not requests/certifi. A fresh python.org
    (non-Homebrew) macOS Python install has no CA bundle linked by default,
    which surfaces as ssl.SSLCertVerificationError on the very first esearch
    call. One-time environment fix (already applied on this machine):
    /Applications/Python <ver>/Install Certificates.command. Not something
    this script or requirements.txt can pin its way around -- see
    requirements.txt's own comment on the biopython line.

  - Query scoping: an unscoped term like "pembrolizumab AND open
    access[filter]" full-text-matches ANY article mentioning the word
    anywhere, including a completely unrelated cardiac-surgery case report
    that surfaced as a top "recent" result in a live test. Every keyword
    below is scoped to [Title/Abstract] specifically to keep results
    genuinely on-topic, not just word-matched.

  - Batch efetch: a single Entrez.efetch(db='pmc', id='id1,id2,...',
    rettype='xml', retmode='xml') call genuinely returns multiple <article>
    elements in one response -- confirmed live -- so ids are batched
    (EFETCH_BATCH_SIZE) instead of one HTTP round-trip per article.

  - <abstract> tag detection: a naive `'<abstract>' in raw_text` substring
    check gave a false negative on a real article that DOES have an
    abstract -- JATS tags commonly carry attributes (e.g.
    <abstract abstract-type="...">), which a literal string search for
    '<abstract>' does not match. This script parses with lxml.etree and
    uses `.find()`/`.itertext()` throughout; never a substring check on raw
    XML text.

  - Rate limiting: Bio.Entrez does NOT throttle requests for you. NCBI's
    documented fair-use policy (E-utilities) is 3 req/sec without an API
    key, 10 req/sec with one -- _REQUEST_INTERVAL enforces this explicitly
    between efetch batches (esearch is a single call per run, needs none).

GRAPH INTEGRATION -- entity resolution runs as a separate process
(build_kg_pubmed.py, in .venv-kg) rather than inline here, for the same
reason seed_bulk_data.py already shells out to build_kg.py: scispacy's
dependency pins have no wheels for this project's Python 3.14 main venv.
See build_kg_pubmed.py's own module docstring for the graph schema
((Article)-[:MENTIONS]->(Drug)-[:MAPPED_TO_RXNORM]->(Concept)) and the
explicit, deliberate scope note on why disease extraction is NOT
implemented (no reliable disease-typed NER/linker wired into this
pipeline -- fabricating "Disease" nodes from untyped NER spans would mean
writing false positives into the graph with no confidence signal to filter
them, the exact failure mode MIN_LINKER_CONFIDENCE exists to prevent).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from Bio import Entrez
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lxml import etree
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from embeddings import EMBEDDING_MODEL, vector_params
from fetch_and_embed_trials import QDRANT_HOST, QDRANT_PORT, index_records

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_KG_PYTHON = PROJECT_ROOT / ".venv-kg" / "bin" / "python"
BUILD_KG_PUBMED_SCRIPT = PROJECT_ROOT / "build_kg_pubmed.py"

# NCBI's fair-access policy requires an identifying contact email on every
# E-utilities request (see requirements.txt's biopython comment). Overridable
# via env for anyone else running this script.
Entrez.email = os.getenv("NCBI_ENTREZ_EMAIL", "jaganjagannath666@gmail.com")
_NCBI_API_KEY = os.getenv("NCBI_API_KEY")
if _NCBI_API_KEY:
    Entrez.api_key = _NCBI_API_KEY

# 3 req/sec without a key, 10 req/sec with one -- padded below the ceiling,
# not right at it. See module docstring: Bio.Entrez does not enforce this
# itself.
_REQUEST_INTERVAL = 0.12 if _NCBI_API_KEY else 0.35

COLLECTION_NAME = "pubmed_literature"

# Title/Abstract-scoped, per the verified query-scoping note above. A
# reasonable general "recent biopharma/oncology literature" default --
# override with --query for a narrower pull (e.g. one specific drug).
DEFAULT_QUERY = (
    "(immune checkpoint inhibitor[Title/Abstract] OR pembrolizumab[Title/Abstract] "
    "OR nivolumab[Title/Abstract] OR monoclonal antibody[Title/Abstract]) "
    "AND cancer[Title/Abstract] AND open access[filter]"
)

EFETCH_BATCH_SIZE = 10  # ids per efetch call -- verified live to return multiple <article>s in one response

PUBMED_NAMESPACE = uuid.UUID("a17b6e3f-52d4-4a2c-8e6d-9c1f4b7a3e02")

# langchain_text_splitters.RecursiveCharacterTextSplitter, character-based
# (not token-based) -- generous relative to text-embedding-3-small's 8191
# token ceiling per input, since a 1500-char chunk is comfortably under that
# even for dense scientific prose.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150


# =============================================================================
# FETCH
# =============================================================================
def esearch_ids(query: str, target_count: int) -> list[str]:
    print(f"[search]  db=pmc term={query!r} target={target_count}")
    handle = Entrez.esearch(db="pmc", term=query, retmax=target_count, sort="pub+date")
    record = Entrez.read(handle)
    handle.close()
    ids = list(record.get("IdList", []))
    print(f"[search]  matched {record.get('Count')} upstream, using {len(ids)}")
    return ids


def efetch_articles_xml(ids: list[str]) -> list:
    """Batch efetch -- EFETCH_BATCH_SIZE comma-joined ids per call, sleeping
    _REQUEST_INTERVAL between calls to respect NCBI's fair-use rate limit."""
    articles = []
    for i in range(0, len(ids), EFETCH_BATCH_SIZE):
        batch = ids[i:i + EFETCH_BATCH_SIZE]
        handle = Entrez.efetch(db="pmc", id=",".join(batch), rettype="xml", retmode="xml")
        raw = handle.read()
        handle.close()
        root = etree.fromstring(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
        found = root.findall(".//article")
        articles.extend(found)
        print(f"[fetch]   batch {i // EFETCH_BATCH_SIZE + 1}: +{len(found)} article(s)")
        time.sleep(_REQUEST_INTERVAL)
    return articles


# =============================================================================
# PARSE (JATS XML -- lxml.etree, never raw-string substring checks; see docstring)
# =============================================================================
def _text(node) -> str:
    """All text content under an lxml element, whitespace-collapsed. JATS
    nests plain text inside many inline tags (<italic>, <xref>, <sup>...) --
    `.text` alone silently drops most of it; itertext() recovers all of it."""
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _pmcid(article) -> str | None:
    for id_node in article.findall(".//article-id"):
        if id_node.get("pub-id-type") in ("pmc", "pmcid"):
            raw = (id_node.text or "").strip()
            if raw:
                return raw if raw.upper().startswith("PMC") else f"PMC{raw}"
    return None


def parse_article(article) -> dict | None:
    pmcid = _pmcid(article)
    if not pmcid:
        return None
    abstract = _text(article.find(".//abstract"))
    body = _text(article.find(".//body"))
    if not abstract and not body:
        return None  # nothing to vectorize
    year_node = article.find(".//pub-date/year")
    return {
        "pmcid": pmcid,
        "title": _text(article.find(".//article-title")) or "(no title)",
        "abstract": abstract,
        "body": body,
        "journal": _text(article.find(".//journal-title")) or "Unknown journal",
        "pub_year": (year_node.text or "").strip() if year_node is not None else None,
    }


# =============================================================================
# CHUNK + EMBED-READY RECORDS
# =============================================================================
def build_chunks(articles: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    records = []
    for art in articles:
        full_text = "\n\n".join(filter(None, [art["abstract"], art["body"]]))
        for idx, chunk in enumerate(splitter.split_text(full_text)):
            document = (
                f"Title: {art['title']}\n"
                f"Journal: {art['journal']} ({art['pub_year'] or 'n.d.'})\n\n"
                f"{chunk}"
            )
            records.append({
                "document": document,
                "id": str(uuid.uuid5(PUBMED_NAMESPACE, f"{art['pmcid']}:{idx}")),
                "payload": {
                    "PMCID": art["pmcid"],
                    "Title": art["title"],
                    "Journal": art["journal"],
                    "PubYear": art["pub_year"],
                    "ChunkIndex": idx,
                    "Text": chunk,
                    "SourceURL": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{art['pmcid']}/",
                },
            })
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
        # Hybrid schema on creation -- same rationale as
        # fetch_and_embed_trials.ensure_collection: sparse vectors can't be
        # added to an existing collection later.
        from sparse_embeddings import sparse_vector_params
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vector_params(),
            sparse_vectors_config=sparse_vector_params(),
        )
        print(f"[index]   created collection '{COLLECTION_NAME}' (hybrid: dense + bm25)")
    else:
        print(f"[index]   collection '{COLLECTION_NAME}' exists (upserting)")

    for field in ("PMCID", "Journal", "PubYear"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    print(f"[index]   payload indexes: PMCID, Journal, PubYear")


# =============================================================================
# NEO4J (subprocess into .venv-kg -- see build_kg_pubmed.py)
# =============================================================================
def _run_build_kg_pubmed(batch_path: Path, verbose: bool) -> None:
    if not VENV_KG_PYTHON.exists():
        print(f"[neo4j]   SKIPPED -- {VENV_KG_PYTHON} not found "
              f"(see build_kg.py's docstring to set up .venv-kg)", file=sys.stderr)
        return
    cmd = [str(VENV_KG_PYTHON), str(BUILD_KG_PUBMED_SCRIPT), "--input", str(batch_path)]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=not verbose, text=True)
    if result.returncode != 0:
        print(f"[neo4j]   build_kg_pubmed.py failed (exit {result.returncode})", file=sys.stderr)
        if not verbose and result.stdout:
            print(result.stdout[-2000:], file=sys.stderr)
        if not verbose and result.stderr:
            print(result.stderr[-2000:], file=sys.stderr)
    else:
        print(f"[neo4j]   build_kg_pubmed.py OK for {batch_path.name}")


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=50,
                        help="articles to fetch (default 50, matches AC1's test batch)")
    parser.add_argument("--query", default=DEFAULT_QUERY,
                        help="PMC esearch term, Title/Abstract-scoped")
    parser.add_argument("--batch-size", type=int, default=16, help="Qdrant upsert batch size")
    parser.add_argument("--recreate", action="store_true",
                        help="drop and rebuild the collection first")
    parser.add_argument("--skip-neo4j", action="store_true",
                        help="skip build_kg_pubmed.py entity resolution")
    parser.add_argument("--verbose-kg", action="store_true",
                        help="stream build_kg_pubmed.py's own output instead of capturing it")
    args = parser.parse_args()

    started = time.time()
    print("=" * 74)
    print(f"medical-rag :: PubMed Central ingestion  |  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 74)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        client.get_collections()
    except Exception as exc:
        print(f"[qdrant]  cannot reach Qdrant at {QDRANT_HOST}:{QDRANT_PORT} -> {exc}",
              file=sys.stderr)
        return 1
    print(f"[qdrant]  {QDRANT_HOST}:{QDRANT_PORT} | model={EMBEDDING_MODEL} "
          f"dim={vector_params().size}")

    ids = esearch_ids(args.query, args.limit)
    if not ids:
        print("[search]  no articles matched -- aborting", file=sys.stderr)
        return 1

    xml_articles = efetch_articles_xml(ids)
    articles = [a for a in (parse_article(x) for x in xml_articles) if a]
    skipped = len(xml_articles) - len(articles)
    print(f"[parse]   {len(articles)} article(s) with usable text"
          + (f" ({skipped} skipped -- no pmcid or empty abstract+body)" if skipped else ""))
    if not articles:
        print("[parse]   nothing to index -- aborting", file=sys.stderr)
        return 1

    records = build_chunks(articles)
    print(f"[chunk]   {len(records)} chunk(s) from {len(articles)} article(s) "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    ensure_collection(client, args.recreate)
    index_records(client, records, args.batch_size, collection_name=COLLECTION_NAME)

    if args.skip_neo4j:
        print("[neo4j]   SKIPPED (--skip-neo4j)")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        batch_dir = PROJECT_ROOT / "data" / "raw" / "pubmed"
        batch_dir.mkdir(parents=True, exist_ok=True)
        kg_path = batch_dir / f"pubmed_kg_batch_{stamp}.json"
        kg_path.write_text(json.dumps({"articles": articles}), encoding="utf-8")
        _run_build_kg_pubmed(kg_path, args.verbose_kg)

    info = client.get_collection(COLLECTION_NAME)
    elapsed = time.time() - started
    print("-" * 74)
    print(f"points_count : {info.points_count}   status: {info.status}")
    print(f"articles     : {len(articles)}")
    print(f"done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
