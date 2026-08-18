"""Corporate news & earnings-call transcript ingestion: RSS press releases
plus free earnings call transcripts -> Qdrant `corporate_news`.

    python fetch_news_and_transcripts.py
    python fetch_news_and_transcripts.py --skip-transcripts
    python fetch_news_and_transcripts.py --rss-limit 5

VERIFIED before writing this (not assumed from the spec's wording):

  - RSS feeds: FDA press releases (fda.gov), Johnson & Johnson
    (jnj.com/media-center.rss), and AbbVie (news.abbvie.com) are ALL real,
    live, and carry genuine recent biopharma press releases -- fetched and
    inspected directly, including confirming real recent item titles (e.g.
    a genuine "...receives U.S. FDA Priority Review..." J&J release).
    JNJ's and AbbVie's feed URLs were found via each site's own
    `<link rel="alternate" type="application/rss+xml">` autodiscovery tag,
    not guessed.

    Pfizer's own /rss.xml IS a real RSS feed but is NOT press releases --
    verified live that its actual items are a generic CMS "recent content"
    stream mixing in clinical-trial-protocol summaries and portfolio-
    company blurbs (e.g. "Oblenio Bio", raw trial protocol titles), nothing
    resembling a corporate press release. Merck's /feed/ returns a real,
    well-formed RSS envelope with ZERO <item> entries. Neither ticker has a
    working, relevant press-release RSS feed discoverable via their public
    site as of this writing -- documented here rather than silently forced
    into RSS_FEEDS with broken or irrelevant content.

  - yfinance (the spec's suggested transcript source): verified directly
    that its entire public API surface (checked via dir(yf.Ticker(...)))
    has no method or attribute containing "transcript" anywhere in its
    name -- it is a financial-data library (prices, EPS estimates,
    earnings DATES only), not a transcript source. Not used here; see
    requirements.txt's own comment on this.

  - Real, free, UNPAYWALLED earnings call transcripts do exist, at Motley
    Fool (fool.com/earnings/call-transcripts/...) -- verified live: a real
    MRK Q2 2026 transcript page's own embedded page data carries an
    explicit `"datasource":"sanity-cms.freeArticle"` marker, and its
    `div.transcript-content` contains 57,000+ characters of genuine
    transcript text (real speaker names, real financial figures, real
    pipeline data like Phase III HIV regimen results) with no paywall gate
    encountered.

    There is no reliable, general "give me ticker X's latest transcript"
    discovery endpoint on Fool's public site -- checked and ruled out: no
    sitemap.xml for transcripts, and a ticker's own /quote/ page only
    INCIDENTALLY embeds a transcript link when Fool's editorial widget
    happens to be currently featuring it (confirmed true for MRK, false
    for PFE/JNJ/ABBV in the same live check). Rather than claim automatic
    multi-ticker discovery that doesn't actually work reliably, this
    script takes an explicit, configurable list of transcript URLs
    (TRANSCRIPT_URLS below). The defaults are real, live-verified URLs --
    each ticker's most recent transcript available on Fool as of this
    writing, found via web search and confirmed live with a real HTTP 200
    and real transcript-shaped content before being hardcoded here.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from embeddings import EMBEDDING_MODEL, vector_params
from fetch_and_embed_trials import QDRANT_HOST, QDRANT_PORT, index_records

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

# Real iXBRL-ish/XML-declaration pages trip bs4's own cosmetic warning when
# parsed with an HTML parser -- same non-issue already documented in
# fetch_sec_edgar.py; get_text() extraction is correct regardless.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

COLLECTION_NAME = "corporate_news"

# Only feeds VERIFIED live to be real, relevant press-release/news feeds --
# see module docstring for what was checked and rejected.
RSS_FEEDS = [
    {"name": "FDA Press Announcements",
     "url": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"},
    {"name": "Johnson & Johnson Media Center",
     "url": "https://www.jnj.com/media-center.rss"},
    {"name": "AbbVie News Center",
     "url": "https://news.abbvie.com/index.php?s=2429&pagetemplate=rss"},
]

# Explicit, verified transcript URLs -- see module docstring for why this
# is not automatic multi-ticker discovery. Each is the most recent
# transcript available on Motley Fool (a real, free, non-paywalled source,
# verified live) for that ticker as of this writing; pass --transcript-url
# to ingest an additional/newer one without editing this file.
TRANSCRIPT_URLS = [
    {"ticker": "MRK", "company": "Merck & Co., Inc.",
     "url": "https://www.fool.com/earnings/call-transcripts/2026/08/11/merck-mrk-q2-2026-earnings-call-transcript/"},
    {"ticker": "PFE", "company": "Pfizer Inc.",
     "url": "https://www.fool.com/earnings/call-transcripts/2026/08/11/pfizer-pfe-q2-2026-earnings-call-transcript/"},
    {"ticker": "JNJ", "company": "Johnson & Johnson",
     "url": "https://www.fool.com/earnings/call-transcripts/2026/01/21/johnson-johnson-jnj-earnings-call-transcript/"},
    {"ticker": "ABBV", "company": "AbbVie Inc.",
     "url": "https://www.fool.com/earnings/call-transcripts/2026/08/03/abbvie-abbv-q2-2026-earnings-call-transcript/"},
]

NEWS_NAMESPACE = uuid.UUID("b6d4e8f2-3a91-4c7d-8e5b-1f9a6c2d4e8b")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# Same keyword-windowed section-isolation technique fetch_sec_edgar.py
# already uses for SEC filings (see that module for the live verification
# this pattern is based on), retargeted to earnings-call vocabulary --
# directly implements the spec's "specifically targeting forward-looking
# statements, pipeline updates, and financial guidance" instruction rather
# than blind fixed-size chunking of an entire ~50K+ character transcript,
# most of which is routine Q&A/financial-line-item detail.
TRANSCRIPT_KEYWORDS = [
    "pipeline", "forward-looking", "guidance", "Phase 1", "Phase 2", "Phase 3",
    "FDA", "PDUFA", "deprioritiz", "discontinu", "topline", "readout",
    "regulatory", "approval", "label", "accelerat",
]
WINDOW_BEFORE = 400
WINDOW_AFTER = 1800
MERGE_GAP = 400


# =============================================================================
# SHARED: keyword-windowed section isolation
# =============================================================================
def isolate_sections(text: str, keywords: list[str]) -> str:
    """Grab WINDOW_BEFORE/WINDOW_AFTER characters around every keyword
    match, merge windows within MERGE_GAP characters of each other into one
    contiguous span, and concatenate the merged spans."""
    hits = []
    for kw in keywords:
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
# RSS PRESS RELEASES
# =============================================================================
def fetch_rss_entries(feed: dict, limit: int) -> list[dict]:
    print(f"[news]    fetching {feed['name']} ({feed['url']})")
    parsed = feedparser.parse(feed["url"])
    if parsed.bozo and not parsed.entries:
        print(f"[news]    WARNING: {feed['name']} failed to parse: {parsed.bozo_exception}",
              file=sys.stderr)
        return []

    entries = []
    for e in parsed.entries[:limit]:
        title = (e.get("title") or "").strip()
        raw_summary = e.get("summary") or e.get("description") or ""
        summary = BeautifulSoup(raw_summary, "lxml").get_text(" ", strip=True)
        link = e.get("link") or ""
        pub = e.get("published") or e.get("pubDate") or ""
        if not title or not link:
            continue
        entries.append({"FeedName": feed["name"], "Title": title,
                        "Text": summary or title, "Link": link, "PubDate": pub})
    print(f"[news]    {feed['name']}: {len(entries)} usable entries")
    return entries


def build_rss_records(entries: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    records = []
    for e in entries:
        full_text = f"{e['Title']}\n\n{e['Text']}"
        for idx, chunk in enumerate(splitter.split_text(full_text)):
            document = f"Source: {e['FeedName']}\nTitle: {e['Title']}\n\n{chunk}"
            records.append({
                "document": document,
                "id": str(uuid.uuid5(NEWS_NAMESPACE, f"{e['Link']}:{idx}")),
                "payload": {
                    "SourceType": "press_release",
                    "FeedName": e["FeedName"],
                    "Title": e["Title"],
                    "PubDate": e["PubDate"],
                    "ChunkIndex": idx,
                    "Text": chunk,
                    "SourceURL": e["Link"],
                },
            })
    return records


# =============================================================================
# EARNINGS CALL TRANSCRIPTS
# =============================================================================
_TRANSCRIPT_DATE_RE = re.compile(r"/call-transcripts/(\d{4})/(\d{2})/(\d{2})/")


def fetch_transcript(entry: dict) -> dict | None:
    print(f"[news]    fetching transcript: {entry['ticker']} ({entry['url']})")
    try:
        resp = requests.get(entry["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[news]    FAILED {entry['ticker']}: {exc}", file=sys.stderr)
        return None

    soup = BeautifulSoup(resp.content, "lxml")
    body = soup.find("div", class_="transcript-content")
    if body is None:
        print(f"[news]    FAILED {entry['ticker']}: transcript-content container "
              f"not found (page layout may have changed)", file=sys.stderr)
        return None

    text = body.get_text(separator=" ", strip=True)
    if not text:
        print(f"[news]    FAILED {entry['ticker']}: empty transcript body", file=sys.stderr)
        return None

    # The URL path itself encodes the call date (YYYY/MM/DD) -- more
    # reliable, code-parseable ground truth than regexing the "DATE ..."
    # free-text line inside the transcript's own header.
    m = _TRANSCRIPT_DATE_RE.search(entry["url"])
    call_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None

    print(f"[news]    {entry['ticker']}: {len(text):,} chars fetched")
    return {**entry, "Text": text, "CallDate": call_date}


def build_transcript_records(transcripts: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    records = []
    for t in transcripts:
        section_text = isolate_sections(t["Text"], TRANSCRIPT_KEYWORDS)
        if not section_text.strip():
            print(f"[news]    {t['ticker']}: no pipeline/guidance/forward-looking "
                  f"mentions found -- skipped")
            continue
        for idx, chunk in enumerate(splitter.split_text(section_text)):
            document = (
                f"Company: {t['company']} ({t['ticker']})\n"
                f"Earnings Call Transcript -- {t['CallDate'] or 'date unknown'}\n\n{chunk}"
            )
            records.append({
                "document": document,
                "id": str(uuid.uuid5(NEWS_NAMESPACE, f"{t['ticker']}:{t['CallDate']}:{idx}")),
                "payload": {
                    "SourceType": "earnings_transcript",
                    "Ticker": t["ticker"],
                    "Company": t["company"],
                    "CallDate": t["CallDate"],
                    "ChunkIndex": idx,
                    "Text": chunk,
                    "SourceURL": t["url"],
                },
            })
        print(f"[news]    {t['ticker']}: {len(section_text):,} section chars isolated "
              f"-> {len(splitter.split_text(section_text))} chunk(s)")
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

    for field in ("SourceType", "Ticker", "FeedName", "PubDate", "CallDate"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    print(f"[index]   payload indexes: SourceType, Ticker, FeedName, PubDate, CallDate")


# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rss-limit", type=int, default=15,
                        help="entries to keep per RSS feed (default 15)")
    parser.add_argument("--batch-size", type=int, default=16, help="Qdrant upsert batch size")
    parser.add_argument("--recreate", action="store_true",
                        help="drop and rebuild the collection first")
    parser.add_argument("--skip-rss", action="store_true")
    parser.add_argument("--skip-transcripts", action="store_true")
    parser.add_argument("--transcript-url", action="append", default=[],
                        metavar="TICKER=URL",
                        help="ingest an additional transcript, e.g. "
                             "--transcript-url 'MRK=https://www.fool.com/...'")
    args = parser.parse_args()

    started = time.time()
    print("=" * 74)
    print(f"medical-rag :: corporate news + earnings transcript ingestion  |  "
          f"{datetime.now(timezone.utc).isoformat()}")
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

    records: list[dict] = []

    if args.skip_rss:
        print("[news]    RSS ingestion SKIPPED (--skip-rss)")
    else:
        rss_entries = []
        for feed in RSS_FEEDS:
            rss_entries.extend(fetch_rss_entries(feed, args.rss_limit))
        rss_records = build_rss_records(rss_entries)
        print(f"[chunk]   {len(rss_records)} chunk(s) from {len(rss_entries)} press release(s)")
        records.extend(rss_records)

    if args.skip_transcripts:
        print("[news]    transcript ingestion SKIPPED (--skip-transcripts)")
    else:
        transcript_entries = list(TRANSCRIPT_URLS)
        for extra in args.transcript_url:
            ticker, _, url = extra.partition("=")
            if not ticker or not url:
                print(f"[news]    ignoring malformed --transcript-url {extra!r} "
                      f"(expected TICKER=URL)", file=sys.stderr)
                continue
            transcript_entries.append({"ticker": ticker.upper(), "company": ticker.upper(), "url": url})

        transcripts = [t for t in (fetch_transcript(e) for e in transcript_entries) if t]
        print(f"[news]    {len(transcripts)}/{len(transcript_entries)} transcript(s) fetched successfully")
        transcript_records = build_transcript_records(transcripts)
        print(f"[chunk]   {len(transcript_records)} chunk(s) from {len(transcripts)} transcript(s)")
        records.extend(transcript_records)

    if not records:
        print("[index]   nothing to index -- aborting", file=sys.stderr)
        return 1

    ensure_collection(client, args.recreate)
    index_records(client, records, args.batch_size, collection_name=COLLECTION_NAME)

    info = client.get_collection(COLLECTION_NAME)
    elapsed = time.time() - started
    print("-" * 74)
    print(f"points_count : {info.points_count}   status: {info.status}")
    print(f"total chunks : {len(records)}")
    print(f"done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
