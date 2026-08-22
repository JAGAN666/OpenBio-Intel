"""Loss-of-exclusivity (LOE) ingestion: FDA Orange Book + Purple Book -> Neo4j.

The patent-cliff intelligence commercial platforms sell (Evaluate,
DrugPatentWatch, Cortellis) is built on data FDA gives away:

- ORANGE BOOK (small molecules): monthly zip of tilde-delimited files at
  https://www.fda.gov/media/76860/download -- products.txt (48K rows),
  patent.txt (22K), exclusivity.txt (2.4K). Brand (Appl_Type=N) products
  only are ingested here: ANDAs (generics) hold no patents/exclusivity and
  would triple the node count while diluting every LOE query.

- PURPLE BOOK (biologics): there is NO full-database extract published --
  verified live; purplebooksearch.fda.gov/downloads lists only monthly
  "Historical Data Changes" CSVs (~2.2K rows each), and the site 404s
  plain curl via Akamai bot detection (a browser User-Agent is required).
  This script stacks every listed monthly CSV, dedupes by BLA number
  keeping the NEWEST occurrence, which reconstructs current state for
  every product that changed in the covered window -- honest, partial
  biologics coverage, labeled as such in the graph (source property).

Graph model (attached to the EXISTING drug graph, not a parallel one):
    (p:RegulatoryProduct {source, appl_no, product_no})
        .trade_name .ingredient .applicant .approval_date
        .patent_expiry .exclusivity_expiry .loe_estimate  (max of the two)
    (d:Drug)-[:HAS_REGULATORY_PRODUCT]->(p)   -- exact lowercase name match
        on trade name OR ingredient, the same conservative honest-miss
        policy build_kg.py uses for RxNorm.

Usage:
    uv run python fetch_exclusivity.py            # full load (idempotent MERGE)
    uv run python fetch_exclusivity.py --skip-purple
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

ORANGE_BOOK_ZIP = "https://www.fda.gov/media/76860/download"
PURPLE_BOOK_PAGE = "https://purplebooksearch.fda.gov/downloads"
# Required: purplebooksearch.fda.gov's Akamai layer serves 404 "apology"
# pages to default curl/requests UAs -- verified live.
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")

BATCH = 500

MERGE_PRODUCTS = """
UNWIND $rows AS row
MERGE (p:RegulatoryProduct {source: row.source, appl_no: row.appl_no,
                            product_no: row.product_no})
SET p.trade_name = row.trade_name,
    p.ingredient = row.ingredient,
    p.applicant = row.applicant,
    p.approval_date = row.approval_date,
    p.patent_expiry = row.patent_expiry,
    p.exclusivity_expiry = row.exclusivity_expiry,
    p.loe_estimate = row.loe_estimate
"""

LINK_DRUGS = """
UNWIND $rows AS row
MATCH (p:RegulatoryProduct {source: row.source, appl_no: row.appl_no,
                            product_no: row.product_no})
MATCH (d:Drug)
WHERE toLower(d.name) = toLower(row.trade_name)
   OR toLower(d.name) = toLower(row.ingredient)
MERGE (d)-[:HAS_REGULATORY_PRODUCT]->(p)
"""


def _parse_date(text: str | None) -> str | None:
    """'Aug 24, 2026' / '08/24/2026' / '2026-08-24' -> ISO, else None."""
    if not text or not text.strip():
        return None
    text = text.strip()
    for fmt in ("%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def load_orange_book() -> list[dict]:
    print("[ob] downloading Orange Book zip ...")
    # Browser UA needed here too -- fda.gov's Akamai layer bot-blocks the
    # default python-requests UA (verified live: 404 "apology" redirect),
    # even though the same URL serves plain curl fine.
    r = requests.get(ORANGE_BOOK_ZIP, headers={"User-Agent": BROWSER_UA}, timeout=120)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))

    def rows(name: str):
        with zf.open(name) as f:
            text = io.TextIOWrapper(f, encoding="latin-1")
            header = next(text).rstrip("\n").split("~")
            for line in text:
                yield dict(zip(header, line.rstrip("\n").split("~")))

    patents = defaultdict(list)      # (appl_no, product_no) -> [iso dates]
    for p in rows("patent.txt"):
        d = _parse_date(p.get("Patent_Expire_Date_Text"))
        if d:
            patents[(p["Appl_No"], p["Product_No"])].append(d)

    exclusivity = defaultdict(list)
    for e in rows("exclusivity.txt"):
        d = _parse_date(e.get("Exclusivity_Date"))
        if d:
            exclusivity[(e["Appl_No"], e["Product_No"])].append(d)

    out = []
    n_products = 0
    for pr in rows("products.txt"):
        if pr.get("Appl_Type") != "N":   # brand/NDA only -- see module docstring
            continue
        n_products += 1
        key = (pr["Appl_No"], pr["Product_No"])
        pat = max(patents.get(key, []), default=None)
        exc = max(exclusivity.get(key, []), default=None)
        loe = max(filter(None, [pat, exc]), default=None)
        ingredient = (pr.get("Ingredient") or "").strip()
        out.append({
            "source": "orange_book",
            "appl_no": pr["Appl_No"], "product_no": pr["Product_No"],
            "trade_name": (pr.get("Trade_Name") or "").strip(),
            "ingredient": ingredient,
            "applicant": (pr.get("Applicant_Full_Name") or pr.get("Applicant") or "").strip(),
            "approval_date": _parse_date(pr.get("Approval_Date")),
            "patent_expiry": pat, "exclusivity_expiry": exc, "loe_estimate": loe,
        })
    print(f"[ob] {n_products} brand products parsed; "
          f"{sum(1 for r in out if r['loe_estimate'])} carry a patent/exclusivity date")
    return out


def load_purple_book() -> list[dict]:
    print("[pb] discovering monthly CSVs ...")
    page = requests.get(PURPLE_BOOK_PAGE, headers={"User-Agent": BROWSER_UA},
                        timeout=60, allow_redirects=True)
    page.raise_for_status()
    links = sorted(set(re.findall(r'href="([^"]*PurpleBook[^"]*\.csv)"', page.text)))
    print(f"[pb] {len(links)} monthly files listed")

    latest: dict[str, dict] = {}   # BLA number -> row from newest file
    for url in links:              # sorted => later files overwrite earlier
        r = requests.get(url, headers={"User-Agent": BROWSER_UA}, timeout=120)
        if r.status_code != 200:
            print(f"[pb]   skip {url.rsplit('/', 1)[-1]}: HTTP {r.status_code}")
            continue
        # Some monthly files carry cp1252 punctuation (0x96 en-dash etc.) --
        # decode permissively; column names/dates are pure ASCII anyway.
        reader = csv.reader(io.StringIO(r.content.decode("utf-8-sig", errors="replace")))
        header = None
        for row in reader:
            if header is None:
                if row and row[0].strip() == "N/R/U":
                    header = row
                continue
            rec = dict(zip(header, row))
            bla = (rec.get("BLA Number") or "").strip()
            proper = (rec.get("Proper Name") or "").strip()
            if not bla or not proper:
                continue
            exc = max(filter(None, [
                _parse_date(rec.get("Exclusivity Expiration Date")),
                _parse_date(rec.get("Ref. Product Exclusivity Exp. Date")),
                _parse_date(rec.get("Orphan Exclusivity Exp. Date")),
            ]), default=None)
            latest[bla] = {
                "source": "purple_book",
                "appl_no": bla, "product_no": "001",
                "trade_name": (rec.get("Proprietary Name") or "").strip(),
                "ingredient": proper,
                "applicant": (rec.get("Applicant") or "").strip(),
                "approval_date": _parse_date(rec.get("Approval Date")),
                "patent_expiry": None,   # biologics: no Orange-Book-style patent list
                "exclusivity_expiry": exc, "loe_estimate": exc,
            }
    print(f"[pb] {len(latest)} distinct BLAs reconstructed from monthly changes")
    return list(latest.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-purple", action="store_true")
    args = ap.parse_args()

    rows = load_orange_book()
    if not args.skip_purple:
        rows += load_purple_book()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for i in range(0, len(rows), BATCH):
            session.run(MERGE_PRODUCTS, rows=rows[i:i + BATCH])
        print(f"[neo4j] merged {len(rows)} RegulatoryProduct nodes")
        linked = 0
        for i in range(0, len(rows), BATCH):
            session.run(LINK_DRUGS, rows=rows[i:i + BATCH])
        counts = session.run(
            "MATCH (:Drug)-[r:HAS_REGULATORY_PRODUCT]->() RETURN count(r) AS c"
        ).single()
        print(f"[neo4j] Drug->RegulatoryProduct links: {counts['c']}")
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
