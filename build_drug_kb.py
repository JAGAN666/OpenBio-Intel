"""Build the drug mechanism knowledge base in Neo4j from free sources.

Sources (all downloaded on demand into data/drug_kb/, re-used on re-run):

  1. Open Targets Platform (CC0) -- drug_molecule (22K molecules with
     ChEMBL ids, synonyms, trade names, clinical stage), drug_mechanism_of_
     action (6.5K curated MoA rows: action type, MoA phrase, target genes
     with provenance), target (Ensembl id -> HGNC symbol).
  2. IUPHAR/BPS Guide to PHARMACOLOGY (CC BY-SA 4.0) -- ligands.csv +
     interactions.csv: precise target + action for ~14K ligands, including
     late-stage development codes Open Targets lacks a mechanism for
     (verified: BMS-986278 -> LPA1 receptor antagonist).

Why a KB tier at all: the registry record names an agent but rarely
states how it works, and the extractor is forbidden from guessing. A
curated lookup keyed by every known synonym makes the mechanism column
deterministic for every approved drug and most clinical-stage assets;
the LLM tier only has to cover what no database knows yet.

Graph written (see drug_kb.py for the read side):
    (:Substance {key, pref_name, chembl_id, gtp_id, inchikey, stage, drug_type})
    (:Mechanism {id, moa, action_type, target_name, target_symbols, source, ref_url})
    (Substance)-[:HAS_MECHANISM]->(Mechanism)
    (:DrugName {norm})-[:RESOLVES_TO {priority}]->(Substance)

Idempotent (MERGE everywhere). Usage:
    uv run python build_drug_kb.py            # download + build
    uv run python build_drug_kb.py --verify   # probe a few names
AWS: one-off ECS run-task on the backend image with this command.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from drug_kb import lookup_mechanisms, mechanism_phrase, norm_name

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

DATA_DIR = Path(os.getenv("DRUG_KB_DIR", Path(__file__).resolve().parent / "data" / "drug_kb"))
OT_BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/latest/output/"
OT_DATASETS = ("drug_molecule", "drug_mechanism_of_action", "target")
GTP_FILES = {
    "ligands.csv": "https://www.guidetopharmacology.org/DATA/ligands.csv",
    "interactions.csv": "https://www.guidetopharmacology.org/DATA/interactions.csv",
}
BATCH = 2000

csv.field_size_limit(10_000_000)


# --- download ---------------------------------------------------------------
def _get(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "openbio-intel-kb/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[download] {url}: {exc} (attempt {attempt + 1})")
            time.sleep(2 ** attempt)
    raise SystemExit(f"[download] failed: {url}")


def download_all() -> None:
    for d in OT_DATASETS:
        html = urllib.request.urlopen(OT_BASE + d + "/", timeout=60).read().decode()
        parts = sorted(set(re.findall(r'part-[^"]+\.parquet', html)))
        for f in parts:
            _get(OT_BASE + d + "/" + f, DATA_DIR / "opentargets" / d / f)
        print(f"[download] opentargets/{d}: {len(parts)} file(s)")
    for name, url in GTP_FILES.items():
        _get(url, DATA_DIR / "gtopdb" / name)
    print("[download] gtopdb: ligands.csv, interactions.csv")


# --- parse ------------------------------------------------------------------
def _parquet_rows(d: str, columns: list[str] | None = None) -> list[dict]:
    import pyarrow.parquet as pq
    rows: list[dict] = []
    for f in sorted((DATA_DIR / "opentargets" / d).glob("*.parquet")):
        rows += pq.read_table(f, columns=columns).to_pylist()
    return rows


def load_opentargets():
    targets = {r["id"]: r for r in _parquet_rows(
        "target", ["id", "approvedSymbol", "approvedName"])}
    mols = _parquet_rows("drug_molecule", [
        "id", "name", "parentId", "synonyms", "tradeNames", "crossReferences",
        "maximumClinicalStage", "drugType", "inchiKey"])
    moas = _parquet_rows("drug_mechanism_of_action")
    print(f"[ot] {len(mols)} molecules, {len(moas)} mechanism rows, {len(targets)} targets")

    substances: dict[str, dict] = {}
    names: list[tuple[str, str, int]] = []  # (norm, key, priority)
    for m in mols:
        key = f"chembl:{m['id']}"
        substances[key] = {
            "key": key, "pref_name": (m["name"] or "").title() if m["name"] and m["name"].isupper() else m["name"],
            "chembl_id": m["id"], "gtp_id": None, "inchikey": m.get("inchiKey"),
            "stage": m.get("maximumClinicalStage"), "drug_type": m.get("drugType"),
            "parent_chembl": m.get("parentId"),
        }
        if m["name"]:
            names.append((norm_name(m["name"]), key, 0))
        for s in m.get("synonyms") or []:
            if s and s.get("label"):
                pri = 2 if (s.get("source") or "") == "AACT" else 1
                names.append((norm_name(s["label"]), key, pri))
        for t in m.get("tradeNames") or []:
            if t and t.get("label"):
                names.append((norm_name(t["label"]), key, 1))

    mechanisms: list[dict] = []
    links: list[tuple[str, str]] = []
    for i, r in enumerate(moas):
        syms = [targets.get(t, {}).get("approvedSymbol") for t in (r.get("targets") or [])]
        syms = [s for s in syms if s]
        ref_url = None
        for ref in r.get("references") or []:
            if ref.get("urls"):
                ref_url = ref["urls"][0]
                break
        mid = f"ot:{i}"
        mechanisms.append({
            "id": mid, "moa": r.get("mechanismOfAction"),
            "action_type": r.get("actionType"), "target_name": r.get("targetName"),
            "target_symbols": syms, "source": "opentargets/chembl",
            "ref_url": ref_url or "https://platform.opentargets.org/",
        })
        for c in r.get("chemblIds") or []:
            links.append((f"chembl:{c}", mid))
    return substances, names, mechanisms, links


def load_gtopdb(substances: dict, names: list, mechanisms: list, links: list):
    lig_path = DATA_DIR / "gtopdb" / "ligands.csv"
    int_path = DATA_DIR / "gtopdb" / "interactions.csv"

    def _rows(path: Path):
        with open(path, newline="", encoding="utf-8") as f:
            first = f.readline()
            if not first.lstrip('"').startswith("#"):  # '"# GtoPdb Version..."'
                f.seek(0)
            return list(csv.DictReader(f))

    ligands = _rows(lig_path)
    inters = _rows(int_path)
    print(f"[gtp] {len(ligands)} ligands, {len(inters)} interactions")

    chembl_to_key = {v["chembl_id"]: k for k, v in substances.items() if v.get("chembl_id")}
    # Open Targets preferred names (priority-0, unique) -- a GtoPdb ligand
    # whose name/INN equals one is the SAME substance even without a
    # ChEMBL id on the GtoPdb side; creating a second node would make
    # every lookup of that name a tie between two keys (= unresolved).
    pref_counts: dict[str, int] = defaultdict(int)
    for norm, key, pri in names:
        if pri == 0:
            pref_counts[norm] += 1
    pref_to_key = {norm: key for norm, key, pri in names
                   if pri == 0 and pref_counts[norm] == 1}
    gtp_key: dict[str, str] = {}
    n_new = 0
    for l in ligands:
        if (l.get("Species") or "").strip() not in ("", "Human"):
            continue
        lid = l["Ligand ID"]
        chembl = (l.get("ChEMBL ID") or "").strip()
        key = chembl_to_key.get(chembl) if chembl else None
        if key is None:
            for nm in (l.get("INN"), l.get("Name")):
                k = pref_to_key.get(norm_name(re.sub(r"<[^>]+>", "", nm or "")))
                if k:
                    key = k
                    break
        if key is None:
            key = f"gtp:{lid}"
            if key not in substances:
                substances[key] = {
                    "key": key, "pref_name": l.get("Name"), "chembl_id": chembl or None,
                    "gtp_id": lid, "inchikey": l.get("InChIKey") or None,
                    "stage": "APPROVAL" if (l.get("Approved") or "").strip() else None,
                    "drug_type": l.get("Type"), "parent_chembl": None,
                }
                n_new += 1
        else:
            substances[key]["gtp_id"] = lid
        gtp_key[lid] = key
        for nm, pri in ((l.get("Name"), 0), (l.get("INN"), 0)):
            if nm and nm.strip():
                names.append((norm_name(re.sub(r"<[^>]+>", "", nm)), key, pri))
        for syn in (l.get("Synonyms") or "").split("|"):
            syn = re.sub(r"<[^>]+>", "", syn).strip()
            syn = re.sub(r"\s*\[.*?\]\s*$", "", syn)  # "compound 33 [PMID: ...]"
            if syn and not syn.lower().startswith("compound "):
                names.append((norm_name(syn), key, 1))
    print(f"[gtp] {n_new} substances not in Open Targets")

    by_lig: dict[str, list[dict]] = defaultdict(list)
    for r in inters:
        if (r.get("Target Species") or "").strip() not in ("", "Human"):
            continue
        by_lig[r["Ligand ID"]].append(r)
    n_mech = 0
    for lid, rows in by_lig.items():
        key = gtp_key.get(lid)
        if not key:
            continue
        # prefer primary-target rows; else all, deduped by (symbol, action)
        primary = [r for r in rows if (r.get("Primary Target") or "").lower() == "true"]
        use = primary or rows
        seen = set()
        for r in use:
            sym = (r.get("Target Gene Symbol") or "").strip()
            tname = re.sub(r"<[^>]+>", "", r.get("Target") or "").strip()
            action = (r.get("Action") or r.get("Type") or "").strip()
            if not (sym or tname) or action.lower() in ("", "none", "unknown", "other"):
                continue
            sig = (sym or tname, action.lower())
            if sig in seen:
                continue
            seen.add(sig)
            mid = f"gtp:{lid}:{len(seen)}"
            mechanisms.append({
                "id": mid, "moa": f"{tname or sym} {action.lower()}",
                "action_type": action.upper(), "target_name": tname or sym,
                "target_symbols": [sym] if sym else [], "source": "gtopdb",
                "ref_url": f"https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId={lid}",
            })
            links.append((key, mid))
            n_mech += 1
    print(f"[gtp] {n_mech} mechanism rows")


# --- write ------------------------------------------------------------------
SCHEMA = [
    "CREATE CONSTRAINT substance_key IF NOT EXISTS FOR (s:Substance) REQUIRE s.key IS UNIQUE",
    "CREATE CONSTRAINT mechanism_id IF NOT EXISTS FOR (m:Mechanism) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT drugname_norm IF NOT EXISTS FOR (n:DrugName) REQUIRE n.norm IS UNIQUE",
    "CREATE INDEX drug_name_idx IF NOT EXISTS FOR (d:Drug) ON (d.name)",
]
W_SUB = """UNWIND $rows AS r
MERGE (s:Substance {key: r.key})
SET s.pref_name = r.pref_name, s.chembl_id = r.chembl_id, s.gtp_id = r.gtp_id,
    s.inchikey = r.inchikey, s.stage = r.stage, s.drug_type = r.drug_type,
    s.parent_chembl = r.parent_chembl"""
W_MECH = """UNWIND $rows AS r
MERGE (m:Mechanism {id: r.id})
SET m.moa = r.moa, m.action_type = r.action_type, m.target_name = r.target_name,
    m.target_symbols = r.target_symbols, m.source = r.source, m.ref_url = r.ref_url"""
W_LINK = """UNWIND $rows AS r
MATCH (s:Substance {key: r.key}) MATCH (m:Mechanism {id: r.mid})
MERGE (s)-[:HAS_MECHANISM]->(m)"""
W_NAME = """UNWIND $rows AS r
MATCH (s:Substance {key: r.key})
MERGE (n:DrugName {norm: r.norm})
MERGE (n)-[e:RESOLVES_TO]->(s)
ON CREATE SET e.priority = r.priority
ON MATCH SET e.priority = CASE WHEN r.priority < e.priority THEN r.priority ELSE e.priority END"""
# Child salt forms inherit the parent molecule's mechanisms.
W_INHERIT = """MATCH (c:Substance) WHERE c.parent_chembl IS NOT NULL
MATCH (p:Substance {chembl_id: c.parent_chembl})-[:HAS_MECHANISM]->(m)
WHERE NOT (c)-[:HAS_MECHANISM]->()
MERGE (c)-[:HAS_MECHANISM]->(m)"""


def _batched(session, query: str, rows: list[dict], label: str) -> None:
    for i in range(0, len(rows), BATCH):
        session.run(query, rows=rows[i:i + BATCH]).consume()
    print(f"[neo4j] {label}: {len(rows)}")


def write(driver, substances, names, mechanisms, links) -> None:
    with driver.session() as s:
        for q in SCHEMA:
            s.run(q).consume()
        _batched(s, W_SUB, list(substances.values()), "Substance")
        _batched(s, W_MECH, mechanisms, "Mechanism")
        _batched(s, W_LINK, [{"key": k, "mid": m} for k, m in links
                             if k in substances], "HAS_MECHANISM")
        # best priority per (norm, key)
        best: dict[tuple[str, str], int] = {}
        for norm, key, pri in names:
            if len(norm) < 3 or key not in substances:
                continue
            best[(norm, key)] = min(pri, best.get((norm, key), 9))
        _batched(s, W_NAME, [{"norm": n, "key": k, "priority": p}
                             for (n, k), p in best.items()], "DrugName->RESOLVES_TO")
        s.run(W_INHERIT).consume()
        counts = s.run("""
            MATCH (s:Substance) WITH count(s) AS subs
            MATCH (m:Mechanism) WITH subs, count(m) AS mechs
            MATCH (n:DrugName) WITH subs, mechs, count(n) AS names
            MATCH (x:Substance)-[:HAS_MECHANISM]->() RETURN subs, mechs, names,
                   count(DISTINCT x) AS with_mech""").single()
        print(f"[neo4j] substances={counts['subs']} mechanisms={counts['mechs']} "
              f"names={counts['names']} substances_with_mechanism={counts['with_mech']}")


def verify(driver) -> None:
    probes = ["BMS-986278", "ABX464", "Pembrolizumab 200 mg IV Q3W", "Keytruda",
              "Placebo matching BMS-986278", "semaglutide", "AMG 510", "NVL-655",
              "Lenvatinib (E7080)", "Metformin 500mg tablets", "Ad5-nCoV"]
    with driver.session() as s:
        facts = lookup_mechanisms(s, probes)
    for p in probes:
        f = facts.get(p)
        if f:
            print(f"  {p!r:<36} -> {f['pref_name']} [{f['stage']}] via {f['matched_key']!r}: "
                  f"{mechanism_phrase(f) or '(no mechanism)'}")
        else:
            print(f"  {p!r:<36} -> (unresolved)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="only probe a few names")
    args = ap.parse_args()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    if args.verify:
        verify(driver)
        return 0
    t0 = time.perf_counter()
    download_all()
    substances, names, mechanisms, links = load_opentargets()
    load_gtopdb(substances, names, mechanisms, links)
    write(driver, substances, names, mechanisms, links)
    verify(driver)
    print(f"[done] {(time.perf_counter() - t0) / 60:.1f} min")
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
