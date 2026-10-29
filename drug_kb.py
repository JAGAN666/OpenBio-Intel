"""Drug-name normalisation + runtime mechanism lookup against the drug
knowledge base that build_drug_kb.py writes into Neo4j.

Graph shape (all written by build_drug_kb.py):

    (:DrugName {norm})-[:RESOLVES_TO {priority}]->(:Substance {key, pref_name,
        chembl_id, gtp_id, inchikey, stage, drug_type})
    (:Substance)-[:HAS_MECHANISM]->(:Mechanism {id, moa, action_type,
        target_name, target_symbols, source, ref_url})

`priority` on the name edge encodes how trustworthy the name->substance
link is: 0 = the substance's own preferred name / INN, 1 = a curated
synonym or trade name (ChEMBL, GtoPdb), 2 = a registry-derived label
(Open Targets' AACT synonyms, which are noisy -- "pembrolizumab" is
attached to nivolumab there). Lookups take the best priority and treat
a tie between different substances as ambiguous (no answer beats a
wrong one).

Name variants tried for a registry intervention string such as
"Pembrolizumab 200 mg IV Q3W (MK-3475)": the full string, the string
with dose/route/schedule tokens stripped, the text before the first
parenthesis/comma, and the parenthesised alias itself.
"""
from __future__ import annotations

import re

_DOSE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|g|ml|mL|iu|units?|%|mg/kg|mg/m2|mg/m²|"
    r"mcg/kg|ug/kg|gy|x)\b", re.IGNORECASE)
_ROUTE_WORDS = {
    "tablet", "tablets", "capsule", "capsules", "injection", "injections",
    "oral", "orally", "iv", "i.v.", "intravenous", "intravenously", "sc",
    "s.c.", "subcutaneous", "subcutaneously", "im", "intramuscular",
    "solution", "infusion", "infusions", "dose", "doses", "low", "high",
    "once", "twice", "daily", "weekly", "monthly", "qd", "bid", "tid",
    "q2w", "q3w", "q4w", "qw", "arm", "group", "cohort", "regimen",
    "formulation", "extended", "release", "er", "xr", "sr", "cream",
    "ointment", "gel", "patch", "spray", "drops", "eye", "topical",
    "inhaled", "inhalation", "nasal", "film", "coated", "powder", "vial",
    "pen", "prefilled", "syringe", "for", "of", "with", "and", "plus",
    "the", "a", "an", "in", "to", "per", "day", "days", "week", "weeks",
    "cycle", "cycles", "standard", "care", "therapy", "treatment", "drug",
    "study", "test", "reference", "product", "active", "comparator",
    "control", "placebo", "matching", "matched", "sham", "vehicle",
}


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").casefold()).strip()


def name_variants(raw: str) -> list[str]:
    """Ordered, de-duplicated candidate keys for one registry name."""
    raw = (raw or "").strip()
    if not raw:
        return []
    out: list[str] = []

    def _add(s: str) -> None:
        n = norm_name(s)
        if len(n) >= 3 and n not in out:
            out.append(n)

    _add(raw)
    head = re.split(r"[(,;:/]| \+ | - ", raw, maxsplit=1)[0]
    _add(head)
    stripped = _DOSE_RE.sub(" ", head)
    tokens = [t for t in norm_name(stripped).split() if t not in _ROUTE_WORDS]
    _add(" ".join(tokens))
    # parenthesised aliases: "Pembrolizumab (MK-3475)" -> the code too
    for inner in re.findall(r"\(([^()]{2,60})\)", raw):
        _add(inner)
    # leading development code: "BMS-986278 Batched method, Dose A" ->
    # "bms 986278"; "AZD1234 10 mg" -> "azd1234"
    m = re.match(r"([a-z]{1,6})[ -]?(\d{3,7}[a-z]?)\b", norm_name(head))
    if m:
        _add(f"{m.group(1)} {m.group(2)}")
        _add(m.group(1) + m.group(2))
    # first token alone when it looks like a dev code or a single INN
    if tokens and (re.search(r"\d", tokens[0]) or len(tokens) == 1):
        _add(tokens[0])
    # dev codes: "bms 986278" also as "bms986278"
    for v in list(out):
        m = re.fullmatch(r"([a-z]{1,5}) (\d{3,7})", v)
        if m:
            _add(m.group(1) + m.group(2))
    return out


LOOKUP_QUERY = """
UNWIND $keys AS k
MATCH (n:DrugName {norm: k})-[r:RESOLVES_TO]->(s:Substance)
OPTIONAL MATCH (s)-[:HAS_MECHANISM]->(m:Mechanism)
WITH k, r.priority AS priority, s,
     collect(DISTINCT {moa: m.moa, action_type: m.action_type,
                       target_name: m.target_name,
                       target_symbols: m.target_symbols,
                       source: m.source, ref_url: m.ref_url}) AS mechs
RETURN k, priority, s.key AS key, s.pref_name AS pref_name,
       s.chembl_id AS chembl_id, s.gtp_id AS gtp_id, s.stage AS stage,
       s.drug_type AS drug_type, mechs
"""


def lookup_mechanisms(session, names: list[str]) -> dict[str, dict]:
    """{raw name: fact} for every name the KB can resolve unambiguously.

    fact = {pref_name, chembl_id, stage, mechanisms: [{moa, action_type,
    target_name, target_symbols, source, ref_url}], matched_key}
    Names resolving to nothing are simply absent."""
    variants = {n: name_variants(n) for n in names}
    keys = sorted({k for vs in variants.values() for k in vs})
    if not keys:
        return {}
    rows = session.run(LOOKUP_QUERY, keys=keys).data()
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(r["k"], []).append(r)

    out: dict[str, dict] = {}
    for raw, vs in variants.items():
        for k in vs:  # best variant first
            cands = by_key.get(k)
            if not cands:
                continue
            best_p = min(c["priority"] for c in cands)
            best = {c["key"]: c for c in cands if c["priority"] == best_p}
            if len(best) != 1:
                continue  # ambiguous at this variant; try the next
            c = next(iter(best.values()))
            mechs = [m for m in (c["mechs"] or []) if m.get("moa")]
            out[raw] = {
                "pref_name": c["pref_name"], "chembl_id": c["chembl_id"],
                "gtp_id": c["gtp_id"], "stage": c["stage"],
                "drug_type": c["drug_type"], "matched_key": k,
                "mechanisms": mechs,
            }
            break
    return out


def mechanism_phrase(fact: dict) -> str:
    """One analyst-grade phrase from a KB fact, e.g.
    'PD-1 inhibitor (PDCD1)' or 'LPA1 receptor antagonist (LPAR1)';
    multiple distinct mechanisms joined with '; '."""
    seen: list[str] = []
    mechs = fact.get("mechanisms") or []
    # Curated ChEMBL/Open Targets rows are the canonical phrasing; GtoPdb
    # interaction rows only fill in when no curated row exists.
    curated = [m for m in mechs if (m.get("source") or "").startswith("opentargets")]
    for m in curated or mechs:
        moa = (m.get("moa") or "").strip()
        syms = [s for s in (m.get("target_symbols") or []) if s]
        phrase = moa
        if syms and not any(s.casefold() in moa.casefold() for s in syms):
            phrase += f" ({', '.join(syms[:3])})"
        if phrase and phrase not in seen:
            seen.append(phrase)
    return "; ".join(seen)
