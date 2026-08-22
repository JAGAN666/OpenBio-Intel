"""Mine REAL lookalike entity pairs from the corpus into a golden eval set.

The user-reported failure class: asking about drug "AABC" surfaced trials
for lookalike "AABD". This script makes that class regression-proof by
finding actual near-miss name pairs in the live Neo4j graph (shared
dev-code stem with different suffixes, or edit distance <= 2 after
normalization), where BOTH names have small exact ground truths -- so the
eval can assert perfect precision AND recall without an LLM in the loop.

Ground truth per drug = the trials reachable by INVESTIGATES edge whose
drug name matches the boundary regex (the same verification standard the
runtime uses -- see research_agent._alias_pattern). For each mined pair
(A, B): A's case lists A's trials as expected and B's non-shared trials as
forbidden, and vice versa.

Hand-written cases appended at the end: brand->generic aliasing
(Keytruda -> pembrolizumab) and class queries labeled
expect_no_entity_filter -- the classifier must NOT emit entities for them.

Usage:  uv run python eval/build_entity_golden.py
Writes: eval/golden_entities.jsonl  (committed; rebuild deliberately)
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from neo4j import GraphDatabase  # noqa: E402

from research_agent import (  # noqa: E402
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    _alias_pattern,
    _expand_entity_aliases,
    _norm_text,
)

OUT = Path(__file__).resolve().parent / "golden_entities.jsonl"

# Pair-mining bounds. Small-N only: cases must be fully checkable against
# the runtime cap, and tiny ground truths are where kNN padding leaked
# lookalikes in the first place.
MIN_TRIALS = 1
MAX_TRIALS = 8
MAX_MINED_PAIRS = 15

# Dev-code shape: letter prefix + separator? + digits (+ optional suffix).
_CODE_RE = re.compile(r"^([a-z]{2,6})[\s-]?(\d{2,7})([a-z]?)$")


def _drug_trial_map(session) -> dict[str, set[str]]:
    """name -> set of NCT ids, boundary-regex-verified per drug name."""
    rows = session.run(
        "MATCH (d:Drug)<-[:INVESTIGATES]-(t:Trial) "
        "RETURN d.name AS name, collect(DISTINCT t.id) AS ncts"
    )
    out: dict[str, set[str]] = {}
    for r in rows:
        name = (r["name"] or "").strip()
        if len(_norm_text(name)) < 4:
            continue
        pat = _alias_pattern(name)
        if pat is None:
            continue
        out[name] = set(r["ncts"])
    return out


def _lookalike_pairs(names: list[str]) -> list[tuple[str, str]]:
    """Find (A, B) name pairs that a substring/kNN matcher would confuse."""
    pairs: set[tuple[str, str]] = set()
    norm = {n: _norm_text(n) for n in names}

    # 1. Shared dev-code stem, different number/suffix (BMS-986278 vs -986279).
    by_stem: dict[str, list[str]] = defaultdict(list)
    for n in names:
        m = _CODE_RE.match(norm[n].replace(" ", ""))
        if m:
            by_stem[m.group(1)].append(n)
    for stem_names in by_stem.values():
        stem_names.sort()
        for i, a in enumerate(stem_names):
            for b in stem_names[i + 1:]:
                if norm[a] != norm[b]:
                    pairs.add((a, b))

    # 2. Edit distance <= 2 on normalized names (AABC vs AABD class).
    #    Bucketed by 4-char prefix first -- a raw O(n^2) sweep over every
    #    drug name never finishes at corpus scale (verified: killed after
    #    5+ min), and real lookalike confusions share a stem anyway; a pair
    #    differing in its first four characters is not a plausible
    #    "one-keystroke-away" confusion.
    def _ed_le2(x: str, y: str) -> bool:
        if abs(len(x) - len(y)) > 2:
            return False
        # classic banded DP, early-exit
        prev = list(range(len(y) + 1))
        for i, cx in enumerate(x, 1):
            cur = [i] + [0] * len(y)
            row_min = i
            for j, cy in enumerate(y, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                             prev[j - 1] + (cx != cy))
                row_min = min(row_min, cur[j])
            if row_min > 2:
                return False
            prev = cur
        return prev[-1] <= 2

    by_prefix: dict[str, list[str]] = defaultdict(list)
    for n in names:
        compact = norm[n].replace(" ", "")
        if len(compact) >= 4:
            by_prefix[compact[:4]].append(n)
    for bucket in by_prefix.values():
        for i, a in enumerate(bucket):
            for b in bucket[i + 1:]:
                if norm[a] != norm[b] and _ed_le2(norm[a], norm[b]):
                    pairs.add(tuple(sorted((a, b))))

    return sorted(pairs)


HAND_CASES = [
    # Brand -> generic aliasing: asking by brand must recall the generic's
    # trials (RxNorm concept walk), not return zero rows.
    {"case_type": "brand_generic", "query_entity": "Keytruda", "kind": "drug",
     "alias_must_include": "pembrolizumab"},
    {"case_type": "brand_generic", "query_entity": "Wegovy", "kind": "drug",
     "alias_must_include": "semaglutide"},
    # Class/target/indication queries: the classifier must emit NO entities
    # (else the whole answer gets wrongly hard-filtered).
    {"case_type": "class_query", "expect_no_entity_filter": True,
     "query": "Compare Phase 3 trials of GLP-1 receptor agonists for obesity"},
    {"case_type": "class_query", "expect_no_entity_filter": True,
     "query": "What checkpoint inhibitor trials are recruiting in NSCLC?"},
    {"case_type": "class_query", "expect_no_entity_filter": True,
     "query": "Show me KRAS G12C inhibitor programs in Phase 2 or later"},
    {"case_type": "class_query", "expect_no_entity_filter": True,
     "query": "Antibody-drug conjugate trials in breast cancer"},
]


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        trial_map = _drug_trial_map(session)
    driver.close()

    small = {n: ncts for n, ncts in trial_map.items()
             if MIN_TRIALS <= len(ncts) <= MAX_TRIALS}
    print(f"[golden] {len(trial_map)} drugs with trials; "
          f"{len(small)} in the {MIN_TRIALS}-{MAX_TRIALS} trial band")

    pairs = _lookalike_pairs(sorted(small))
    print(f"[golden] {len(pairs)} candidate pairs mined")

    # A candidate pair is only a TRUE lookalike (distinct entities a sloppy
    # matcher would confuse) if the runtime itself would keep them apart:
    #   1. Neither name's boundary pattern matches the other -- otherwise
    #      they are formatting/strength variants of the SAME entity
    #      ('"Hydrocortisone"' vs 'Hydrocortisone 1%') and 'forbidden'
    #      would punish correct behavior.
    #   2. Alias expansion doesn't bridge them -- two spellings mapped to
    #      one RxNorm concept are one drug, not lookalikes.
    def _regex_distinct(a: str, b: str) -> bool:
        pa, pb = _alias_pattern(a), _alias_pattern(b)
        return bool(pa and pb and not pa.search(b) and not pb.search(a))

    def _alias_distinct(a: str, b: str) -> bool:
        a_aliases = _expand_entity_aliases(a, "drug")
        b_aliases = _expand_entity_aliases(b, "drug")
        pats_a = [p for p in map(_alias_pattern, a_aliases) if p]
        pats_b = [p for p in map(_alias_pattern, b_aliases) if p]
        return (not any(p.search(x) for p in pats_a for x in b_aliases)
                and not any(p.search(x) for p in pats_b for x in a_aliases))

    # Cheap regex filter over everything; the alias check costs a Neo4j
    # regex scan per name (~seconds each over 150K Drug nodes -- thousands
    # of candidates took >15 min and were killed), so only the dev-code-
    # shaped survivors (the user-reported AABC/AABD class) are checked,
    # and only until MAX_MINED_PAIRS pass.
    cheap = [pq for pq in pairs if _regex_distinct(*pq)]
    cheap.sort(key=lambda pq: (not all(_CODE_RE.match(
        _norm_text(n).replace(" ", "")) for n in pq), pq))
    print(f"[golden] {len(cheap)} regex-distinct candidates; alias-checking "
          f"until {MAX_MINED_PAIRS} pass")
    pairs = []
    for pq in cheap:
        if len(pairs) >= MAX_MINED_PAIRS:
            break
        if _alias_distinct(*pq):
            pairs.append(pq)
    print(f"[golden] {len(pairs)} true lookalike pairs after "
          f"same-entity exclusion")

    cases: list[dict] = []
    for a, b in pairs:
        a_ncts, b_ncts = small[a], small[b]
        shared = a_ncts & b_ncts  # co-investigated trials are NOT forbidden
        for query, own, other in ((a, a_ncts, b_ncts), (b, b_ncts, a_ncts)):
            forbidden = sorted(other - own - shared)
            if not forbidden:
                continue
            cases.append({
                "case_type": "lookalike", "query_entity": query,
                "kind": "drug", "lookalike_of": b if query == a else a,
                "expected_nct_ids": sorted(own),
                "forbidden_nct_ids": forbidden,
            })
    cases.extend(HAND_CASES)

    with OUT.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    n_look = sum(1 for c in cases if c["case_type"] == "lookalike")
    print(f"[golden] wrote {OUT.name}: {n_look} lookalike, "
          f"{len(HAND_CASES)} hand-written")
    for c in cases[:6]:
        if c["case_type"] == "lookalike":
            print(f"  e.g. {c['query_entity']!r} vs {c['lookalike_of']!r} "
                  f"({len(c['expected_nct_ids'])} expected, "
                  f"{len(c['forbidden_nct_ids'])} forbidden)")


if __name__ == "__main__":
    main()
