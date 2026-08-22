"""Unit tests for the entity-exactness primitives -- no services, no LLMs.

These encode the product guarantee that triggered the work: a query about
drug AABC must NEVER match lookalike AABD (or AABCD), while legitimate
formatting variants (hyphens, spaces, strength suffixes, parentheticals)
must still match. If any of these fail, lookalike leakage is possible and
the build must not ship.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent import (  # noqa: E402
    _alias_pattern,
    _entity_matches_trial,
    _norm_text,
)


def _matches(alias: str, text: str) -> bool:
    p = _alias_pattern(alias)
    return bool(p and p.search(text))


class TestLookalikeRejection:
    def test_one_letter_suffix_difference(self):
        assert _matches("AABC", "A study of AABC in melanoma")
        assert not _matches("AABC", "A study of AABD in melanoma")
        assert not _matches("AABD", "A study of AABC in melanoma")

    def test_longer_code_not_matched_by_shorter(self):
        assert not _matches("AABC", "A study of AABCD tablets")
        assert not _matches("AABC", "XAABC dosing")

    def test_dev_code_number_variants(self):
        assert _matches("BMS-986278", "BMS-986278 in IPF")
        assert not _matches("BMS-986278", "BMS-986279 in IPF")
        assert not _matches("BMS-986278", "BMS-98627 in IPF")
        assert not _matches("BMS-986278", "BMS-9862781 in IPF")

    def test_numeric_neighbors(self):
        assert _matches("ABX464", "ABX464 (obefazimod)")
        assert not _matches("ABX464", "ABX465 dosing")
        assert not _matches("ABX464", "ABX4640 dosing")


class TestFormattingVariants:
    def test_hyphen_space_equivalence(self):
        assert _matches("BMS 986278", "BMS-986278")
        assert _matches("BMS-986278", "BMS 986278")
        assert _matches("BMS-986278", "BMS986278")

    def test_strength_suffixes(self):
        assert _matches("pembrolizumab", "Pembrolizumab 200mg IV")
        assert _matches("pembrolizumab", "Pembrolizumab (MK-3475)")
        assert not _matches("pembrolizumab", "pembrolizumabX")

    def test_case_insensitive(self):
        assert _matches("Keytruda", "KEYTRUDA(R) injection")
        assert _matches("keytruda", "Keytruda")

    def test_multiword(self):
        assert _matches("trastuzumab deruxtecan", "Trastuzumab Deruxtecan (T-DXd)")
        assert not _matches("trastuzumab deruxtecan", "trastuzumab emtansine")


class TestTrialMatching:
    TRIAL = {
        "BriefTitle": "A Phase 3 Study of AABC Plus Chemotherapy",
        "interventions": [{"type": "DRUG", "name": "AABC 50mg"},
                          {"type": "DRUG", "name": "Carboplatin"}],
        "conditions": ["Non-small Cell Lung Cancer"],
        "LeadSponsorName": "Merck Sharp & Dohme LLC",
        "BriefSummary": "Evaluates AABC combined with chemo.",
    }

    def test_drug_hits_and_lookalike_misses(self):
        p_good = [_alias_pattern("AABC")]
        p_bad = [_alias_pattern("AABD")]
        assert _entity_matches_trial(p_good, self.TRIAL, "drug")
        assert not _entity_matches_trial(p_bad, self.TRIAL, "drug")

    def test_company_checks_sponsor_only(self):
        p_merck = [_alias_pattern("Merck")]
        assert _entity_matches_trial(p_merck, self.TRIAL, "company")
        trial2 = dict(self.TRIAL, LeadSponsorName="Pfizer Inc.",
                      BriefSummary="Comparator arm uses a Merck product.")
        # mention in the summary is NOT sponsorship
        assert not _entity_matches_trial(p_merck, trial2, "company")

    def test_alias_any_of(self):
        pats = [_alias_pattern("Keytruda"), _alias_pattern("AABC")]
        assert _entity_matches_trial(pats, self.TRIAL, "drug")


class TestNorm:
    def test_norm(self):
        assert _norm_text("BMS-986,278!") == "bms 986 278"
        assert _norm_text("  Pembrolizumab  ") == "pembrolizumab"
        assert _alias_pattern("") is None
