"""Unit tests for drug-name normalisation -- no services."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drug_kb import mechanism_phrase, name_variants, norm_name  # noqa: E402


class TestNameVariants:
    def test_dose_and_route_stripped(self):
        v = name_variants("Pembrolizumab 200 mg IV Q3W")
        assert "pembrolizumab" in v
        assert v[0] == "pembrolizumab 200 mg iv q3w"  # full string tried first

    def test_head_before_paren_alias(self):
        v = name_variants("Lenvatinib (E7080)")
        assert v.index("lenvatinib") < v.index("e7080")

    def test_dev_code_run_together(self):
        v = name_variants("BMS-986278")
        assert "bms 986278" in v and "bms986278" in v

    def test_combination_splits_on_plus(self):
        v = name_variants("Pembrolizumab + Lenvatinib")
        assert "pembrolizumab" in v

    def test_short_names_dropped(self):
        assert name_variants("IV") == []

    def test_norm(self):
        assert norm_name("Keytruda®") == "keytruda"


class TestMechanismPhrase:
    def test_curated_preferred_over_gtopdb(self):
        fact = {"mechanisms": [
            {"moa": "LPA1 receptor antagonist", "target_symbols": ["LPAR1"], "source": "gtopdb"},
            {"moa": "Programmed cell death protein 1 inhibitor", "target_symbols": ["PDCD1"],
             "source": "opentargets/chembl"},
        ]}
        assert mechanism_phrase(fact) == "Programmed cell death protein 1 inhibitor (PDCD1)"

    def test_symbol_not_duplicated(self):
        fact = {"mechanisms": [{"moa": "GTPase KRas inhibitor", "target_symbols": ["KRAS"],
                                "source": "opentargets/chembl"}]}
        assert mechanism_phrase(fact) == "GTPase KRas inhibitor"

    def test_empty(self):
        assert mechanism_phrase({"mechanisms": []}) == ""
