"""One-off index migration for the entity-exactness engine.

1. Neo4j: range + text indexes on Drug(name) and Trial(sponsor) -- the
   exact-fetch queries anchor on these; regex itself can't use an index
   but the equality/anchor paths and future exact lookups can.
2. Qdrant: FULL-TEXT payload indexes on clinical_trials.interventionNames
   and .LeadSponsorName (replacing the KEYWORD indexes -- exact-value
   matching on raw strings like "Pembrolizumab 200mg" was useless for
   entity lookup, which is why lookalike padding leaked through kNN
   instead). MatchText hits are always post-verified with the boundary
   regex in research_agent._alias_pattern, so tokenizer looseness cannot
   reintroduce lookalikes.

Online operations on both stores; safe to run against a live deployment.
Idempotent. Usage:  uv run python migrate_entity_indexes.py
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

TEXT_FIELDS = ("interventionNames", "LeadSponsorName")


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        for stmt in (
            "CREATE INDEX drug_name_range IF NOT EXISTS FOR (d:Drug) ON (d.name)",
            "CREATE TEXT INDEX drug_name_text IF NOT EXISTS FOR (d:Drug) ON (d.name)",
            "CREATE INDEX trial_sponsor_range IF NOT EXISTS FOR (t:Trial) ON (t.sponsor)",
        ):
            s.run(stmt)
            print(f"[neo4j] {stmt.split(' IF ')[0]}: ok")
    driver.close()

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
    for field in TEXT_FIELDS:
        try:
            client.delete_payload_index("clinical_trials", field_name=field)
            print(f"[qdrant] dropped keyword index on {field}")
        except Exception as exc:  # noqa: BLE001 -- absent index is fine
            print(f"[qdrant] no existing index on {field} ({exc})")
        client.create_payload_index(
            "clinical_trials",
            field_name=field,
            field_schema=qmodels.TextIndexParams(
                type="text",
                tokenizer=qmodels.TokenizerType.WORD,
                lowercase=True,
            ),
            wait=True,
        )
        print(f"[qdrant] full-text index on {field}: ok")


if __name__ == "__main__":
    main()
