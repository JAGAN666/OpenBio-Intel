"""Export the Pydantic contract as JSON Schema for the frontend.

    python export_schema.py

Pydantic is the single source of truth for the API contract. This writes
`frontend/schema.json`, which `npm run sync-types` compiles into
`frontend/types/trial.ts`. A backend field rename therefore becomes a
frontend *compile error* rather than an undefined value in a table cell.

Run this whenever SmartTableResponse or TrialRow changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from research_agent import SmartTableResponse

OUT = Path(__file__).resolve().parent / "frontend" / "schema.json"


def main() -> int:
    schema = SmartTableResponse.model_json_schema()

    # json-schema-to-typescript names the root interface from `title`.
    schema.setdefault("title", "SmartTableResponse")

    # Pydantic omits `additionalProperties`, and json-schema-to-typescript then
    # emits an `[k: string]: unknown` index signature on every interface. That
    # is WEAKER than a hand-written type -- it silently permits `row.typo` --
    # which defeats the point of generating types at all. Closing the objects
    # here affects the exported schema only; Pydantic's runtime behaviour
    # (ignore extras) is unchanged, so the LLM path cannot start failing.
    def close(obj: dict) -> None:
        if obj.get("type") == "object" or "properties" in obj:
            obj["additionalProperties"] = False

    close(schema)
    for definition in schema.get("$defs", {}).values():
        close(definition)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    defs = list(schema.get("$defs", {}).keys())
    print(f"[export_schema] wrote {OUT}")
    print(f"[export_schema] root      : {schema['title']}")
    print(f"[export_schema] $defs     : {defs}")
    print(f"[export_schema] properties: {list(schema.get('properties', {}).keys())}")
    print(f"[export_schema] required  : {schema.get('required')}")
    print("[export_schema] next: cd frontend && npm run sync-types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
