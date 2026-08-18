"""Export the Pydantic contracts as JSON Schema for the frontend.

    python export_schema.py

Pydantic is the single source of truth for the API contract. This writes
`frontend/schema.json` (SmartTableResponse) and `frontend/schema_landscape.json`
(LandscapeMatrix), which `npm run sync-types` / `npm run sync-types:landscape`
compile into `frontend/types/trial.ts` / `frontend/types/landscape.ts`. A
backend field rename therefore becomes a frontend *compile error* rather
than an undefined value in a table cell.

Run this whenever SmartTableResponse, TrialRow, LandscapeMatrix, or any of
its nested models change.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from research_agent import CatalystTimeline, LandscapeMatrix, SmartTableResponse

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

# (Pydantic model, output filename, root schema title) -- one entry per
# contract exported to the frontend. Adding a new API response type is a
# one-line addition here, not a second copy of main()'s logic.
EXPORTS: list[tuple[type[BaseModel], str, str]] = [
    (SmartTableResponse, "schema.json", "SmartTableResponse"),
    (LandscapeMatrix, "schema_landscape.json", "LandscapeMatrix"),
    (CatalystTimeline, "schema_catalysts.json", "CatalystTimeline"),
]


def _export_one(model: type[BaseModel], filename: str, title: str) -> dict:
    schema = model.model_json_schema()

    # json-schema-to-typescript names the root interface from `title`.
    schema.setdefault("title", title)

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

    out = FRONTEND_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    print(f"[export_schema] wrote {out}")
    print(f"[export_schema]   root      : {schema['title']}")
    print(f"[export_schema]   $defs     : {list(schema.get('$defs', {}).keys())}")
    print(f"[export_schema]   properties: {list(schema.get('properties', {}).keys())}")
    print(f"[export_schema]   required  : {schema.get('required')}")
    return schema


def main() -> int:
    for model, filename, title in EXPORTS:
        _export_one(model, filename, title)
    print("[export_schema] next: cd frontend && npm run sync-types && npm run sync-types:landscape "
         "&& npm run sync-types:catalysts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
