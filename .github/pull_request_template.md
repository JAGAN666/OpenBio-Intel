## What & why

<!-- One or two sentences: what changes, and what problem it solves. -->

## How it was verified

<!-- This project's standing rule: verified live, not assumed. Paste the
     command(s) you ran and the actual output that proves it works —
     a real query against a running stack, eval numbers before/after,
     a build log. "It should work" doesn't merge. -->

## Checklist

- [ ] `uv run pytest -q` passes (retrieval eval skips cleanly without a corpus)
- [ ] `flake8 --select=E9,F63,F7,F82` clean; frontend `npm run lint` + `tsc --noEmit` clean if touched
- [ ] Changed a Pydantic response model? Regenerated frontend types (`export_schema.py` + `npm run sync-types`)
- [ ] Changed dependencies? Committed the regenerated `uv.lock` (and `requirements-etl.lock.txt` if ETL)
- [ ] New Qdrant collection or agent tool? Registered its sparse-text builder / threaded its payload through synthesis (see `sparse_embeddings.py` and `tools_node`)
