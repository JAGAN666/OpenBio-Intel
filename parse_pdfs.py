"""Vision-language PDF parsing via LlamaParse.

Standard OCR reads a page as a flat raster scan (left-to-right,
top-to-bottom), so anything beyond a single-column layout -- a two-column
conference poster, an efficacy table with merged headers like
"Arm A (n=120)" spanning ORR/PFS/OS subcolumns, a footnote marker tied to one
specific cell -- gets its text interleaved and the row/column relationships
destroyed. LlamaParse's premium mode routes each page through a
vision-language model instead: it looks at the page as an IMAGE and reasons
about which cells belong to which row/column before emitting text, so it
reconstructs a real Markdown table instead of a scrambled character soup.

    python parse_pdfs.py path/to/document.pdf

NOTE ON `llama-parse`
    The `llama-parse` PyPI package is a thin, deprecated re-export of
    `llama_cloud_services.LlamaParse` (itself flagged for eventual migration
    to a newer unified `llama-cloud` SDK, per its own DeprecationWarning).
    Both names resolve to the identical class today -- verified directly
    (`llama_parse.LlamaParse is llama_cloud_services.LlamaParse`) -- so
    importing from `llama_parse`, as specified, is not a functionality
    tradeoff, just a name that will eventually need updating.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from llama_parse import LlamaParse  # noqa: E402


def run_async(coro):
    """Run a coroutine from synchronous code -- the entry point every
    caller (this module's own CLI, ingest_pipeline.py) actually uses.

    Deliberately does NOT call nest_asyncio.apply() unconditionally at
    import time. That was the first implementation, and it was WRONG:
    verified directly against this exact environment (Python 3.14, current
    httpx/httpcore/anyio/sniffio) that patching the loop eagerly breaks the
    plain, non-nested asyncio.run() path this code actually takes --
    httpcore's connection pool calls sniffio.current_async_library() to pick
    an async backend, and a nest_asyncio-patched loop makes that call raise
    AsyncLibraryNotFoundError even though nothing is actually nested. A live
    parse against this exact PDF failed with that traceback until the
    unconditional nest_asyncio.apply() was removed; the identical call
    succeeded immediately afterward.
    nest_asyncio is still a real dependency for the case it was installed
    for -- calling this from inside an ALREADY-running loop (Jupyter, an
    ASGI worker) -- so the patch is applied lazily, only in that fallback,
    only if asyncio.run() actually raises for that reason.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "cannot be called from a running event loop" not in str(exc):
            raise
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(coro)

_parser: LlamaParse | None = None


def _client() -> LlamaParse:
    """Lazy singleton. Constructing LlamaParse validates the API key
    eagerly, so a missing key fails fast here rather than mid-batch."""
    global _parser
    if _parser is None:
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise SystemExit(
                "[parse_pdfs] LLAMA_CLOUD_API_KEY is not set.\n"
                "             Add it to .env:  LLAMA_CLOUD_API_KEY=llx-..."
            )
        _parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",  # vision-language table reconstruction,
                                      # not a plain OCR text dump
            num_workers=4,           # parallel page-batch parsing
            premium_mode=True,       # forces the VLM path -- aggressively
                                      # preserves table structure on dense
                                      # documents (posters, FDA filings)
            verbose=True,
        )
    return _parser


async def process_pdf(file_path: str) -> str:
    """Upload one PDF, await vision-based Markdown parsing, return the text.

    A single call can internally yield more than one JobResult if LlamaParse
    partitions a very large document (e.g. a multi-hundred-page filing) --
    both the single- and multi-result cases are handled the same way here,
    and the parts are joined with a blank line between them.
    """
    result = await _client().aparse(file_path)
    job_results = result if isinstance(result, list) else [result]

    parts: list[str] = []
    for jr in job_results:
        if jr.error:
            raise RuntimeError(f"LlamaParse failed on {file_path!r}: {jr.error}")
        parts.append(await jr.aget_markdown())
    return "\n\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="path to a PDF file")
    args = parser.parse_args()

    path = Path(args.pdf)
    if not path.is_file():
        print(f"[parse_pdfs] not a file: {path}", file=sys.stderr)
        return 1

    print(f"[parse_pdfs] parsing {path} (premium vision-language mode)…")
    markdown = run_async(process_pdf(str(path)))
    print(f"[parse_pdfs] {len(markdown):,} chars of Markdown extracted")
    print("-" * 74)
    print(markdown[:2000])
    if len(markdown) > 2000:
        print(f"... ({len(markdown) - 2000:,} more chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
