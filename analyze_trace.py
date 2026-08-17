"""Explain a faithfulness score: which claims failed, and why.

    python analyze_trace.py
    python analyze_trace.py --trace eval_traces/eval_trace_final.json

A faithfulness score alone ("0.77") is not actionable -- it doesn't say which
sentences the judge rejected or what its stated reason was. This script
re-runs the two judge calls that make up ragas' Faithfulness metric --
statement decomposition, then NLI verdicts -- against the SAME saved trace
and SAME judge config as evaluate_agent.py, and prints only the statements
that scored 0, with the judge's own `reason` field for each.

IMPORTANT ASYMMETRY, worth understanding before reading the output:
ragas' Faithfulness metric concatenates ALL retrieved contexts into a single
blob and judges each statement against that whole blob (see
ragas/metrics/_faithfulness.py: `contexts_str = "\n".join(retrieved_contexts)`).
The judge is never asked "which context supports this statement" -- so there
is no ground truth for NCTId attribution to report. The `likely source`
printed below is OUR OWN post-hoc heuristic (lexical overlap between the
failed statement and each context block), not something the judge returned.
It is a best-effort pointer for a human reviewer, not a citation.

The verdict call is CHUNKED (a handful of statements per request, all against
the same full context), not sent as one giant call -- unchunked, a ~40
statement answer produces enough expected JSON output (verdict + reason per
statement) to reliably 504 against NVIDIA's gateway. Chunking is a request
shape decision only; it does not change what the judge is allowed to see.
"""

from __future__ import annotations

# --- same compatibility shim as evaluate_agent.py, required before `ragas` ---
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:  # pragma: no cover
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # noqa: D401
            """Inert stand-in; ragas only uses this in isinstance() tests."""

        _stub.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = _stub

import argparse
import asyncio
import json
import os
import re
import textwrap
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv

from evaluate_agent import EMBED_MODEL, JUDGE_MODEL, build_judge  # reuse, don't drift

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

DEFAULT_TRACE = Path(__file__).resolve().parent / "eval_traces" / "eval_trace_final.json"


def load_trace(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"[analyze] trace not found: {path}\n"
            f"          Generate one with: python evaluate_agent.py --dump {path}"
        )
    trace = json.loads(path.read_text(encoding="utf-8"))
    missing = {"user_input", "retrieved_contexts", "response"} - trace.keys()
    if missing:
        raise SystemExit(f"[analyze] trace is missing required keys: {missing}")
    return trace


def guess_source(statement: str, contexts: list[str]) -> tuple[str | None, float]:
    """Best-effort lexical match, NOT a judge-reported attribution.

    ragas judges every statement against all contexts concatenated as one
    blob (see module docstring) -- it has no notion of "this context". This
    picks the context with the highest token overlap to the failed
    statement, purely to give a reviewer somewhere to start looking.
    """
    stop = {
        "the", "a", "an", "is", "was", "are", "were", "of", "in", "to", "and",
        "or", "for", "with", "by", "as", "on", "at", "this", "that", "it",
        "its", "be", "has", "have", "not", "no",
    }

    def tokens(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower())
                if len(w) > 2 and w not in stop}

    stmt_tokens = tokens(statement)
    if not stmt_tokens:
        return None, 0.0

    best_nct, best_score = None, 0.0
    for ctx in contexts:
        m = re.search(r"NCTId:\s*(NCT\d+)", ctx)
        nct = m.group(1) if m else None
        overlap = len(stmt_tokens & tokens(ctx)) / len(stmt_tokens)
        if overlap > best_score:
            best_nct, best_score = nct, overlap

    return best_nct, best_score


async def analyze(trace: dict, judge_model: str, embed_model: str,
                  max_tokens: int, chunk_size: int) -> None:
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness
    from ragas.run_config import RunConfig

    llm, _embeddings = build_judge(judge_model, embed_model, max_tokens)
    wrapped_llm = LangchainLLMWrapper(llm)
    # Match the RunConfig that succeeded in evaluate_agent.py rather than
    # leaving ragas' bare default (timeout=180s, max_retries=10). Calling the
    # private methods directly, as this script does, bypasses evaluate()'s
    # internal wiring that would otherwise propagate a custom RunConfig here.
    wrapped_llm.set_run_config(RunConfig(timeout=900, max_retries=4, max_wait=90))
    metric = Faithfulness(llm=wrapped_llm)

    row = {
        "user_input": trace["user_input"],
        "response": trace["response"],
        "retrieved_contexts": trace["retrieved_contexts"],
    }

    print(f"[1/2] decomposing the answer into atomic statements "
          f"({len(row['response']):,} chars)…")
    gen = await metric._create_statements(row, callbacks=None)  # noqa: SLF001
    statements = gen.statements
    print(f"      {len(statements)} statements extracted")

    # Chunked, not one call for all statements. The NLI-verdict call re-sends
    # the full concatenated context PLUS a verdict+reason per statement as
    # expected output -- for 40 statements that is several thousand output
    # tokens in a single response, which reliably 504'd against NVIDIA's
    # gateway on this trace (confirmed: the much smaller statement-generation
    # call never failed). Chunking keeps context size constant per call but
    # caps the output size, and runs sequentially so a flaky/rate-limited
    # endpoint isn't hit concurrently from multiple chunks at once.
    n_chunks = (len(statements) + chunk_size - 1) // chunk_size
    print(f"[2/2] judging {len(statements)} statements against "
          f"{len(row['retrieved_contexts'])} concatenated contexts "
          f"({n_chunks} chunk(s) of <= {chunk_size})…")
    all_verdicts = []
    for i in range(0, len(statements), chunk_size):
        chunk = statements[i:i + chunk_size]
        print(f"      chunk {i // chunk_size + 1}/{n_chunks} "
              f"({len(chunk)} statements)…")
        result = await metric._create_verdicts(row, chunk, callbacks=None)  # noqa: SLF001
        all_verdicts.extend(result.statements)

    from ragas.metrics._faithfulness import NLIStatementOutput
    verdicts = NLIStatementOutput(statements=all_verdicts)
    score = metric._compute_score(verdicts)  # noqa: SLF001

    passed = [v for v in verdicts.statements if v.verdict]
    failed = [v for v in verdicts.statements if not v.verdict]

    line = "=" * 78
    print(f"\n{line}\nFAITHFULNESS BREAKDOWN\n{line}")
    print(f"  judge          : {judge_model}")
    print(f"  statements     : {len(statements)} total  "
          f"({len(passed)} grounded, {len(failed)} unsupported)")
    print(f"  recomputed score: {score:.4f}")

    if not failed:
        print("\n  No unsupported statements — every claim was grounded. "
              "(If the last evaluate_agent.py run scored below 1.0, this is "
              "likely a different agent run; scoring is not perfectly "
              "deterministic even at temperature=0.)")
        return

    print(f"\n{'-' * 78}\nUNSUPPORTED CLAIMS  (verdict = 0)\n{'-' * 78}")
    for i, v in enumerate(failed, 1):
        nct, overlap = guess_source(v.statement, row["retrieved_contexts"])
        source_note = (
            f"{nct}  (lexical overlap {overlap:.0%}, heuristic only)"
            if nct and overlap >= 0.15
            else "no single context clearly implicated — possibly a "
                 "cross-context synthesis or a claim with no source at all"
        )
        print(f"\n[{i}] CLAIM:")
        for line_ in textwrap.wrap(v.statement, 74):
            print(f"      {line_}")
        print(f"    JUDGE'S REASON:")
        for line_ in textwrap.wrap(v.reason, 74):
            print(f"      {line_}")
        print(f"    LIKELY SOURCE: {source_note}")

    print(f"\n{'-' * 78}")
    print(f"{len(failed)} of {len(statements)} statements flagged unsupported "
          f"— see engineering assessment.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--judge-max-tokens", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="statements per NLI-verdict call; lower this if "
                             "you still see 504s from the judge endpoint")
    args = parser.parse_args()

    print("=" * 78)
    print("medical-rag :: faithfulness failure analysis")
    print("=" * 78)
    print(f"[analyze] trace: {args.trace}")

    trace = load_trace(args.trace)
    print(f"[analyze] question: {trace['user_input'][:80]}…")
    print(f"[analyze] {len(trace['retrieved_contexts'])} contexts, "
          f"{len(trace['response']):,}-char answer")

    asyncio.run(analyze(trace, args.judge_model, args.embed_model,
                        args.judge_max_tokens, args.chunk_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
