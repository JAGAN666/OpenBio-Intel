"""Automated Ragas evaluation of the LangGraph research agent.

    python evaluate_agent.py

Pipeline:
  A. run the agent on a test query
  B. capture the trace: question, retrieved contexts, generated answer
  C. shape it into a HuggingFace Dataset in the format Ragas expects
  D. score with faithfulness + answer_relevancy using an INDEPENDENT judge
  E. print the scores

The judge is deliberately NOT the generator. If Claude both wrote the answer
and graded it, `faithfulness` would measure self-consistency rather than
grounding -- the same priors that produced a claim would also vouch for it.
NVIDIA's Nemotron is a different model family, so it re-derives entailment
from the contexts alone. It is also free, which keeps the eval loop cheap
enough to run on every change.
"""

from __future__ import annotations

# --- compatibility shim (must precede the ragas import) ---------------------
# ragas 0.4.3 hard-imports langchain_community.chat_models.vertexai, which the
# sunsetting langchain-community 0.4.x no longer ships. The symbol is used in
# exactly one place: a MULTIPLE_COMPLETION_SUPPORTED list of isinstance()
# targets. Our judge is never a Vertex model, so a stub is behaviourally inert
# -- and far safer than downgrading langchain-community, which would drag
# langchain-core below 1.0 and break langgraph and the agent itself.
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:  # pragma: no cover - depends on installed langchain-community
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # noqa: D401 - placeholder for an isinstance check
            """Inert stand-in; ragas only uses this in isinstance() tests."""

        _stub.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = _stub

import argparse
import json
import math
import os
import textwrap
import time
from pathlib import Path

# --- TLS trust store --------------------------------------------------------
# ragas scores over aiohttp, which builds an ssl context from the SYSTEM trust
# store. The python.org macOS build ships no root certificates there, so calls
# to integrate.api.nvidia.com fail with CERTIFICATE_VERIFY_FAILED and the
# metric silently returns NaN. (Sync clients like httpx are unaffected -- they
# use certifi directly, which is why a smoke test passes while scoring fails.)
# Point OpenSSL at certifi before any ssl context is created.
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

from research_agent import DEFAULT_MODEL, make_graph

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

TEST_QUERY = (
    "Compare the mechanisms of Phase 3 oncology trials involving "
    "brentuximab and zongertinib."
)

# NVIDIA API Catalog — free tier. Override with --judge-model.
#
# NOT `meta/llama-3.1-nemotron-70b-instruct`: absent from the catalog
# (Nemotron lives under the `nvidia/` namespace now).
#
# NOT `nvidia/llama-3.3-nemotron-super-49b-v1.5` either: it works, but it is a
# *reasoning* model and spends most of its time emitting chain-of-thought
# before the JSON ragas asks for. Benchmarked on an identical extraction
# prompt: 27.1s vs 4.5s for the model below. On a real faithfulness job that
# overran the HTTP client's socket-read timeout and the metric came back NaN.
# nemotron-3-super is MoE (~12B active), so it is far faster despite the
# larger nominal size, and returns JSON as the first token.
JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"


# =============================================================================
# STEP A + B — run the agent and capture the trace
# =============================================================================
def format_context(trial: dict) -> str:
    """Serialise one retrieved trial exactly as the agent saw it.

    Faithfulness is only meaningful if `retrieved_contexts` is what the
    generator actually conditioned on -- not a prettier reconstruction. These
    are the same fields the tool returned into the agent's context window.
    """
    interventions = ", ".join(
        f"{iv.get('type')}: {iv.get('name')}" for iv in trial.get("interventions") or []
    )
    return (
        f"NCTId: {trial.get('NCTId')}\n"
        f"BriefTitle: {trial.get('BriefTitle')}\n"
        f"Phase: {', '.join(trial.get('Phase') or [])}\n"
        f"OverallStatus: {trial.get('OverallStatus')}\n"
        f"LeadSponsorName: {trial.get('LeadSponsorName')}\n"
        f"StudyType: {trial.get('studyType')}\n"
        f"Conditions: {', '.join(trial.get('conditions') or [])}\n"
        f"Interventions: {interventions}\n"
        f"BriefSummary: {trial.get('BriefSummary')}"
    )


def run_agent(query: str, model: str) -> tuple[str, list[str], str, dict]:
    """Execute the LangGraph workflow and pull out the eval triple."""
    print(f"[A] running agent (generator={model})")
    print(f"[A] query: {query}")
    graph = make_graph(model, verbose=False)

    started = time.perf_counter()
    final = graph.invoke(
        {"messages": [HumanMessage(content=query)], "tool_rounds": 0, "result": None},
        config={"recursion_limit": 25},
    )
    elapsed = time.perf_counter() - started

    result = final.get("result")
    if result is None:
        raise SystemExit("[A] agent produced no SmartTableResponse — aborting")

    # --- contexts: every distinct trial the Qdrant tool node returned ------
    contexts: dict[str, str] = {}
    tool_calls = 0
    for msg in final["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        tool_calls += 1
        try:
            payload = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        for trial in payload.get("trials", []):
            nct = trial.get("NCTId")
            if nct and nct not in contexts:  # dedupe across repeated searches
                contexts[nct] = format_context(trial)

    meta = {
        "elapsed_s": round(elapsed, 1),
        "tool_calls": tool_calls,
        "unique_contexts": len(contexts),
        "table_rows": len(result.table_data),
    }
    print(f"[B] captured trace: {tool_calls} tool call(s), "
          f"{len(contexts)} unique contexts, {len(result.table_data)} table rows "
          f"({elapsed:.1f}s)")
    return query, list(contexts.values()), result.narrative_summary, meta


# =============================================================================
# STEP C — shape for Ragas
# =============================================================================
def build_dataset(question: str, contexts: list[str], answer: str) -> Dataset:
    """Ragas 0.4 expects user_input / retrieved_contexts / response.

    The older question/answer/contexts trio belongs to ragas 0.1.x and fails
    column validation on this version.
    """
    ds = Dataset.from_dict(
        {
            "user_input": [question],
            "retrieved_contexts": [contexts],
            "response": [answer],
        }
    )
    print(f"[C] dataset columns: {ds.column_names} | rows: {ds.num_rows}")
    return ds


# =============================================================================
# STEP D — judge
# =============================================================================
def build_judge(judge_model: str, embed_model: str, max_tokens: int = 16384):
    # ChatNVIDIA (langchain-nvidia-ai-endpoints) goes over aiohttp, whose
    # per-socket read timeout is not exposed through the constructor we call
    # here. Diagnosed directly: a single Faithfulness call reliably died with
    # aiohttp.SocketTimeoutError after ~440s regardless of the ragas-level
    # timeout passed to RunConfig, because that timeout wraps the whole job,
    # not the socket read. Using langchain-openai against NVIDIA's
    # OpenAI-compatible endpoint (the spec's documented fallback) goes over
    # httpx instead, whose `timeout=` we DO control -- verified faithfulness
    # complete in ~7 minutes with no error once on that path.
    from langchain_openai import ChatOpenAI
    # Embeddings stay on ChatNVIDIA's sibling client, not langchain-openai:
    # NVIDIA's asymmetric embedding model requires an `input_type` field
    # (query vs passage) that OpenAIEmbeddings has no way to send, and the
    # bare REST call 400s without it ("'input_type' parameter is required for
    # asymmetric models"). NVIDIAEmbeddings sets this correctly; only the LLM
    # side needed the httpx-based client for its configurable read timeout.
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise SystemExit(
            "[D] NVIDIA_API_KEY is not set.\n"
            "    Add it to .env — free key at https://build.nvidia.com"
        )

    # max_tokens=16384, not 8192: measured directly against this judge model,
    # decomposing a ~3.3k-char answer into atomic statements alone consumed
    # ~3,000 completion tokens before the per-statement verification pass even
    # starts. 8192 hit LLMDidNotFinishException; 16384 did not.
    judge = ChatOpenAI(
        model=judge_model,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=key,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=900.0,
        max_retries=2,
    )
    embeddings = NVIDIAEmbeddings(model=embed_model, api_key=key)
    print(f"[D] judge: {judge_model}  |  embeddings: {embed_model}")
    return judge, embeddings


# =============================================================================
# STEP E — report
# =============================================================================
def extract_scores(result) -> dict[str, float]:
    """EvaluationResult exposes scores differently across ragas versions."""
    scores: dict[str, float] = {}
    try:
        df = result.to_pandas()
        for col in df.columns:
            if col in ("faithfulness", "answer_relevancy"):
                scores[col] = float(df[col].mean())
    except Exception:  # noqa: BLE001 - fall back to mapping access
        for name in ("faithfulness", "answer_relevancy"):
            try:
                scores[name] = float(result[name])
            except Exception:  # noqa: BLE001
                pass
    return scores


def print_report(question: str, contexts: list[str], answer: str,
                 scores: dict[str, float], meta: dict) -> None:
    line = "=" * 74
    print(f"\n{line}\nRAGAS EVALUATION\n{line}")
    print(f"  generator      : {DEFAULT_MODEL}")
    print(f"  judge          : {meta.get('judge')}")
    print(f"  agent runtime  : {meta['elapsed_s']}s "
          f"({meta['tool_calls']} tool calls, {meta['unique_contexts']} contexts)")
    print(f"\n  question       : {textwrap.shorten(question, 60, placeholder='…')}")
    print(f"  answer length  : {len(answer):,} chars")

    print(f"\n{'-' * 74}")
    print(f"  {'METRIC':<22}{'SCORE':>9}   INTERPRETATION")
    print(f"{'-' * 74}")
    blurbs = {
        "faithfulness": "claims entailed by the retrieved contexts",
        "answer_relevancy": "answer addresses the question asked",
    }
    for name in ("faithfulness", "answer_relevancy"):
        v = scores.get(name)
        # NaN means the judge call failed (network/TLS/parse), NOT a score of
        # zero. Reporting it as 0.0 would be a silently wrong evaluation.
        if v is None or math.isnan(v):
            print(f"  {name:<22}{'FAILED':>9}   judge returned no score — see errors above")
            continue
        filled = int(round(v * 20))
        bar = "█" * filled + "░" * (20 - filled)
        print(f"  {name:<22}{v:>9.4f}   {blurbs[name]}")
        print(f"  {'':<22}{bar}")
    print(f"{'-' * 74}")

    if scores.get("faithfulness", 0) >= 0.99:
        print("\n  faithfulness == 1.0 — every claim in the narrative is entailed by")
        print("  the retrieved records. This is what the strict-grounding prompt is for.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=TEST_QUERY)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="generator (agent) model")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--judge-timeout", type=int, default=900,
                        help="per-metric judge timeout in seconds")
    parser.add_argument("--judge-max-tokens", type=int, default=16384,
                        help="raise if faithfulness returns FAILED with "
                             "LLMDidNotFinishException")
    parser.add_argument("--dump", metavar="PATH", help="write the captured trace to JSON")
    parser.add_argument("--from-trace", metavar="PATH",
                        help="re-score a saved trace instead of re-running the "
                             "agent (no generator cost; use when iterating on "
                             "the judge or the metrics)")
    args = parser.parse_args()

    print("=" * 74)
    print("medical-rag :: Ragas evaluation pipeline")
    print("=" * 74)

    # A + B
    if args.from_trace:
        trace = json.loads(Path(args.from_trace).read_text(encoding="utf-8"))
        question = trace["user_input"]
        contexts = trace["retrieved_contexts"]
        answer = trace["response"]
        meta = {"elapsed_s": 0.0, "tool_calls": "—",
                "unique_contexts": len(contexts), "table_rows": "—"}
        print(f"[A] replaying saved trace: {args.from_trace}")
        print(f"[B] {len(contexts)} contexts, {len(answer):,}-char answer")
    else:
        question, contexts, answer, meta = run_agent(args.query, args.model)
    if not contexts:
        raise SystemExit("[B] no contexts retrieved — faithfulness would be undefined")

    if args.dump:
        Path(args.dump).write_text(
            json.dumps({"user_input": question, "retrieved_contexts": contexts,
                        "response": answer}, indent=2),
            encoding="utf-8",
        )
        print(f"[B] trace written to {args.dump}")

    # C
    dataset = build_dataset(question, contexts, answer)

    # D
    judge, embeddings = build_judge(args.judge_model, args.embed_model,
                                    args.judge_max_tokens)
    meta["judge"] = args.judge_model

    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness
    from ragas.run_config import RunConfig

    # ragas defaults to a 180s per-job timeout. Faithfulness is several
    # sequential judge calls (decompose the answer, then verify each
    # statement against every context); on a 49B judge over 11 contexts
    # that overruns the default and surfaces as a bare TimeoutError.
    # timeout wraps the ENTIRE metric computation as one asyncio.wait_for --
    # for faithfulness that means statement generation plus every per-context
    # verification call, sequentially. Confirmed directly: an isolated,
    # unbounded single_turn_ascore() on this trace took 420.5s; a longer
    # answer with more statements can run past 900s. max_retries is lowered
    # from ragas' default of 10 -- with a slow judge, a handful of retried
    # attempts can itself exceed the wait_for budget before any one attempt
    # finishes, which is a worse failure mode than one unhurried attempt.
    run_config = RunConfig(timeout=args.judge_timeout, max_retries=2, max_workers=4)

    print("[D] scoring… (the judge makes several calls per metric)")
    started = time.perf_counter()
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
    )
    print(f"[D] scored in {time.perf_counter() - started:.1f}s")

    # E
    scores = extract_scores(result)
    print_report(question, contexts, answer, scores, meta)
    return 0 if scores else 1


if __name__ == "__main__":
    raise SystemExit(main())
