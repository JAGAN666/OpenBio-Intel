"""OpenBio-Intel MCP server -- the platform's tools for any MCP client.

Exposes the same federated intelligence tools the in-app agent uses
(clinical trials hybrid search, knowledge graph, FDA approvals + rejection
letters, PubMed, SEC filings, corporate news, FAERS safety signals,
patent-cliff/LOE lookup) plus the full Smart Table pipeline, over stdio --
so Claude Desktop, Claude Code, or any MCP-capable agent can query a
seeded OpenBio-Intel stack directly.

Positioning vs BioMCP and similar API-wrapper servers: those give agents
RAW API access per call; this serves a PRE-BUILT hybrid-indexed corpus
(600K+ trials with BM25+dense fusion and reranking), an RxNorm knowledge
graph, and synthesized comparison tables.

Setup (Claude Desktop / Claude Code):
    {
      "mcpServers": {
        "openbio-intel": {
          "command": "uv",
          "args": ["run", "--project", "/path/to/OpenBio-Intel", "python", "mcp_server.py"]
        }
      }
    }
Requires the same environment as the backend (.env with OPENAI_API_KEY +
reachable Qdrant/Neo4j -- `docker compose up -d qdrant neo4j` and a seeded
corpus). run_smart_table additionally needs the orchestration provider's
key (ANTHROPIC_API_KEY or KIMI_API_KEY per LLM_PROVIDER).
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from fastmcp import FastMCP  # noqa: E402

mcp = FastMCP(
    "openbio-intel",
    instructions=(
        "Biopharma clinical market intelligence over a locally-hosted "
        "OpenBio-Intel stack: hybrid-indexed ClinicalTrials.gov corpus, "
        "RxNorm knowledge graph, FDA approvals AND rejection letters, "
        "PubMed, SEC filings, corporate news, live FAERS safety signals, "
        "and Orange/Purple Book exclusivity data. Tool results are JSON "
        "strings with a has_results flag; treat has_results=false as 'the "
        "corpus genuinely has nothing', not an error."
    ),
)

# LangChain tool objects -> MCP tools. Imported lazily-ish (research_agent
# is heavy) but at module scope: an MCP server exists to serve these, so
# failing fast at startup beats failing on first call.
import research_agent as ra  # noqa: E402


def _lc(tool, /, **kwargs) -> str:
    """Invoke a LangChain @tool and return its JSON-string observation."""
    return tool.invoke(kwargs)


@mcp.tool()
def search_clinical_trials(query: str, phase_filter: str | None = None) -> str:
    """Hybrid (BM25+dense+rerank) search over the 600K+ trial ClinicalTrials.gov
    corpus. phase_filter accepts 'Phase 2' or 'Phase 3' as a hard filter."""
    return _lc(ra.search_clinical_trials, query=query, phase_filter=phase_filter)


@mcp.tool()
def query_knowledge_graph(entity: str) -> str:
    """Exact drug-entity traversal in the Neo4j RxNorm knowledge graph:
    resolves brand/generic names and returns every linked trial. Falls back
    to vector search automatically when the graph has no match."""
    return _lc(ra.query_knowledge_graph, entity=entity)


@mcp.tool()
def search_fda_records(query: str) -> str:
    """FDA drug approval/application records (openFDA drugsfda): approval
    status, application numbers, approved products and ingredients."""
    return _lc(ra.search_fda_records, query=query)


@mcp.tool()
def search_fda_crls(query: str) -> str:
    """FDA Complete Response Letters (rejection letters, full text,
    2025+): why FDA refused applications -- CMC, trial design, safety."""
    return _lc(ra.search_fda_crls, query=query)


@mcp.tool()
def search_pubmed_literature(query: str) -> str:
    """Open-access PubMed Central full-text literature: mechanisms,
    published efficacy, pathway discussion."""
    return _lc(ra.search_pubmed_literature, query=query)


@mcp.tool()
def search_sec_filings(query: str) -> str:
    """SEC 10-K/8-K excerpts isolated to pipeline/clinical/R&D sections:
    corporate strategy and formally disclosed timelines."""
    return _lc(ra.search_sec_filings, query=query)


@mcp.tool()
def search_corporate_news(query: str) -> str:
    """Press releases + earnings-call transcripts: the most real-time
    layer, where PDUFA dates and interim updates surface first."""
    return _lc(ra.search_corporate_news, query=query)


@mcp.tool()
def get_safety_signals(drug_name: str, reaction: str | None = None) -> str:
    """LIVE FAERS post-market safety screen: reporting odds ratios for a
    drug's most-reported adverse events (association, not causation)."""
    return _lc(ra.get_safety_signals, drug_name=drug_name, reaction=reaction)


@mcp.tool()
def get_exclusivity(drug_name: str) -> str:
    """Patent-cliff / loss-of-exclusivity lookup from FDA Orange + Purple
    Book data: listed patent and exclusivity expiry dates per product."""
    return _lc(ra.get_exclusivity, drug_name=drug_name)


_graph = None


@mcp.tool()
def run_smart_table(question: str) -> str:
    """Run the FULL Smart Table pipeline: multi-tool agent retrieval ->
    parallel per-trial structured extraction -> cited comparison table +
    narrative. Slower (tens of seconds to minutes) and costs LLM calls;
    use the individual search tools for quick lookups."""
    global _graph
    if _graph is None:
        _graph = ra.make_graph(ra.DEFAULT_MODEL, verbose=False)
    from langchain_core.messages import HumanMessage
    final = _graph.invoke(
        {"messages": [HumanMessage(content=question)], "tenant_id": None,
         "tool_rounds": 0, "result": None, "is_in_domain": None,
         "has_results": None, "synthesis_retries": 0, "synthesis_error": None,
         "retrieved_trials": [], "extracted_rows": [], "retrieved_literature": [],
         "retrieved_fda": [], "retrieved_crls": [], "retrieved_safety": [],
         "retrieved_exclusivity": []},
        config={"recursion_limit": 25},
    )
    result = final.get("result")
    if result is None:
        return '{"error": "pipeline produced no result", "has_results": false}'
    return result.model_dump_json(indent=2)


if __name__ == "__main__":
    mcp.run()
