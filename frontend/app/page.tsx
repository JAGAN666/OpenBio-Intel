"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, Database, Loader2, Search, Sparkles } from "lucide-react";

import BriefingCard from "@/components/BriefingCard";
import ExportButtons from "@/components/ExportButtons";
import ProgressPanel, { type StreamProgress, type StreamStatus } from "@/components/ProgressPanel";
import TrialsTable from "@/components/TrialsTable";
import { runJob } from "@/lib/runJob";
import type { SmartTableResponse } from "@/types/trial";

// 127.0.0.1, not localhost, on purpose: `localhost` resolves to IPv6 (::1)
// first on macOS, and this machine already has an unrelated
// `python -m http.server 8000` bound to the IPv6 wildcard. Uvicorn binds IPv4,
// so `localhost:8000` would reach the wrong server and 404. Pinning IPv4
// removes the ambiguity. Override with NEXT_PUBLIC_API_URL for deployments.
const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

const EXAMPLES = [
  "Compare the mechanisms and sponsors of Phase 3 oncology trials",
  "Which trials use pembrolizumab, and what are they combining it with?",
  "Build a comparative table of Phase 3 lung cancer trials",
];

export default function DashboardPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<SmartTableResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [timeline, setTimeline] = useState<StreamStatus[]>([]);
  const [mapProgress, setMapProgress] = useState<StreamProgress | null>(null);

  // The agent runs multiple LLM turns and can take a minute or more; a ticking
  // counter is the difference between "working" and "hung" for the user.
  // The counter is reset in runAnalysis rather than here — calling setState
  // directly inside an effect triggers react-hooks/set-state-in-effect.
  useEffect(() => {
    if (!loading) return;
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [loading]);

  async function runAnalysis(q: string) {
    const trimmed = q.trim();
    if (trimmed.length < 3 || loading) return;

    setElapsed(0);
    setLoading(true);
    setError(null);
    setData(null);
    setTimeline([]);
    setMapProgress(null);

    try {
      // Job-based execution (see lib/runJob.ts): the query survives ALB
      // hiccups, deploys, and even this tab closing -- the worker keeps
      // running server-side, and EventSource's auto-reconnect + the
      // server's full event replay recover the exact progress state.
      const result = await runJob<SmartTableResponse>(API_URL, "research", trimmed, {
        onReplayReset: () => {
          setTimeline([]);
          setMapProgress(null);
        },
        onStatus: (d) => setTimeline((prev) => [...prev, d as StreamStatus]),
        onProgress: (d) => setMapProgress(d as StreamProgress),
      });
      setData(result);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? `Could not reach the agent at ${API_URL}. Is uvicorn running?`
          : err instanceof Error
            ? err.message
            : String(err),
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="flex-1">
      <div className="mx-auto max-w-7xl px-6 py-10">
        {/* --- header ---------------------------------------------------- */}
        <header className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900 dark:text-slate-100">
            Smart Table
          </h1>
          <p className="mt-1.5 max-w-3xl text-sm text-slate-500 dark:text-slate-400">
            Ask an analyst question. The agent reasons over a Qdrant corpus of
            clinical trials and returns a cited briefing plus a comparative grid —
            grounded strictly in the retrieved records.
          </p>
        </header>

        {/* --- search bar ------------------------------------------------ */}
        <section className="mb-8 rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void runAnalysis(query);
            }}
            className="flex flex-col gap-3 sm:flex-row"
          >
            <div className="relative flex-1">
              <Search
                className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
                aria-hidden
              />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={loading}
                placeholder="Compare the mechanisms and sponsors of Phase 3 oncology trials"
                aria-label="Analyst question"
                className="w-full rounded-lg border border-slate-300 bg-white py-3 pl-10 pr-4 text-sm text-slate-900 placeholder:text-slate-400 focus:border-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-100 disabled:bg-slate-50 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:ring-sky-900/50 dark:disabled:bg-slate-900"
              />
            </div>
            <button
              type="submit"
              disabled={loading || query.trim().length < 3}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-sky-700 px-6 py-3 text-sm font-semibold text-white transition-colors hover:bg-sky-800 disabled:cursor-not-allowed disabled:bg-slate-300 dark:bg-sky-600 dark:hover:bg-sky-500 dark:disabled:bg-slate-800 dark:disabled:text-slate-500"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                  Running…
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" aria-hidden />
                  Run Analysis
                </>
              )}
            </button>
          </form>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400 dark:text-slate-500">Try:</span>
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                disabled={loading}
                onClick={() => {
                  setQuery(ex);
                  void runAnalysis(ex);
                }}
                className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-600 transition-colors hover:border-sky-300 hover:text-sky-700 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:border-sky-700 dark:hover:text-sky-400"
              >
                {ex}
              </button>
            ))}
          </div>
        </section>

        {/* --- loading (live SSE progress) --------------------------------- */}
        {loading && (
          <ProgressPanel timeline={timeline} mapProgress={mapProgress} elapsed={elapsed} />
        )}

        {/* --- error ----------------------------------------------------- */}
        {error && !loading && (
          <section className="mb-8 flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-6 py-5 dark:border-red-900/60 dark:bg-red-950/40">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-600 dark:text-red-400" aria-hidden />
            <div>
              <p className="text-sm font-semibold text-red-900 dark:text-red-300">Analysis failed</p>
              <p className="mt-1 text-sm text-red-700 dark:text-red-400">{error}</p>
            </div>
          </section>
        )}

        {/* --- empty state ----------------------------------------------- */}
        {!loading && !error && !data && (
          <section className="rounded-xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center dark:border-slate-700 dark:bg-slate-900">
            <Database className="mx-auto h-6 w-6 text-slate-300 dark:text-slate-600" aria-hidden />
            <p className="mt-3 text-sm font-medium text-slate-500 dark:text-slate-400">
              No analysis yet
            </p>
            <p className="mt-1 text-xs text-slate-400 dark:text-slate-500">
              Enter a question above, or pick one of the examples, to query the agent.
            </p>
          </section>
        )}

        {/* --- results --------------------------------------------------- */}
        {data && !loading && (
          <>
            <div className="mb-8">
              <BriefingCard summary={data.narrative_summary} />
            </div>
            <ExportButtons apiUrl={API_URL} data={data} />
            <TrialsTable data={data.table_data} />
            <footer className="mt-8 text-xs text-slate-400 dark:text-slate-600">
              Rows flagged <span className="font-medium">Design Details Only</span> still
              carry trial design context — the badge reflects whether the source record
              named a mechanism, not whether data exists.
            </footer>
          </>
        )}
      </div>
    </main>
  );
}
