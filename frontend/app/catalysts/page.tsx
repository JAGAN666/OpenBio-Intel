"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, CalendarClock, Loader2, Search, Sparkles } from "lucide-react";

import CatalystTracker from "@/components/CatalystTracker";
import { runJob } from "@/lib/runJob";
import type { CatalystTimeline } from "@/types/catalysts";

// Same IPv4-pinning reasoning as app/page.tsx -- see that file's comment.
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

const EXAMPLES = [
  "Upcoming Phase 3 readouts in Oncology",
  "Upcoming PDUFA dates",
  "Upcoming Phase 3 readouts in Rheumatoid Arthritis",
];

export default function CatalystsPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<CatalystTimeline | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  // The catalyst graph does a broad two-collection retrieval, a live
  // ClinicalTrials.gov date-enrichment call, PLUS one structured-output LLM
  // call -- no interim progress to report the way /api/research's SSE
  // stream does, so a ticking counter is what tells the analyst "still
  // working," not "hung."
  useEffect(() => {
    if (!loading) return;
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [loading]);

  async function runCatalystSearch(q: string) {
    const trimmed = q.trim();
    if (trimmed.length < 3 || loading) return;

    setElapsed(0);
    setLoading(true);
    setError(null);
    setData(null);

    try {
      // Job-based (lib/runJob.ts) -- same reliability story as the other
      // two pages: server-side execution, reconnect-safe streaming.
      setData(await runJob<CatalystTimeline>(API_URL, "catalysts", trimmed));
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
            Clinical Catalyst &amp; Readout Tracker
          </h1>
          <p className="mt-1.5 max-w-3xl text-sm text-slate-500 dark:text-slate-400">
            Enter a therapeutic area or event type. The agent retrieves upcoming
            clinical trials — enriched with real, live-looked-up completion
            dates — and SEC filings for stated PDUFA dates, then builds a
            chronological timeline grounded strictly in the retrieved evidence.
          </p>
        </header>

        {/* --- search bar ------------------------------------------------ */}
        <section className="mb-8 rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void runCatalystSearch(query);
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
                placeholder="Upcoming Phase 3 readouts in Oncology"
                aria-label="Catalyst search query"
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
                  Building…
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" aria-hidden />
                  Build Timeline
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
                  void runCatalystSearch(ex);
                }}
                className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-600 transition-colors hover:border-sky-300 hover:text-sky-700 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:border-sky-700 dark:hover:text-sky-400"
              >
                {ex}
              </button>
            ))}
          </div>
        </section>

        {/* --- loading ----------------------------------------------------- */}
        {loading && (
          <section className="mb-8 flex flex-col items-center gap-3 rounded-xl border border-slate-200 bg-white px-6 py-16 text-center shadow-sm dark:border-slate-800 dark:bg-slate-900">
            <Loader2 className="h-6 w-6 animate-spin text-sky-600 dark:text-sky-400" aria-hidden />
            <p className="text-sm font-medium text-slate-600 dark:text-slate-300">
              Retrieving trials and SEC filings, looking up real completion
              dates, and synthesizing the timeline…
            </p>
            <p className="text-xs text-slate-400 dark:text-slate-500">{elapsed}s elapsed</p>
          </section>
        )}

        {/* --- error ----------------------------------------------------- */}
        {error && !loading && (
          <section className="mb-8 flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-6 py-5 dark:border-red-900/60 dark:bg-red-950/40">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-600 dark:text-red-400" aria-hidden />
            <div>
              <p className="text-sm font-semibold text-red-900 dark:text-red-300">Timeline build failed</p>
              <p className="mt-1 text-sm text-red-700 dark:text-red-400">{error}</p>
            </div>
          </section>
        )}

        {/* --- empty state ----------------------------------------------- */}
        {!loading && !error && !data && (
          <section className="rounded-xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center dark:border-slate-700 dark:bg-slate-900">
            <CalendarClock className="mx-auto h-6 w-6 text-slate-300 dark:text-slate-600" aria-hidden />
            <p className="mt-3 text-sm font-medium text-slate-500 dark:text-slate-400">No timeline yet</p>
            <p className="mt-1 text-xs text-slate-400 dark:text-slate-500">
              Enter a therapeutic area or event type above, or pick one of the
              examples, to build a catalyst timeline.
            </p>
          </section>
        )}

        {/* --- results --------------------------------------------------- */}
        {data && !loading && <CatalystTracker data={data} />}
      </div>
    </main>
  );
}
