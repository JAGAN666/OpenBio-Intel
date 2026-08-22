"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, Layers3, Loader2, Search, Sparkles } from "lucide-react";

import IndicationMatrix from "@/components/IndicationMatrix";
import { runJob } from "@/lib/runJob";
import type { LandscapeMatrix } from "@/types/landscape";

// Same IPv4-pinning reasoning as app/page.tsx -- see that file's comment.
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

const EXAMPLES = [
  "Non-Small Cell Lung Cancer",
  "Multiple Myeloma",
  "Rheumatoid Arthritis",
];

export default function LandscapePage() {
  const [area, setArea] = useState("");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<LandscapeMatrix | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  // The landscape graph does a broad three-collection retrieval PLUS one
  // large structured-output LLM call over all of it at once (no per-trial
  // parallel workers to report interim progress the way /api/research's SSE
  // stream does) -- a single request can genuinely run a minute or more, so
  // a ticking counter is what tells the analyst "still working," not "hung."
  useEffect(() => {
    if (!loading) return;
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [loading]);

  async function runLandscape(therapeuticArea: string) {
    const trimmed = therapeuticArea.trim();
    if (trimmed.length < 3 || loading) return;

    setElapsed(0);
    setLoading(true);
    setError(null);
    setData(null);

    try {
      // Job-based (lib/runJob.ts): the matrix build keeps running server-
      // side through deploys/disconnects, and reconnects recover progress.
      setData(await runJob<LandscapeMatrix>(API_URL, "landscape", trimmed));
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
            Indication Landscape
          </h1>
          <p className="mt-1.5 max-w-3xl text-sm text-slate-500 dark:text-slate-400">
            Enter a therapeutic area. The agent retrieves across clinical trials,
            FDA approval records, and PubMed literature, then builds a competitive
            matrix of mechanism/target against development phase — grounded
            strictly in the retrieved evidence.
          </p>
        </header>

        {/* --- search bar ------------------------------------------------ */}
        <section className="mb-8 rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void runLandscape(area);
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
                value={area}
                onChange={(e) => setArea(e.target.value)}
                disabled={loading}
                placeholder="Non-Small Cell Lung Cancer"
                aria-label="Therapeutic area"
                className="w-full rounded-lg border border-slate-300 bg-white py-3 pl-10 pr-4 text-sm text-slate-900 placeholder:text-slate-400 focus:border-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-100 disabled:bg-slate-50 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:ring-sky-900/50 dark:disabled:bg-slate-900"
              />
            </div>
            <button
              type="submit"
              disabled={loading || area.trim().length < 3}
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
                  Build Landscape
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
                  setArea(ex);
                  void runLandscape(ex);
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
              Retrieving trials, FDA records, and literature — then synthesizing
              the matrix…
            </p>
            <p className="text-xs text-slate-400 dark:text-slate-500">
              {elapsed}s elapsed — a broad indication can take a minute or more.
            </p>
          </section>
        )}

        {/* --- error ----------------------------------------------------- */}
        {error && !loading && (
          <section className="mb-8 flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-6 py-5 dark:border-red-900/60 dark:bg-red-950/40">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-red-600 dark:text-red-400" aria-hidden />
            <div>
              <p className="text-sm font-semibold text-red-900 dark:text-red-300">Landscape build failed</p>
              <p className="mt-1 text-sm text-red-700 dark:text-red-400">{error}</p>
            </div>
          </section>
        )}

        {/* --- empty state ----------------------------------------------- */}
        {!loading && !error && !data && (
          <section className="rounded-xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center dark:border-slate-700 dark:bg-slate-900">
            <Layers3 className="mx-auto h-6 w-6 text-slate-300 dark:text-slate-600" aria-hidden />
            <p className="mt-3 text-sm font-medium text-slate-500 dark:text-slate-400">No landscape yet</p>
            <p className="mt-1 text-xs text-slate-400 dark:text-slate-500">
              Enter a therapeutic area above, or pick one of the examples, to
              build a competitive matrix.
            </p>
          </section>
        )}

        {/* --- results --------------------------------------------------- */}
        {data && !loading && <IndicationMatrix data={data} />}
      </div>
    </main>
  );
}
