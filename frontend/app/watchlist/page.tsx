"use client";

import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  Bell,
  ExternalLink,
  FileWarning,
  Loader2,
  Plus,
  RefreshCw,
  Trash2,
} from "lucide-react";

// Same IPv4-pinning reasoning as app/page.tsx -- see that file's comment.
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type Entry = { id: string; type: string; value: string; added_at: string };
type TrialChange = {
  nct_id: string; title: string; status: string; last_update: string; url: string;
};
type CrlChange = {
  application_numbers: string[]; company: string; letter_date: string; letter_type: string;
};
type DigestItem = {
  entry: Entry; trial_changes: TrialChange[]; new_crls: CrlChange[]; errors: string[];
};
type Digest = {
  checked_at: string | null; since?: string; entries_checked?: number; items: DigestItem[];
};

const TYPE_LABELS: Record<string, string> = {
  drug: "Drug", company: "Company", nct: "NCT ID", topic: "Topic",
};

export default function WatchlistPage() {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [digest, setDigest] = useState<Digest | null>(null);
  const [newType, setNewType] = useState("drug");
  const [newValue, setNewValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [w, d] = await Promise.all([
        fetch(`${API_URL}/api/watchlist`).then((r) => r.json()),
        fetch(`${API_URL}/api/watchlist/digest`).then((r) => r.json()),
      ]);
      setEntries(w.entries ?? []);
      setDigest(d);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    // False positive for this rule: refresh() only calls setState AFTER its
    // awaited fetches resolve, never synchronously during the effect --
    // this is the standard fetch-on-mount shape, not a cascading render.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  async function addEntry() {
    const value = newValue.trim();
    if (value.length < 2 || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/watchlist`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: newType, value }),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
      setNewValue("");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function removeEntry(id: string) {
    setBusy(true);
    try {
      await fetch(`${API_URL}/api/watchlist/${id}`, { method: "DELETE" });
      await refresh();
    } finally {
      setBusy(false);
    }
  }

  async function runCheck() {
    if (checking) return;
    setChecking(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/watchlist/check`, { method: "POST" });
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
      setDigest(await res.json());
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setChecking(false);
    }
  }

  return (
    <main className="mx-auto max-w-5xl space-y-6 px-6 py-8">
      <header className="flex items-center gap-3">
        <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-sky-700 text-white dark:bg-sky-600">
          <Bell className="h-5 w-5" aria-hidden />
        </span>
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-slate-100">
            Watchlist
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Watch drugs, companies, trials, or topics — the check scans
            ClinicalTrials.gov updates and new FDA Complete Response Letters
            since the last run.
          </p>
        </div>
        <button
          type="button"
          onClick={runCheck}
          disabled={checking || entries.length === 0}
          className="ml-auto inline-flex items-center gap-2 rounded-lg bg-sky-700 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-sky-800 disabled:opacity-40 dark:bg-sky-600 dark:hover:bg-sky-500"
        >
          {checking ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          ) : (
            <RefreshCw className="h-4 w-4" aria-hidden />
          )}
          Check now
        </button>
      </header>

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-400">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          {error}
        </div>
      )}

      {/* add form */}
      <section className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <select
          value={newType}
          onChange={(e) => setNewType(e.target.value)}
          className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300"
        >
          {Object.entries(TYPE_LABELS).map(([v, label]) => (
            <option key={v} value={v}>{label}</option>
          ))}
        </select>
        <input
          value={newValue}
          onChange={(e) => setNewValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && addEntry()}
          placeholder='e.g. "pembrolizumab", "Coherus BioSciences", "NCT06712888"'
          className="min-w-[16rem] flex-1 rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
        />
        <button
          type="button"
          onClick={addEntry}
          disabled={busy || newValue.trim().length < 2}
          className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 disabled:opacity-40 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
        >
          <Plus className="h-4 w-4" aria-hidden />
          Watch
        </button>
      </section>

      {/* entries */}
      <section className="rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <header className="border-b border-slate-200 px-5 py-3 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:border-slate-800 dark:text-slate-400">
          Watched entities ({entries.length})
        </header>
        {entries.length === 0 ? (
          <p className="px-5 py-6 text-sm text-slate-400 dark:text-slate-500">
            Nothing watched yet — add a drug, company, NCT id, or topic above.
          </p>
        ) : (
          <ul className="divide-y divide-slate-100 dark:divide-slate-800">
            {entries.map((e) => (
              <li key={e.id} className="flex items-center gap-3 px-5 py-3">
                <span className="rounded-md border border-slate-200 bg-slate-50 px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400">
                  {TYPE_LABELS[e.type] ?? e.type}
                </span>
                <span className="text-sm font-medium text-slate-800 dark:text-slate-200">
                  {e.value}
                </span>
                <button
                  type="button"
                  onClick={() => removeEntry(e.id)}
                  disabled={busy}
                  aria-label={`Stop watching ${e.value}`}
                  className="ml-auto rounded-md p-1.5 text-slate-400 transition-colors hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/40 dark:hover:text-red-400"
                >
                  <Trash2 className="h-4 w-4" aria-hidden />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* digest */}
      <section className="rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <header className="flex items-center gap-2 border-b border-slate-200 px-5 py-3 dark:border-slate-800">
          <span className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Latest changes
          </span>
          {digest?.checked_at && (
            <span className="ml-auto text-xs text-slate-400 dark:text-slate-500">
              checked {new Date(digest.checked_at).toLocaleString()}
            </span>
          )}
        </header>
        {!digest?.checked_at ? (
          <p className="px-5 py-6 text-sm text-slate-400 dark:text-slate-500">
            No check has run yet — press “Check now”.
          </p>
        ) : digest.items.length === 0 ? (
          <p className="px-5 py-6 text-sm text-slate-400 dark:text-slate-500">
            No changes detected since the previous check.
          </p>
        ) : (
          <div className="divide-y divide-slate-100 dark:divide-slate-800">
            {digest.items.map((item) => (
              <div key={item.entry.id} className="space-y-2 px-5 py-4">
                <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                  {item.entry.value}
                  <span className="ml-2 text-xs font-normal text-slate-400">
                    {item.trial_changes.length} trial update(s)
                    {item.new_crls.length > 0 && ` · ${item.new_crls.length} new CRL(s)`}
                  </span>
                </p>
                <ul className="space-y-1">
                  {item.trial_changes.slice(0, 8).map((t) => (
                    <li key={t.nct_id} className="flex items-baseline gap-2 text-sm">
                      <a
                        href={t.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex shrink-0 items-center gap-1 font-mono text-xs text-sky-700 hover:text-sky-900 dark:text-sky-400"
                      >
                        {t.nct_id}
                        <ExternalLink className="h-3 w-3" aria-hidden />
                      </a>
                      <span className="text-xs text-slate-400">{t.last_update}</span>
                      <span className="truncate text-slate-600 dark:text-slate-400">
                        {t.title}
                      </span>
                    </li>
                  ))}
                  {item.new_crls.map((c, i) => (
                    <li key={`crl-${i}`} className="flex items-center gap-2 text-sm text-red-700 dark:text-red-400">
                      <FileWarning className="h-4 w-4 shrink-0" aria-hidden />
                      New Complete Response Letter — {c.company} (
                      {(c.application_numbers ?? []).join(", ")}), {c.letter_date}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
