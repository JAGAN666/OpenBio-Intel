"use client";

import {
  type ColumnDef,
  createColumnHelper,
  createSortedRowModel,
  flexRender,
  rowSortingFeature,
  sortFn_alphanumeric,
  sortFn_text,
  tableFeatures,
  useTable,
} from "@tanstack/react-table";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  CircleDashed,
  ExternalLink,
  Table2,
  Target,
} from "lucide-react";

import type { SourceCitation, TrialRow } from "@/types/trial";

const CTGOV = "https://clinicaltrials.gov/study/";

// TanStack Table v9 composes features explicitly — there is no
// `useReactTable` / `getCoreRowModel()` any more. Defined at module scope, as
// the docs recommend, so it is not rebuilt on every render.
const features = tableFeatures({
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
  sortFns: { alphanumeric: sortFn_alphanumeric, text: sortFn_text },
});

const columnHelper = createColumnHelper<typeof features, TrialRow>();

/** Intervention name -> pill. */
function Pill({ label }: { label: string }) {
  return (
    <span className="inline-flex max-w-[15rem] items-center truncate rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs font-medium text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300">
      {label}
    </span>
  );
}

// Short display labels per citation source_type -- the full reference is in
// the link's title tooltip, so the chip itself stays scannable.
const SOURCE_LABELS: Record<string, string> = {
  registry: "CT.gov",
  pdf_literature: "Poster/PDF",
  fda: "FDA",
  pubmed: "PubMed",
  sec: "SEC",
  news: "News",
};

/**
 * One provenance chip per cited document. Linked when the citation carries a
 * URL; a plain chip otherwise (the reference text still shows on hover).
 * This is the auditability contract: an analyst should never see a value
 * they cannot click through to a primary source.
 */
function SourceChips({ sources }: { sources?: SourceCitation[] }) {
  if (!sources || sources.length === 0) {
    return <span className="text-xs text-slate-400 dark:text-slate-600">—</span>;
  }
  return (
    <div className="flex flex-wrap gap-1">
      {sources.map((s, i) => {
        const label = SOURCE_LABELS[s.source_type] ?? s.source_type;
        const chip = (
          <span className="inline-flex items-center gap-1 rounded-md border border-sky-200 bg-sky-50 px-1.5 py-0.5 text-[0.65rem] font-semibold text-sky-700 dark:border-sky-900/60 dark:bg-sky-950/40 dark:text-sky-400">
            {label}
          </span>
        );
        return s.url ? (
          <a
            key={`${s.source_type}-${s.reference}-${i}`}
            href={s.url}
            target="_blank"
            rel="noopener noreferrer"
            title={s.reference}
            className="transition-opacity hover:opacity-70"
          >
            {chip}
          </a>
        ) : (
          <span key={`${s.source_type}-${s.reference}-${i}`} title={s.reference}>
            {chip}
          </span>
        );
      })}
    </div>
  );
}

/**
 * The Badge Rule: branch on the boolean, never on the prose.
 * `false` does NOT mean "no data" — the text below still carries trial design
 * detail, so the badge is subdued rather than an error state.
 */
function MechanismBadge({ described }: { described: boolean }) {
  return described ? (
    <span className="inline-flex items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[0.7rem] font-semibold uppercase tracking-wide text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-400">
      <Target className="h-3 w-3" aria-hidden />
      Target Identified
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-slate-100 px-2 py-0.5 text-[0.7rem] font-semibold uppercase tracking-wide text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400">
      <CircleDashed className="h-3 w-3" aria-hidden />
      Design Details Only
    </span>
  );
}

// Explicitly typed: each accessor yields a different TValue (string,
// string[], …), so the heterogeneous array will not infer to a single
// ColumnDef union on its own.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const columns: ColumnDef<typeof features, TrialRow, any>[] = [
  columnHelper.accessor("nct_id", {
    header: "NCT ID",
    cell: (info) => {
      const id = info.getValue();
      return (
        <a
          href={`${CTGOV}${id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="group inline-flex items-center gap-1 font-mono text-sm text-sky-700 hover:text-sky-900 dark:text-sky-400 dark:hover:text-sky-300"
        >
          {id}
          <ExternalLink
            className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-100"
            aria-hidden
          />
        </a>
      );
    },
  }),
  columnHelper.accessor("phase", {
    header: "Phase",
    cell: (info) => (
      <span className="whitespace-nowrap text-sm text-slate-700 dark:text-slate-300">{info.getValue()}</span>
    ),
  }),
  columnHelper.accessor("sponsor", {
    header: "Sponsor",
    cell: (info) => (
      <span className="text-sm text-slate-700 dark:text-slate-300">{info.getValue()}</span>
    ),
  }),
  columnHelper.accessor("interventions", {
    header: "Interventions",
    enableSorting: false,
    cell: (info) => {
      // Deduped, not just re-keyed -- the extraction LLM occasionally
      // repeats the same intervention name twice in one trial's array
      // (e.g. "Intensity-Modulated Radiation Therapy" appearing twice),
      // which both collided React's key={name} and rendered the same pill
      // twice for no reason.
      const items = Array.from(new Set(info.row.original.interventions));
      return (
        <div className="flex flex-wrap gap-1">
          {items.length === 0 ? (
            <span className="text-xs text-slate-400 dark:text-slate-600">—</span>
          ) : (
            items.map((name: string) => <Pill key={name} label={name} />)
          )}
        </div>
      );
    },
  }),
  columnHelper.accessor("mechanism_or_findings", {
    header: "Mechanism / Findings",
    enableSorting: false,
    cell: (info) => (
      <div className="min-w-[22rem] space-y-1.5">
        <MechanismBadge described={info.row.original.mechanism_described} />
        <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">{info.getValue()}</p>
      </div>
    ),
  }),
  columnHelper.accessor("sources", {
    header: "Sources",
    enableSorting: false,
    cell: (info) => <SourceChips sources={info.row.original.sources} />,
  }),
];

export default function TrialsTable({ data }: { data: TrialRow[] }) {
  const table = useTable({ features, columns, data });

  const described = data.filter((r) => r.mechanism_described).length;

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <header className="flex flex-wrap items-center gap-2 border-b border-slate-200 px-6 py-4 dark:border-slate-800">
        <Table2 className="h-4 w-4 text-slate-500 dark:text-slate-400" aria-hidden />
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300">
          Comparative Trials Grid
        </h2>
        <span className="ml-auto text-xs text-slate-500 dark:text-slate-400">
          {data.length} trials · {described} with target identified ·{" "}
          {data.length - described} design details only
        </span>
      </header>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left">
          <thead className="bg-slate-50 dark:bg-slate-950/50">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  const sortable = header.column.getCanSort();
                  const sorted = header.column.getIsSorted();
                  return (
                    <th
                      key={header.id}
                      scope="col"
                      className="border-b border-slate-200 px-4 py-3 align-bottom text-xs font-semibold uppercase tracking-wide text-slate-500 dark:border-slate-800 dark:text-slate-400"
                    >
                      {sortable ? (
                        <button
                          type="button"
                          onClick={header.column.getToggleSortingHandler()}
                          className="inline-flex items-center gap-1 hover:text-slate-800 dark:hover:text-slate-200"
                        >
                          {flexRender(
                            header.column.columnDef.header,
                            header.getContext(),
                          )}
                          {sorted === "asc" ? (
                            <ArrowUp className="h-3 w-3" aria-hidden />
                          ) : sorted === "desc" ? (
                            <ArrowDown className="h-3 w-3" aria-hidden />
                          ) : (
                            <ArrowUpDown className="h-3 w-3 opacity-40" aria-hidden />
                          )}
                        </button>
                      ) : (
                        flexRender(header.column.columnDef.header, header.getContext())
                      )}
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>

          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/60 dark:border-slate-800 dark:hover:bg-slate-800/40">
                {/* getAllCells, not getVisibleCells — the latter is provided
                    by columnVisibilityFeature, which this table does not use. */}
                {row.getAllCells().map((cell) => (
                  <td key={cell.id} className="px-4 py-4 align-top">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
