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

import type { TrialRow } from "@/types/trial";

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
    <span className="inline-flex max-w-[15rem] items-center truncate rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs font-medium text-slate-700">
      {label}
    </span>
  );
}

/**
 * The Badge Rule: branch on the boolean, never on the prose.
 * `false` does NOT mean "no data" — the text below still carries trial design
 * detail, so the badge is subdued rather than an error state.
 */
function MechanismBadge({ described }: { described: boolean }) {
  return described ? (
    <span className="inline-flex items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[0.7rem] font-semibold uppercase tracking-wide text-emerald-700">
      <Target className="h-3 w-3" aria-hidden />
      Target Identified
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-slate-100 px-2 py-0.5 text-[0.7rem] font-semibold uppercase tracking-wide text-slate-500">
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
          className="group inline-flex items-center gap-1 font-mono text-sm text-sky-700 hover:text-sky-900"
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
      <span className="whitespace-nowrap text-sm text-slate-700">{info.getValue()}</span>
    ),
  }),
  columnHelper.accessor("sponsor", {
    header: "Sponsor",
    cell: (info) => (
      <span className="text-sm text-slate-700">{info.getValue()}</span>
    ),
  }),
  columnHelper.accessor("interventions", {
    header: "Interventions",
    enableSorting: false,
    cell: (info) => {
      const items = info.row.original.interventions;
      return (
        <div className="flex flex-wrap gap-1">
          {items.length === 0 ? (
            <span className="text-xs text-slate-400">—</span>
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
        <p className="text-sm leading-relaxed text-slate-600">{info.getValue()}</p>
      </div>
    ),
  }),
];

export default function TrialsTable({ data }: { data: TrialRow[] }) {
  const table = useTable({ features, columns, data });

  const described = data.filter((r) => r.mechanism_described).length;

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <header className="flex flex-wrap items-center gap-2 border-b border-slate-200 px-6 py-4">
        <Table2 className="h-4 w-4 text-slate-500" aria-hidden />
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-700">
          Comparative Trials Grid
        </h2>
        <span className="ml-auto text-xs text-slate-500">
          {data.length} trials · {described} with target identified ·{" "}
          {data.length - described} design details only
        </span>
      </header>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left">
          <thead className="bg-slate-50">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  const sortable = header.column.getCanSort();
                  const sorted = header.column.getIsSorted();
                  return (
                    <th
                      key={header.id}
                      scope="col"
                      className="border-b border-slate-200 px-4 py-3 align-bottom text-xs font-semibold uppercase tracking-wide text-slate-500"
                    >
                      {sortable ? (
                        <button
                          type="button"
                          onClick={header.column.getToggleSortingHandler()}
                          className="inline-flex items-center gap-1 hover:text-slate-800"
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
              <tr key={row.id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/60">
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
