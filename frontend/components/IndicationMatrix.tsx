"use client";

import { Building2, Dna, Layers3 } from "lucide-react";

import { sourceUrl } from "@/lib/sourceUrl";
import type { DrugEntry, LandscapeMatrix, MechanismRow, PhaseCell } from "@/types/landscape";

const MECHANISM_COL_WIDTH = "14rem";
const PHASE_COL_WIDTH = "11rem";

function DrugCard({ drug }: { drug: DrugEntry }) {
  const url = sourceUrl(drug.source);
  const className =
    "block w-full rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-left shadow-sm transition-colors hover:border-sky-300 hover:bg-sky-50/60 focus:outline-none focus:ring-2 focus:ring-sky-200 dark:border-slate-700 dark:bg-slate-900 dark:hover:border-sky-700 dark:hover:bg-sky-950/40 dark:focus:ring-sky-900/50";

  const content = (
    <>
      <span className="block truncate text-xs font-semibold text-slate-800 dark:text-slate-200">{drug.name}</span>
      <span className="mt-0.5 flex items-center gap-1 truncate text-[0.65rem] text-slate-500 dark:text-slate-400">
        <Building2 className="h-2.5 w-2.5 shrink-0" aria-hidden />
        {drug.sponsor}
      </span>
    </>
  );

  return url ? (
    <a href={url} target="_blank" rel="noopener noreferrer" className={className} title={drug.source}>
      {content}
    </a>
  ) : (
    <div className={className} title={drug.source}>
      {content}
    </div>
  );
}

/**
 * `cell` is undefined when a row's `cells` array (LLM-generated, not
 * code-generated) is missing an entry for this phase entirely -- the AC2
 * "gracefully handles empty cells" requirement covers both an explicit
 * empty `drugs` list AND a wholesale-missing cell, so both render the same
 * subdued empty state rather than a layout gap or a crash.
 */
function PhaseCellView({ cell }: { cell: PhaseCell | undefined }) {
  const drugs = cell?.drugs ?? [];
  if (drugs.length === 0) {
    return (
      <div className="flex h-full min-h-[3.25rem] items-center justify-center text-xs text-slate-300 dark:text-slate-700">
        —
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-1.5 p-1.5">
      {drugs.map((drug, i) => (
        <DrugCard key={`${drug.source}-${drug.name}-${i}`} drug={drug} />
      ))}
    </div>
  );
}

function totalDrugs(row: MechanismRow): number {
  return row.cells.reduce((n, c) => n + (c.drugs?.length ?? 0), 0);
}

export default function IndicationMatrix({ data }: { data: LandscapeMatrix }) {
  if (data.rows.length === 0) {
    return (
      <section className="rounded-xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center dark:border-slate-700 dark:bg-slate-900">
        <Layers3 className="mx-auto h-6 w-6 text-slate-300 dark:text-slate-600" aria-hidden />
        <p className="mt-3 text-sm font-medium text-slate-500 dark:text-slate-400">No mechanisms found</p>
        <p className="mt-1 max-w-md mx-auto text-xs text-slate-400 dark:text-slate-500">
          The agent found no evidence-grounded mechanism of action for{" "}
          <span className="font-medium text-slate-500 dark:text-slate-400">{data.therapeutic_area}</span> across
          trials, FDA records, or literature — not necessarily that none exists.
        </p>
      </section>
    );
  }

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <header className="flex flex-wrap items-center gap-2 border-b border-slate-200 px-6 py-4 dark:border-slate-800">
        <Layers3 className="h-4 w-4 text-slate-500 dark:text-slate-400" aria-hidden />
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300">
          Competitive Landscape — {data.therapeutic_area}
        </h2>
        <span className="ml-auto text-xs text-slate-500 dark:text-slate-400">
          {data.rows.length} mechanism{data.rows.length !== 1 ? "s" : ""} ·{" "}
          {data.rows.reduce((n, r) => n + totalDrugs(r), 0)} drug entries
        </span>
      </header>

      <div className="overflow-x-auto">
        <table
          className="w-full border-collapse text-left"
          style={{ minWidth: `calc(${MECHANISM_COL_WIDTH} + ${data.phases.length} * ${PHASE_COL_WIDTH})` }}
        >
          <thead className="bg-slate-50 dark:bg-slate-950/50">
            <tr>
              <th
                scope="col"
                className="sticky left-0 z-10 border-r border-b border-slate-200 bg-slate-50 px-4 py-3 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:border-slate-800 dark:bg-slate-950/50 dark:text-slate-400"
                style={{ minWidth: MECHANISM_COL_WIDTH, width: MECHANISM_COL_WIDTH }}
              >
                Mechanism / Target
              </th>
              {data.phases.map((phase) => (
                <th
                  key={phase}
                  scope="col"
                  className="border-b border-l border-slate-200 px-3 py-3 text-center text-xs font-semibold uppercase tracking-wide text-slate-500 dark:border-slate-800 dark:text-slate-400"
                  style={{ minWidth: PHASE_COL_WIDTH, width: PHASE_COL_WIDTH }}
                >
                  {phase}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {data.rows.map((row) => {
              // Looked up by phase NAME, not array index -- the schema
              // promises row.cells is exactly [Preclinical, Phase 1, Phase
              // 2, Phase 3, Approved] in order, but this is LLM-generated
              // output, not code-generated; a Map lookup renders correctly
              // even if a future response ever reorders or drops a cell.
              const cellByPhase = new Map(row.cells.map((c) => [c.phase, c]));
              const count = totalDrugs(row);
              return (
                <tr key={row.mechanism} className="border-b border-slate-100 last:border-0 dark:border-slate-800">
                  <th
                    scope="row"
                    className="sticky left-0 z-10 border-r border-slate-200 bg-white px-4 py-3 align-top dark:border-slate-800 dark:bg-slate-900"
                    style={{ minWidth: MECHANISM_COL_WIDTH, width: MECHANISM_COL_WIDTH }}
                  >
                    <div className="flex items-start gap-2">
                      <Dna className="mt-0.5 h-4 w-4 shrink-0 text-sky-600 dark:text-sky-400" aria-hidden />
                      <div>
                        <p className="text-sm font-semibold leading-snug text-slate-800 dark:text-slate-200">
                          {row.mechanism}
                        </p>
                        <p className="mt-0.5 text-[0.7rem] text-slate-400 dark:text-slate-500">
                          {count} drug{count !== 1 ? "s" : ""}
                        </p>
                      </div>
                    </div>
                  </th>
                  {data.phases.map((phase) => (
                    <td key={phase} className="border-l border-slate-100 align-top dark:border-slate-800">
                      <PhaseCellView cell={cellByPhase.get(phase)} />
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
