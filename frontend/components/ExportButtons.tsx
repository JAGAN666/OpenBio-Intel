"use client";

import { useState } from "react";
import { AlertTriangle, FileSpreadsheet, Loader2, Presentation } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { postAndDownload } from "@/lib/downloadFile";
import type { SmartTableResponse } from "@/types/trial";

type ExportFormat = "excel" | "pptx";

const EXPORT_CONFIG: Record<
  ExportFormat,
  { path: string; label: string; fallbackFilename: string; icon: LucideIcon }
> = {
  excel: {
    path: "/api/export/excel",
    label: "Export to Excel",
    fallbackFilename: "clinical_trials_export.xlsx",
    icon: FileSpreadsheet,
  },
  pptx: {
    path: "/api/export/pptx",
    label: "Export to PowerPoint",
    fallbackFilename: "clinical_landscape_analysis.pptx",
    icon: Presentation,
  },
};

/**
 * Exports whatever SmartTableResponse the analyst is already looking at --
 * `data` IS the request body, so this never re-runs the agent.
 */
export default function ExportButtons({
  apiUrl,
  data,
}: {
  apiUrl: string;
  data: SmartTableResponse;
}) {
  const [pending, setPending] = useState<ExportFormat | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleExport(format: ExportFormat) {
    if (pending) return;
    setPending(format);
    setError(null);
    try {
      const { path, fallbackFilename } = EXPORT_CONFIG[format];
      await postAndDownload(`${apiUrl}${path}`, data, fallbackFilename);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="mb-4 flex flex-wrap items-center gap-2">
      {(Object.keys(EXPORT_CONFIG) as ExportFormat[]).map((format) => {
        const { label, icon: Icon } = EXPORT_CONFIG[format];
        const isPending = pending === format;
        return (
          <button
            key={format}
            type="button"
            disabled={pending !== null}
            onClick={() => void handleExport(format)}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:border-sky-300 hover:text-sky-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            ) : (
              <Icon className="h-4 w-4" aria-hidden />
            )}
            {label}
          </button>
        );
      })}
      {error && (
        <span className="inline-flex items-center gap-1 text-xs text-red-600">
          <AlertTriangle className="h-3.5 w-3.5" aria-hidden />
          Export failed: {error}
        </span>
      )}
    </div>
  );
}
