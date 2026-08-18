import {
  AlertTriangle,
  Brain,
  CheckCircle2,
  Database,
  FileText,
  Layers,
  Loader2,
  ShieldCheck,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

// Mirrors api.py's SSE event payloads exactly (_sse("status", ...) /
// _sse("progress", ...)) -- see that module's _stream_events docstring.
export type StreamStatus = {
  node: string;
  phase: "start" | "done" | "error";
  total?: number;
  error?: string;
};

export type StreamProgress = {
  node: string;
  completed: number;
  total: number;
};

const NODE_META: Record<string, { label: string; icon: LucideIcon }> = {
  IntentClassifier: { label: "Checking the question is in-domain", icon: ShieldCheck },
  Agent: { label: "Deciding which tools to call", icon: Brain },
  ToolNode: { label: "Querying trials, graph, literature & FDA records", icon: Database },
  MapWorkers: { label: "Extracting trial data in parallel", icon: Layers },
  Reducer: { label: "Synthesizing the final answer", icon: FileText },
  OutOfDomain: { label: "Question is out of domain", icon: AlertTriangle },
  NoResultsFallback: { label: "No matching trials found", icon: AlertTriangle },
};

function nodeMeta(node: string) {
  return NODE_META[node] ?? { label: node, icon: Loader2 };
}

export default function ProgressPanel({
  timeline,
  mapProgress,
  elapsed,
}: {
  timeline: StreamStatus[];
  mapProgress: StreamProgress | null;
  elapsed: number;
}) {
  // A node can appear more than once (start, then done) -- collapse to its
  // latest phase per node while keeping first-seen order for the timeline.
  const byNode = new Map<string, StreamStatus>();
  const order: string[] = [];
  for (const evt of timeline) {
    if (!byNode.has(evt.node)) order.push(evt.node);
    byNode.set(evt.node, evt);
  }
  const current = timeline[timeline.length - 1];

  return (
    <section
      className="mb-8 rounded-xl border border-sky-200 bg-sky-50 px-6 py-6 dark:border-sky-900/60 dark:bg-sky-950/30"
      aria-live="polite"
    >
      <div className="flex items-center gap-3">
        <Loader2 className="h-5 w-5 shrink-0 animate-spin text-sky-700 dark:text-sky-400" aria-hidden />
        <div>
          <p className="text-sm font-semibold text-sky-900 dark:text-sky-200">
            {current ? nodeMeta(current.node).label : "Starting…"}
          </p>
          <p className="text-xs text-sky-700 dark:text-sky-400">{elapsed}s elapsed</p>
        </div>
      </div>

      {mapProgress && mapProgress.total > 0 && (
        <div className="mt-4">
          <div className="flex items-center justify-between text-xs text-sky-800 dark:text-sky-300">
            <span>Trials extracted (parallel Map workers)</span>
            <span className="font-mono">
              {mapProgress.completed} / {mapProgress.total}
            </span>
          </div>
          <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-sky-100 dark:bg-sky-900/50">
            <div
              className="h-full rounded-full bg-sky-600 transition-all duration-300 dark:bg-sky-500"
              style={{
                width: `${Math.min(100, (mapProgress.completed / mapProgress.total) * 100)}%`,
              }}
            />
          </div>
        </div>
      )}

      {order.length > 0 && (
        <ol className="mt-4 space-y-1.5">
          {order.map((node) => {
            const evt = byNode.get(node)!;
            const meta = nodeMeta(node);
            const Icon =
              evt.phase === "done" ? CheckCircle2 : evt.phase === "error" ? AlertTriangle : meta.icon;
            return (
              <li key={node} className="flex items-center gap-2 text-xs">
                <Icon
                  className={
                    "h-3.5 w-3.5 shrink-0 " +
                    (evt.phase === "done"
                      ? "text-emerald-600 dark:text-emerald-400"
                      : evt.phase === "error"
                        ? "text-red-600 dark:text-red-400"
                        : "text-sky-500 dark:text-sky-400")
                  }
                  aria-hidden
                />
                <span className={evt.phase === "error" ? "text-red-700 dark:text-red-400" : "text-sky-800 dark:text-sky-300"}>
                  {meta.label}
                  {node === "MapWorkers" && evt.total
                    ? ` (${evt.total} trial${evt.total === 1 ? "" : "s"})`
                    : ""}
                </span>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
