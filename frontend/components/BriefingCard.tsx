import { FileText } from "lucide-react";

const CTGOV = "https://clinicaltrials.gov/study/";

/**
 * Renders the narrative with two lightweight transforms:
 *   **bold**  -> <strong>
 *   NCT01234567 -> link to the trial registry
 * Deliberately not a full markdown parser — the payload only ever contains
 * these two constructs, and dangerouslySetInnerHTML is avoided entirely.
 */
function renderInline(text: string, keyPrefix: string) {
  const nodes: React.ReactNode[] = [];
  // Split on **bold** spans and NCT ids in one pass.
  const pattern = /(\*\*[^*]+\*\*|NCT\d{8})/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let i = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) nodes.push(text.slice(last, match.index));
    const token = match[0];

    if (token.startsWith("**")) {
      nodes.push(
        <strong key={`${keyPrefix}-b-${i++}`} className="font-semibold text-slate-900">
          {token.slice(2, -2)}
        </strong>,
      );
    } else {
      nodes.push(
        <a
          key={`${keyPrefix}-l-${i++}`}
          href={`${CTGOV}${token}`}
          target="_blank"
          rel="noopener noreferrer"
          className="font-mono text-[0.85em] text-sky-700 underline decoration-sky-300 underline-offset-2 hover:text-sky-900 hover:decoration-sky-500"
        >
          {token}
        </a>,
      );
    }
    last = match.index + token.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

export default function BriefingCard({ summary }: { summary: string }) {
  const paragraphs = summary.split("\n\n").filter((p) => p.trim().length > 0);

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <header className="flex items-center gap-2 border-b border-slate-200 px-6 py-4">
        <FileText className="h-4 w-4 text-slate-500" aria-hidden />
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-700">
          The Briefing
        </h2>
        <span className="ml-auto text-xs text-slate-400">Executive summary</span>
      </header>

      <div className="space-y-4 px-6 py-5">
        {paragraphs.map((p, idx) => (
          <p key={idx} className="text-[0.95rem] leading-relaxed text-slate-700">
            {renderInline(p, `p${idx}`)}
          </p>
        ))}
      </div>
    </section>
  );
}
