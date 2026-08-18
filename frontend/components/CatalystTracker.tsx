"use client";

import { Building2, CalendarClock, ExternalLink, FlaskConical, Landmark } from "lucide-react";

import { sourceUrl } from "@/lib/sourceUrl";
import type { CatalystEvent, CatalystTimeline } from "@/types/catalysts";

/** Events arrive chronologically sorted from the backend
 * (_finalize_catalyst_timeline) -- these are pure display-key helpers, not
 * a re-sort. */
function groupKey(e: CatalystEvent): string {
  return `${e.year}|${e.quarter}`;
}

function groupLabel(e: CatalystEvent): string {
  return e.quarter ? `${e.quarter} ${e.year}` : String(e.year);
}

/**
 * Two distinct evidence lenses back an event -- a trial's own live-looked-
 * up completion date, or a company's own stated regulatory timeline in an
 * SEC filing (see research_agent.py's CATALYST_SYSTEM). Badging which one
 * grounds a given card is what lets a reader judge its nature at a glance:
 * a trial's projected primary-completion date is a scientific/operational
 * signal, while a company-stated PDUFA date is a regulatory/corporate one.
 */
function SourceBadge({ sourceType }: { sourceType: string }) {
  const isSec = sourceType === "sec";
  const Icon = isSec ? Landmark : FlaskConical;
  const className = isSec
    ? "inline-flex shrink-0 items-center gap-1 rounded-md border border-amber-200 bg-amber-50 px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide text-amber-700 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-400"
    : "inline-flex shrink-0 items-center gap-1 rounded-md border border-sky-200 bg-sky-50 px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide text-sky-700 dark:border-sky-900/60 dark:bg-sky-950/40 dark:text-sky-400";
  return (
    <span className={className}>
      <Icon className="h-3 w-3" aria-hidden />
      {isSec ? "Corporate Filing" : "Clinical Trial"}
    </span>
  );
}

function EventCard({ event }: { event: CatalystEvent }) {
  const url = sourceUrl(event.source);
  return (
    <div className="flex h-full flex-col rounded-xl border border-slate-200 bg-white p-4 shadow-sm transition-colors hover:border-sky-300 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-sky-700">
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm font-semibold leading-snug text-slate-800 dark:text-slate-200">{event.drug_name}</p>
        <SourceBadge sourceType={event.source_type} />
      </div>
      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{event.indication}</p>

      <div className="mt-3 flex flex-col gap-1.5 text-xs text-slate-600 dark:text-slate-400">
        <span className="inline-flex items-center gap-1.5">
          <CalendarClock className="h-3.5 w-3.5 shrink-0 text-slate-400 dark:text-slate-500" aria-hidden />
          <span className="font-medium text-slate-700 dark:text-slate-300">{event.event_type}</span>
          <span className="text-slate-400 dark:text-slate-600">·</span>
          {event.display_date}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Building2 className="h-3.5 w-3.5 shrink-0 text-slate-400 dark:text-slate-500" aria-hidden />
          {event.company}
        </span>
      </div>

      <div className="mt-auto pt-3">
        {url ? (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs font-medium text-sky-700 hover:text-sky-900 dark:text-sky-400 dark:hover:text-sky-300"
          >
            View source ({event.source})
            <ExternalLink className="h-3 w-3" aria-hidden />
          </a>
        ) : (
          <span className="text-xs text-slate-400 dark:text-slate-500">{event.source}</span>
        )}
      </div>
    </div>
  );
}

export default function CatalystTracker({ data }: { data: CatalystTimeline }) {
  if (data.events.length === 0) {
    return (
      <section className="rounded-xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center dark:border-slate-700 dark:bg-slate-900">
        <CalendarClock className="mx-auto h-6 w-6 text-slate-300 dark:text-slate-600" aria-hidden />
        <p className="mt-3 text-sm font-medium text-slate-500 dark:text-slate-400">No catalysts found</p>
        <p className="mx-auto mt-1 max-w-md text-xs text-slate-400 dark:text-slate-500">
          The agent found no evidence-grounded upcoming events for{" "}
          <span className="font-medium text-slate-500 dark:text-slate-400">&ldquo;{data.query}&rdquo;</span> — not
          necessarily that none exist.
        </p>
      </section>
    );
  }

  const groups: { key: string; label: string; events: CatalystEvent[] }[] = [];
  for (const event of data.events) {
    const key = groupKey(event);
    const last = groups[groups.length - 1];
    if (last && last.key === key) {
      last.events.push(event);
    } else {
      groups.push({ key, label: groupLabel(event), events: [event] });
    }
  }

  return (
    <section className="space-y-6">
      <header className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-200 bg-white px-6 py-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <CalendarClock className="h-4 w-4 text-slate-500 dark:text-slate-400" aria-hidden />
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-700 dark:text-slate-300">
          Catalyst Timeline — {data.query}
        </h2>
        <span className="ml-auto text-xs text-slate-500 dark:text-slate-400">
          {data.events.length} event{data.events.length !== 1 ? "s" : ""} across{" "}
          {groups.length} period{groups.length !== 1 ? "s" : ""}
        </span>
      </header>

      <div className="space-y-8">
        {groups.map((group) => (
          <div key={group.key}>
            <div className="mb-3 flex items-center gap-3">
              <span className="h-2 w-2 shrink-0 rounded-full bg-sky-600 dark:bg-sky-500" aria-hidden />
              <h3 className="text-sm font-bold uppercase tracking-wide text-sky-700 dark:text-sky-400">
                {group.label}
              </h3>
              <span className="h-px flex-1 bg-slate-200 dark:bg-slate-800" aria-hidden />
              <span className="text-xs text-slate-400 dark:text-slate-500">
                {group.events.length} event{group.events.length !== 1 ? "s" : ""}
              </span>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {group.events.map((event, i) => (
                <EventCard key={`${event.source}-${i}`} event={event} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
