/**
 * Async-job client: enqueue via POST /api/jobs, then follow the job's SSE
 * stream with a native EventSource.
 *
 * Why EventSource here when the old /api/research/stream needed the manual
 * parseSSE reader: the job stream is a GET (the job id carries everything a
 * POST body used to), and GET unlocks EventSource's built-in AUTOMATIC
 * RECONNECTION. Combined with the server replaying a job's full event log
 * on every (re)connect, a dropped connection -- or a refreshed tab
 * re-subscribing to the same job id -- recovers to exact current state
 * with no client bookkeeping. `onReplayReset` fires on every (re)connect
 * so callers can clear accumulated timeline state before the replay
 * repopulates it (otherwise events would duplicate after a reconnect).
 */

export type JobType = "research" | "landscape" | "catalysts";

export interface JobHandlers {
  onStatus?: (data: unknown) => void;
  onProgress?: (data: unknown) => void;
  onReplayReset?: () => void;
}

export async function runJob<T>(
  apiUrl: string,
  type: JobType,
  query: string,
  handlers: JobHandlers = {},
): Promise<T> {
  const res = await fetch(`${apiUrl}/api/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type, query }),
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  const { job_id } = (await res.json()) as { job_id: string };

  return await new Promise<T>((resolve, reject) => {
    const es = new EventSource(`${apiUrl}/api/jobs/${job_id}/stream`);

    es.onopen = () => handlers.onReplayReset?.();
    es.addEventListener("status", (e) =>
      handlers.onStatus?.(JSON.parse((e as MessageEvent).data)),
    );
    es.addEventListener("progress", (e) =>
      handlers.onProgress?.(JSON.parse((e as MessageEvent).data)),
    );
    es.addEventListener("result", (e) => {
      es.close();
      resolve(JSON.parse((e as MessageEvent).data) as T);
    });
    es.addEventListener("error", (e) => {
      // Two different things land here: a SERVER-SENT `error` event (has
      // .data -- the job genuinely failed, stop) vs a TRANSPORT error (no
      // .data -- EventSource is about to auto-reconnect and the server
      // will replay; do nothing).
      const data = (e as MessageEvent).data;
      if (data) {
        es.close();
        try {
          reject(new Error((JSON.parse(data) as { message: string }).message));
        } catch {
          reject(new Error(String(data)));
        }
      }
    });
  });
}
