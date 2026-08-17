/**
 * Minimal `text/event-stream` parser over a fetch() ReadableStream.
 *
 * Not using the browser's native EventSource: EventSource only supports
 * GET requests, and /api/research/stream is a POST (same ResearchRequest
 * body /api/research already uses, and the query is free text that
 * shouldn't end up in a URL). fetch()'s own ReadableStream works for any
 * method, so this reads the response body directly instead.
 */

export interface SSEMessage {
  event: string;
  data: string;
}

function parseRawMessage(raw: string): SSEMessage | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

/**
 * Yields one {event, data} pair per SSE message. Buffers across reads --
 * a network chunk can end anywhere in the byte stream, not just on the
 * blank-line message boundary, so a message split across two reads must
 * still parse correctly.
 */
export async function* parseSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<SSEMessage> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sepIndex: number;
      while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, sepIndex);
        buffer = buffer.slice(sepIndex + 2);
        const msg = parseRawMessage(raw);
        if (msg) yield msg;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
