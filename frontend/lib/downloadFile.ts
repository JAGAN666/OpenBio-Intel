/**
 * POSTs a JSON body to `url` and saves the binary response as a file
 * download, using the filename the server sent via Content-Disposition
 * (falling back to `fallbackFilename` if that header is absent).
 */
export async function postAndDownload(
  url: string,
  body: unknown,
  fallbackFilename: string,
): Promise<void> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    // Same convention as the research endpoints — FastAPI's {detail: ...}
    // shown verbatim rather than a generic failure.
    let detail = `HTTP ${res.status}`;
    try {
      const errBody = await res.json();
      if (errBody?.detail) {
        detail = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
      }
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }

  const disposition = res.headers.get("Content-Disposition") ?? "";
  const match = /filename="?([^"]+)"?/.exec(disposition);
  const filename = match?.[1] ?? fallbackFilename;

  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  // A temporary, invisible <a download> is the standard way to trigger a
  // browser save from an in-memory blob — there is no direct
  // "save this blob as a file" browser API.
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}
