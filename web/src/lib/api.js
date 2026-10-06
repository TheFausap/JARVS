// Thin client for the JARVS Python backend (jarvs/server.py).
// In dev, Vite proxies /api -> 127.0.0.1:8765; in the built app the same-origin
// Python server answers directly.

export async function getStatus() {
  const r = await fetch("/api/status");
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

export async function getStats() {
  const r = await fetch("/api/stats");
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

export async function setVoice(enabled) {
  const r = await fetch("/api/voice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  return r.json();
}

// Stream a reply over SSE. Calls onToken(delta) as content arrives, then
// onDone({reply, latency_ms}) or onError(detail). Returns a promise that
// resolves when the stream ends.
export async function streamChat(message, { onToken, onDone, onError } = {}) {
  let resp;
  try {
    resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
  } catch (e) {
    onError?.(String(e));
    return;
  }
  if (!resp.ok || !resp.body) {
    onError?.(`backend returned ${resp.status}`);
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  while (!finished) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      let payload;
      try {
        payload = JSON.parse(data);
      } catch {
        continue;
      }
      if (event === "token") onToken?.(payload.t || "");
      else if (event === "done") { onDone?.(payload); finished = true; }
      else if (event === "error") { onError?.(payload.detail || "unknown error"); finished = true; }
    }
  }
  // Don't depend solely on the server closing the socket.
  try { await reader.cancel(); } catch { /* ignore */ }
}
