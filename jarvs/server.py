"""
server.py — the JARVS web backend.

A thin stdlib HTTP server that wraps a single :class:`~jarvs.chat.JarvsSession` (this
is a single-user command center) and exposes it to the React front end as a small JSON
+ SSE API. No framework, no build step; the SPA and its vendored libraries are served
straight from ``jarvs/web``.

Routes:
    GET  /                 the SPA (web/index.html)
    GET  /static/<path>    static assets (app.jsx, styles.css, vendored libs)
    GET  /api/status       backend / voice / memory / meta-dreaming status
    GET  /api/stats        lightweight dashboard metrics (latency, counts, …)
    POST /api/voice        {"enabled": bool} -> toggle server-side voice
    POST /api/chat         {"message": str}  -> SSE stream of token deltas + done

Run:  python -m jarvs.server   (then open http://127.0.0.1:8765)
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import client, config, system_sense
from .chat import JarvsSession

# The production bundle Vite writes (web/ -> jarvs/web_dist via vite.config.js).
_WEB_DIR = os.path.join(os.path.dirname(__file__), "web_dist")

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/json", ".map")


class _Metrics:
    """A couple of live numbers for the dashboard widgets."""
    def __init__(self):
        self.last_latency_ms = 0
        self.last_tokens = 0
        self.turns = 0

    def record(self, latency_ms: int, tokens: int):
        self.last_latency_ms = latency_ms
        self.last_tokens = tokens
        self.turns += 1


class JarvsHandler(BaseHTTPRequestHandler):
    session: JarvsSession = None      # set in serve()
    metrics: _Metrics = None

    # Quieten the default per-request stderr logging.
    def log_message(self, *args):
        pass

    # -- helpers -------------------------------------------------------------- #

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except (ValueError, TypeError, json.JSONDecodeError):
            return {}

    def _serve_file(self, path: str):
        # Resolve inside _WEB_DIR only (no traversal).
        full = os.path.normpath(os.path.join(_WEB_DIR, path))
        if not full.startswith(_WEB_DIR) or not os.path.isfile(full):
            self.send_error(404, "Not found")
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        try:
            with open(full, "rb") as fh:
                data = fh.read()
        except OSError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- status payloads ------------------------------------------------------ #

    def _status_payload(self) -> dict:
        ok, detail = client.health()
        meta = (self.session.meta_dreamer.snapshot()
                if self.session.meta_dreamer else None)
        return {
            "backend_ok": ok,
            "backend_detail": detail,
            "model": config.MODEL,
            "voice": self.session.voice.status(),
            "voice_enabled": self.session.voice.enabled,
            "memory": self.session.memory.stats(),
            "meta": {"promoted": meta["promoted_areas"],
                     "total": meta["total_areas"],
                     "pending": meta["pending_lines"]} if meta else None,
            "turns": self.metrics.turns,
        }

    def _stats_payload(self) -> dict:
        mem = self.session.memory.stats()
        meta = (self.session.meta_dreamer.snapshot()
                if self.session.meta_dreamer else None)
        ok, _ = client.health()
        return {
            "backend_ok": ok,
            "latency_ms": self.metrics.last_latency_ms,
            "last_tokens": self.metrics.last_tokens,
            "turns": self.metrics.turns,
            "episodic": mem.get("episodic", 0) if mem.get("enabled") else 0,
            "semantic": mem.get("semantic", 0) if mem.get("enabled") else 0,
            "links": (mem.get("links") or {}).get("count", 0) if mem.get("enabled") else 0,
            "focus_promoted": meta["promoted_areas"] if meta else 0,
            "focus_pending": meta["pending_lines"] if meta else 0,
            "voice_enabled": self.session.voice.enabled,
            "jobs": (self.session.background.counts()
                     if self.session.background else
                     {"queued": 0, "running": 0, "completed": 0, "failed": 0}),
        }

    def _jobs_payload(self) -> dict:
        if self.session.background is None:
            return {"enabled": False, "jobs": []}
        return {"enabled": True, "jobs": self.session.background.recent(12)}

    def _routines_payload(self) -> dict:
        if self.session.routines is None:
            return {"enabled": False, "routines": []}
        return {"enabled": True, "routines": self.session.routines.list()}

    def _system_payload(self) -> dict:
        if not config.SYSTEM_SENSE:
            return {"enabled": False}
        # Geo is cached; snapshot(include_geo=True) only hits the network on a miss.
        snap = system_sense.snapshot(include_geo=True)
        snap["enabled"] = True
        return snap

    # -- GET ------------------------------------------------------------------ #

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/status":
            self._send_json(self._status_payload())
        elif route == "/api/stats":
            self._send_json(self._stats_payload())
        elif route == "/api/jobs":
            self._send_json(self._jobs_payload())
        elif route == "/api/system":
            self._send_json(self._system_payload())
        elif route == "/api/routines":
            self._send_json(self._routines_payload())
        elif route.startswith("/api/"):
            self.send_error(404, "Not found")
        elif not os.path.isdir(_WEB_DIR):
            # The UI hasn't been built yet — guide rather than 404.
            msg = (b"<!doctype html><meta charset=utf-8>"
                   b"<body style='background:#0a0a0a;color:#8a94a3;font-family:monospace;"
                   b"padding:3rem;line-height:1.7'>"
                   b"<h2 style='color:#00d4ff'>JARVS UI not built</h2>"
                   b"<p>Build the front end, then reload:</p>"
                   b"<pre style='color:#e6e9ef'>cd web &amp;&amp; npm install &amp;&amp; npm run build</pre>"
                   b"<p>The API is live at <code>/api/status</code>.</p></body>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
        else:
            # Serve the SPA bundle; fall back to index.html for client-side routes.
            rel = route.lstrip("/") or "index.html"
            full = os.path.normpath(os.path.join(_WEB_DIR, rel))
            if full.startswith(_WEB_DIR) and os.path.isfile(full):
                self._serve_file(rel)
            else:
                self._serve_file("index.html")

    # -- POST ----------------------------------------------------------------- #

    def do_POST(self):
        route = urlparse(self.path).path
        if route == "/api/voice":
            body = self._read_json()
            self.session.voice.set_enabled(bool(body.get("enabled")))
            self._send_json({"voice": self.session.voice.status(),
                             "voice_enabled": self.session.voice.enabled})
        elif route == "/api/chat":
            self._stream_chat()
        else:
            self.send_error(404, "Not found")

    def _sse_event(self, event: str, data: dict):
        payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        self.wfile.write(payload.encode("utf-8"))
        self.wfile.flush()

    def _stream_chat(self):
        body = self._read_json()
        message = (body.get("message") or "").strip()
        if not message:
            self._send_json({"error": "empty message"}, status=400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # Close after the stream so clients get a clean EOF (no Content-Length on an
        # event stream means the socket close is the only end-of-body signal).
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        start = time.time()
        parts: list[str] = []
        try:
            for delta in self.session.stream_reply(message, speak=True):
                parts.append(delta)
                self._sse_event("token", {"t": delta})
            latency_ms = int((time.time() - start) * 1000)
            reply = "".join(parts).strip()
            if not reply:
                self._sse_event("error", {"detail": client.LAST_GENERATION_ERROR
                                          or "no reply"})
                return
            self.metrics.record(latency_ms, len(reply.split()))
            self._sse_event("done", {"reply": reply, "latency_ms": latency_ms})
        except (BrokenPipeError, ConnectionResetError):
            return                                       # client navigated away
        except Exception as e:  # noqa: BLE001
            try:
                self._sse_event("error", {"detail": f"{type(e).__name__}: {e}"})
            except Exception:  # noqa: BLE001
                pass


def serve(host: str = "", port: int = 0) -> None:
    host = host or os.environ.get("JARVS_WEB_HOST", "127.0.0.1")
    port = port or int(os.environ.get("JARVS_WEB_PORT", "8765"))
    JarvsHandler.session = JarvsSession()
    JarvsHandler.metrics = _Metrics()
    httpd = ThreadingHTTPServer((host, port), JarvsHandler)
    ok, detail = client.health()
    print(f"JARVS web UI → http://{host}:{port}")
    print(f"  · {detail}")
    print(f"  · {JarvsHandler.session.voice.status()}")
    print(f"  · {JarvsHandler.session.memory.status()}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()


if __name__ == "__main__":
    serve()
