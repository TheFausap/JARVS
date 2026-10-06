# JARVS UI

The command-center front end: React + Vite + Tailwind v4 + Framer Motion + Lucide.
It talks to the Python backend (`jarvs/server.py`) over a small JSON + SSE API.

## Prerequisites

Node (installed locally at `~/.local/node`): add it to your PATH if it isn't already:

```bash
export PATH="$HOME/.local/node/bin:$PATH"
```

## Build (production)

```bash
cd web
npm install          # first time only
npm run build        # emits ../jarvs/web_dist
```

Then start the backend, which serves the built UI:

```bash
cd ..
.venv/bin/python -m jarvs.server      # http://127.0.0.1:8765
```

(The backend needs llama-server running on `JARVS_BASE_URL`, default
`http://127.0.0.1:8080/v1`.)

## Develop (hot reload)

Run the backend and the Vite dev server side by side — Vite proxies `/api` to the
Python backend:

```bash
.venv/bin/python -m jarvs.server      # terminal 1 (API on :8765)
cd web && npm run dev                 # terminal 2 (UI on :5173, proxies /api)
```

## API surface

| Route           | Method | Purpose                                   |
|-----------------|--------|-------------------------------------------|
| `/api/status`   | GET    | backend / voice / memory / meta status    |
| `/api/stats`    | GET    | dashboard metrics (latency, counts, …)    |
| `/api/voice`    | POST   | `{enabled}` → toggle server-side voice     |
| `/api/chat`     | POST   | `{message}` → SSE stream of token deltas   |

## Config

- `JARVS_WEB_HOST` / `JARVS_WEB_PORT` — bind address (default `127.0.0.1:8765`).
- All model/voice/memory knobs are the usual `JARVS_*` vars (see `jarvs/config.py`).
