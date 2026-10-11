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
`http://127.0.0.1:8080/v1`. The code foundry talks to its own endpoint,
`JARVS_CODER_BASE_URL` — leave it empty to reuse the base model's server and only
swap the model id (default `Qwen3-Coder-Next`).)

## Develop (hot reload)

Run the backend and the Vite dev server side by side — Vite proxies `/api` to the
Python backend:

```bash
.venv/bin/python -m jarvs.server      # terminal 1 (API on :8765)
cd web && npm run dev                 # terminal 2 (UI on :5173, proxies /api)
```

## API surface

| Route           | Method | Purpose                                       |
|-----------------|--------|-----------------------------------------------|
| `/api/status`   | GET    | backend / voice / memory / meta / coder status|
| `/api/stats`    | GET    | dashboard metrics (latency, counts, …)        |
| `/api/voice`    | POST   | `{enabled}` → toggle server-side voice         |
| `/api/chat`     | POST   | `{message}` → SSE stream of token deltas       |
| `/api/jobs`     | GET    | recent background jobs                         |
| `/api/routines` | GET    | recurring scheduled tasks                      |
| `/api/code`     | GET    | code foundry: coder backend + recent projects  |
| `/api/code`     | POST   | `{topic}` → queue an autonomous build          |
| `/api/system`   | GET    | live host telemetry (CPU/GPU/mem/time/geo)     |

The FORGE panel in the sidebar shows foundry projects; a finished build also lands in
the transcript as a `JARVS · FORGE` message.

## Config

- `JARVS_WEB_HOST` / `JARVS_WEB_PORT` — bind address (default `127.0.0.1:8765`).
- `JARVS_CLAUDE_BRIDGE=1` turns the FORGE panel's builds into a duet: Qwen3-Coder-Next
  writes the project and Claude reviews the design, diagnoses failures and critiques the
  result (`JARVS_CODER_COLLAB_ROUNDS` caps the reviews per build).
- All model/voice/memory knobs are the usual `JARVS_*` vars (see `jarvs/config.py`).
