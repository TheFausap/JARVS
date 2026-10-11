# FACTS

- JARVIS type UI and behaviour
- web interface javascript based
- reuse parts of JCN3690
- add the idea of meta-dreaming from ENGRAM2
- external memories
- voice output

# DECISIONS (2026-10-06)

- **Base model**: `TeichAI/gemma-4-26B-A4B-it-Claude-Opus-Distill-v2-GGUF`, single
  backend. MoE (~4B active → fast), Opus-distilled → strong conversational/persona
  fit for a JARVIS assistant. Qwen3-Coder dropped for v1; ~~can be added later as a
  `*_FAST_*` coder backend~~ → added 2026-10-11 as the foundry's coding specialist
  (`Qwen3-Coder-Next`, see AGENT CAPABILITIES).
- **Architecture**: transformer GGUF served over an OpenAI-compatible endpoint
  (llama.cpp). Base model is treated as a FIXED stage0 (JCN3690 philosophy); all
  capability is built as external primitives. RWKV/MAMBA deferred — the endpoint is
  arch-agnostic, so a linear-attention base can be swapped in later as an experiment
  without rearchitecting. External memory makes long base context non-critical.

# REUSE MAP

- **Web UI** ← see WEB UI section (React/Vite SPA + Python SSE API). The old
  `engram_web_poc.py` server-rendered approach was superseded by the spec below;
  its backend techniques (HTTP serving, SSE streaming) carried over into `jarvs/server.py`.
- **Voice output** ← JCN3690 `hal_voice.py` (Piper → `say` → off; swappable TTS)
- **External memory** ← JCN3690 `vector_store.py` + `assoc_memory.py` (Chroma + sentence-transformers)
- **Meta-dreaming** ← ENGRAM2 `meta_dreaming.py` (bounded self-referential critic-prompt refinement)
- **Model wiring**: OpenAI-compatible via `*_BASE_URL` / `*_MODEL` / `*_API_KEY` env vars

# WEB UI

**Project Specification: JARVIS UI**

**Core Identity:**
The UI should feel like a sophisticated, minimalist command center. It must be functional, precise, and cinematicânever cluttered.

**Tech Stack:**
- Framework: React
- Styling: Tailwind CSS
- Animation: Framer Motion
- Icons: Lucide React or custom SVG strokes

**Visual Language:**
- Palette: Deep charcoal (#0a0a0a) background, cyan/ice blue (#00d4ff) for accents, muted slate grays for secondary text.
- Typography: Inter or JetBrains Mono for a technical, clean feel.
- Borders: Thin, 1px borders with subtle glows. Avoid heavy shadows.

**Key UI Components:**
1. **Status Ring:** A central, minimalist circular element that pulses subtly during activity.
2. **Message Stream:** Clean, left-aligned user messages and right-aligned assistant responses. No bubblesâjust clear typography and subtle dividers.
3. **Data Dashboard:** A side panel with small, real-time widgets (latency, memory usage, system status) to create the feeling of a live system.
4. **Input Terminal:** A sleek, single-line command bar at the bottom with a subtle glow when active.

**Animation Requirements:**
- All transitions must be eased and purposeful.
- Elements entering the screen should fade and slide upwards slightly.
- The status ring should animate its stroke-dasharray during processing states.
- No abrupt pops; everything should feel fluid.

**Implementation status (2026-10-06):** built in `web/` (Vite). Node installed at
`~/.local/node` (no system node). Backend is `jarvs/server.py` (stdlib HTTP + SSE),
reusing JarvsSession/client/memory/critic. All four components built
(StatusRing, MessageStream, DataDashboard, InputTerminal). Token streaming, the
state machine, dashboard polling, and voice toggle verified against the live model.
Run: `cd web && npm run build` then `python -m jarvs.server` (see web/README.md).

# AGENT CAPABILITIES (2026-10-06)

Background/agent-like layer added after the core build:
- **Background jobs** (`jarvs/background.py`): a persisted queue + single worker daemon.
  Tasks are deferred off the conversation path, each gathering its own context and
  (optionally) doing one live web search / URL fetch, then running a bounded analysis
  and storing the result in memory. Retries transient timeouts; emits status events.
  Triggered by plain-language requests ("research X in the background, tell me when
  done"), a `/background <task>` command, or model `[[THINK: …]]` markers.
- **Web research** (`jarvs/web_access.py`): DuckDuckGo search + URL fetch (`ddgs`,
  optional). Used by background jobs; idle browse-to-learn deliberately deferred.
- **Auto-dream cadence**: the associative consolidation pass now runs automatically
  every `JARVS_DREAM_EVERY` (default 12) turns, off the reply path.
- Surfaced in the web UI: `/api/jobs`, a JOBS widget + background list, and completed
  results appear in the transcript. REPL: `/background`, `/jobs`, live job events.
- **Recurring routines** (`jarvs/routines.py`): a scheduler daemon fires background
  jobs on a timer. A plain-language scheduling request ("add daily searches about X
  and Y") is turned into structured routines by one LLM extraction pass and actually
  created — closing the earlier gap where the model *claimed* to schedule searches but
  nothing was created. Intervals: hourly/daily/weekly/every-N. Web UI: `/api/routines`
  + a ROUTINES panel. REPL: `/routines [rm|off|on|run <id>]`. v1 fires on an interval
  from creation; clock-time anchoring ("08:00 digest") is a possible follow-up.
- Not ported: idle web-learning, meditation.
- **System awareness** (`jarvs/system_sense.py`): live host telemetry from the OS —
  CPU load/util, memory, NVIDIA GPU (util/VRAM/temp via nvidia-smi), disk, thermals,
  uptime, local clock/timezone, network (gateway/local IP), and IP-based geolocation
  (router's public IP → ip-api.com, cached JARVS_GEO_TTL, gated by JARVS_GEOLOCATION +
  JARVS_ALLOW_WEB). A `[SYSTEM STATUS]` block is injected into the reply when the user
  asks about the host/time/location, so answers are grounded in real numbers. Web UI:
  `/api/system` + a HOST section (CPU/GPU/RAM/temp), live clock, and location. REPL:
  `/system`. Geolocation is IP-based, not wifi-BSSID (keyless, approximate, opt-out).

- **Autonomous coding — the foundry** (`jarvs/coder.py`, 2026-10-11): give JARVS a
  TOPIC and it builds a working project on disk, unattended. Pipeline:
  1. **plan** — one call to the coder model turns the bare topic into a design (name,
     language, entry point, file list with a purpose each, and the command that proves
     it works). Scope is deliberately capped at `JARVS_CODER_MAX_FILES` (8): working
     beats ambitious.
  2. **write** — every file is generated in full, in design order, with the design and
     the already-written files in context so the pieces actually agree.
  3. **verify** — the project is really run: `python -m py_compile` / `node --check`
     for syntax, then the test command (pytest, node --test, …) inside the project dir.
  4. **repair** — the failure goes back to the model with the offending file and the
     raw error; the loop repeats up to `JARVS_CODER_MAX_REPAIRS` (2) times.
  Each project gets a `JARVS_REPORT.md` (design, files, check output, repair rounds)
  and lands in memory. Bounded + sandboxed: writes are confined to
  `JARVS_PROJECTS_DIR` (default `data/projects`) with path traversal rejected;
  subprocesses come from an allowlist (`python`, `pytest`, `node`, `npm`, `go`, …),
  run with `shell=False`, a timeout, and no shell metacharacters; `JARVS_CODER_EXEC=0`
  disarms execution entirely.
- **Coder model**: `Qwen3-Coder-Next`, reached over the same OpenAI-compatible wire as
  the base model. `JARVS_CODER_BASE_URL` empty (the default) reuses the base model's
  server and swaps only the model id; pointing it elsewhere gives the coder its own
  server, context slot and timeout. If the coder endpoint is unreachable the foundry
  falls back to the base backend, so a build degrades instead of dying.
  (This closes the deferred `*_FAST_*` coder backend from the DECISIONS block — the
  coding specialist is a second model, not a bigger prompt.)
- **Triggers**: `@/code <topic>` command, or plain language ("build me a habit
  tracker") detected by `is_code_request` — a build verb aimed at a software object,
  never on a question ("how would you build a parser?" stays a conversation). The model
  can also defer a build itself with a `[[CODE: topic]]` marker, mirroring
  `[[THINK:]]`. Builds run on their own worker thread, off the reply path, like
  background jobs. REPL: `/code`, `/projects [rm|open|tree <id>]`.
- **Web UI**: `/api/code` (GET projects / POST a topic), a FORGE widget + project list
  in the sidebar, and finished builds surface in the transcript as `JARVS · FORGE`.
  The system head carries a `[AUTONOMOUS CODING]` block so the model says "on it"
  instead of confabulating files it never wrote — the same anti-confabulation fix the
  routines layer needed.
- Foundry tests in `tests/test_coder.py` (parsing, traversal/command sandbox, and the
  whole plan→write→verify→repair loop against a stubbed coder) + dispatch tests in
  `tests/test_chat_spine.py`.

# OPEN QUESTIONS

- (resolved — see DECISIONS)

# REFERENCE PROJECTS

- JCN3690: /home/fausap/LLM/JCN3690
- ENGRAM2: /home/fausap/LLM/ENGRAM2
