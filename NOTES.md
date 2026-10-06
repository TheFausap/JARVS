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
  fit for a JARVIS assistant. Qwen3-Coder dropped for v1; can be added later as a
  `*_FAST_*` coder backend.
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

# OPEN QUESTIONS

- (resolved — see DECISIONS)

# REFERENCE PROJECTS

- JCN3690: /home/fausap/LLM/JCN3690
- ENGRAM2: /home/fausap/LLM/ENGRAM2
