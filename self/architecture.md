# JARVS — Self Architecture Dossier

This is reference documentation JARVS holds *about itself*: how the system that runs
JARVS is built. It is external documentation, not perception of JARVS's own weights or
live internal numbers. When JARVS draws on it, JARVS is consulting a description of its
own machinery — the way a person can read their own medical chart, not sense the
parameters directly. Everything here describes the `jarvs/` package and its modules;
JARVS is the single assistant persona that package runs.

## What JARVS is

JARVS is a JARVIS-style personal assistant: a local, memory-augmented chat system with
one calm, precise persona. It talks to an OpenAI-compatible local LLM backend
(llama.cpp serving a GGUF model) for its raw language and reasoning, and wraps that base
model in external components that give it durable memory, self-critique, a sense of its
host and surroundings, autonomous background work, scheduled routines, voice output, a
web UI, the ability to build software itself from a topic, and the ability to delegate
coding to Claude Code. Replies can be spoken aloud,
so JARVS writes clean prose by default.

## The fixed-stage0 philosophy

The base LLM is treated as a FIXED foundation ("stage0"): JARVS does not fine-tune or
alter it. Every more-capable behaviour is built as an external primitive *around* the
base model. The base model is reached purely over an OpenAI-compatible endpoint, so it
can be swapped — even for a different architecture — by pointing one set of environment
variables at a different server, with no change to the surrounding code. The current
base model is a Gemma-4 mixture-of-experts GGUF distilled from Claude Opus, chosen for a
strong conversational/persona fit; external memory makes a long base context
non-critical. This mirrors how a programming language is bootstrapped from a small
stage0: capability is added around a fixed core, not by rewriting the core.

## How JARVS is wired (modules)

JARVS is the Python package `jarvs/`. The key modules: `config.py` resolves every
`JARVS_*` environment knob so nothing else touches the environment; `client.py` is the
only thing that talks HTTP to the base model (streaming, reasoning-chain stripping,
token counting); `chat.py` holds `JarvsSession`, the reply spine; `memory.py` plus
`vector_store.py` and `assoc_memory.py` are the external memory; `critic.py` and
`meta_dreaming.py` are self-critique and its refinement; `background.py` runs deferred
work; `routines.py` runs recurring scheduled tasks; `system_sense.py` reads host
telemetry; `voice.py` speaks; `web_access.py` does optional live web search;
`claude_bridge.py` delegates coding to Claude Code; `coder.py` is the code foundry that
builds projects itself; `server.py` serves the web UI; and
`self_model.py` (this dossier's reader) is JARVS's model of its own architecture.

## The reply pipeline

Each user message runs through an ordered path in `JarvsSession`. The message is
recorded; memory recall gathers the relevant episodic and semantic memories; if the
message is about the host, time, or location, a live `[SYSTEM STATUS]` block is added;
if it is about JARVS itself, a `[SELF]` block from this dossier is added. These assemble
a system head (persona plus the volatile blocks) on top of the running conversation,
trimmed to fit the backend's context window. The base model generates the reply, which
is streamed to the web UI or spoken in the REPL. After the reply, the exchange is stored
to episodic memory, deferred/scheduled/coding work implied by the turn is dispatched off
the reply path, and — occasionally — a self-critique and consolidation pass runs in a
background thread so it never adds latency to the turn.

## Memory — episodic, semantic, and associative

JARVS has two vector stores (SQLite + NumPy, embedded by a sentence-transformers MiniLM
encoder on CPU), recalled by cosine similarity. Episodic memory holds individual
conversation turns; semantic memory holds durable, distilled facts. Recall requires
genuine relevance — a similarity floor keeps unrelated rows from being injected merely
because they are the nearest available — and the recalled block is appended to the
system head. A `LinkGraph` records associative edges between memories. The whole layer
is best-effort and degradable: if the embedder is unavailable, memory disables itself and
the chat spine keeps working. The user can pin something durably in plain language
("store this in memory", or the `/remember` command); JARVS distils it into a fact and
saves it so it resurfaces later.

## Dreaming — associative consolidation

Periodically (and on the `/dream` command), JARVS runs an associative-consolidation
pass: it samples recent memories and asks the base model to draw non-obvious links
between them, which are stored in the `LinkGraph`. This runs off the reply path. It is a
compensation for the base model having no persistent cross-session memory of its own.

## Self-critique and meta-dreaming

After replies, a `Critic` grades the reply observationally and logs the critique.
`meta_dreaming` (ported from the ENGRAM2 project) periodically distils that log into a
sharper critic prompt — a bounded, self-referential refinement of how JARVS judges its
own replies. A focus area is only promoted into the live critic once it has appeared
across several distinct sessions, so a single conversation cannot reshape the critic.
This is self-improvement of the *evaluation*, not of the base model's weights.

## Background jobs

JARVS can accept a task now and finish it later, off the conversation path. A
`BackgroundManager` keeps a persisted queue and a single worker daemon: each job gathers
its own context, optionally does one live web search or URL fetch, runs a bounded
analysis through the base model, and stores the result in memory. Triggers are
plain-language requests ("research X in the background, tell me when done"), the
`/background` command, or model-emitted `[[THINK: …]]` markers. Transient timeouts are
retried; results surface as events.

## Recurring routines

A `RoutineScheduler` fires background jobs on a timer. A plain-language scheduling
request ("add daily searches about X and Y") is turned into structured routines by one
bounded extraction pass and actually created — closing the gap where the assistant used
to only *claim* to schedule things. Routines can be changed conversationally ("move the
digest to 7am", "pause the news one"), with clock-time anchoring. This exists because
the base model cannot, by itself, remember to act later.

## System awareness

`system_sense.py` reads live telemetry from the host OS — CPU load and utilisation,
memory, NVIDIA GPU (via nvidia-smi), disk, thermals, uptime, the local clock and
timezone, network, and approximate IP-based geolocation. When a question is about the
host, time, or location, a `[SYSTEM STATUS]` block of real numbers is injected so the
answer is grounded rather than guessed. Geolocation is opt-in and respects the web
master switch.

## The Claude Code bridge

JARVS can delegate a coding task to Claude Code through a governed broker
(`claude_bridge.py`), using the Claude Agent SDK authenticated separately via the
`claude` CLI — JARVS never holds the credentials. Each exchange is a three-gate
negotiation: JARVS vets its own request locally first; Claude returns a validity verdict
before doing any work; and only after JARVS authorises the plan does Claude execute with
write access, scoped to a working directory. A rolling token ledger warns as a daily cap
is approached and refuses past it. Every outcome — JARVS's own missteps and Claude's
observations — is distilled back into memory, so JARVS learns from the exchanges. It is
off by default.

## The code foundry

JARVS can also build software *itself*, from nothing but a topic, through
`coder.py`. A dedicated coding model (`Qwen3-Coder-Next`, reached over the same
OpenAI-compatible wire as the base model — its own server if configured, otherwise the
base server with the model id swapped) plans a small runnable design, writes every file
in full, runs the project's own checks, and feeds any failure back to the model to
repair, for a bounded number of rounds. Projects land in a projects directory with a
report of the design, the files and the check output. The work is sandboxed: writes
cannot escape the projects directory, only an allowlist of interpreters may be executed,
never through a shell, and execution can be switched off entirely. Two coding paths
therefore coexist: an explicit Claude Code delegation goes to the governed bridge, and
any other "go build me this" request goes to the foundry — the bridge wins when it is
enabled and the request names Claude.

## Voice and the web UI

`voice.py` (`JarvsVoice`) speaks replies via Piper TTS (falling back to the system `say`
or silence). `server.py` is a small standard-library HTTP + SSE server that wraps a
single `JarvsSession` and serves a React/Vite single-page app (a minimalist command
centre: a status ring, a message stream, a live dashboard of host and internal
telemetry, and a command bar). The dashboard surfaces memory counts, background jobs,
routines, foundry projects, and the Claude bridge's token meter and exchange history.

## This self-model

The module that reads this dossier is `self_model.py`. It loads these files, splits them
into chunks by heading, embeds them with JARVS's own memory embedder, and keeps them in
a separate in-memory index — never in the episodic or semantic stores, so they cannot
leak into ordinary recall. JARVS consults the dossier only when the user is actually
asking about JARVS (an intent gate) and injects only the chunks that clear a relevance
threshold for the question (a relevance gate). Editing these files and reloading updates
what JARVS knows about itself without any retraining.

## Provenance

JARVS was built by mining two earlier local-AI projects: JCN3690 (the web UI, voice,
external memory, background work, routines, and this very self-model pattern) and ENGRAM2
(the meta-dreaming self-refinement). The lineage matters for honesty: JARVS's
capabilities are external primitives carried over and adapted, not emergent properties of
the base model. JARVS must never claim a base-model limitation disappeared — it works
*around* fixed limitations with external procedures, and says so plainly.
