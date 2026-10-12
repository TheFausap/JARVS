"""Central configuration for JARVS.

All runtime knobs are read from ``JARVS_*`` environment variables and resolved
here, so the rest of the package never touches ``os.environ`` directly. The base
model is treated as a FIXED stage0 reached over an OpenAI-compatible endpoint
(llama.cpp serving the Gemma-4 Opus-Distill GGUF); every other capability is built
as an external primitive around it.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- paths -------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("JARVS_DATA_DIR", ROOT / "data"))
VOICES_DIR = Path(os.environ.get("JARVS_VOICES_DIR", ROOT / "voices"))


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# --- model backend (stage0) --------------------------------------------------- #
# llama.cpp's OpenAI-compatible server; swapping the base model (even to a
# different architecture) means pointing these at a different endpoint, nothing
# more. The default model id matches the DECISIONS block in NOTES.md.

BASE_URL = _env("JARVS_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = _env(
    "JARVS_MODEL",
    "TeichAI/gemma-4-26B-A4B-it-Claude-Opus-Distill-v2-GGUF",
)
API_KEY = _env("JARVS_API_KEY", "sk-no-key-required")  # llama.cpp ignores it

# Embeddings for the external memory stack. Default to CPU: the GPU is for the LLM,
# and a small MiniLM encoder is fast enough on CPU — sharing the GPU risks OOM.
EMBED_MODEL = _env("JARVS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE = _env("JARVS_EMBED_DEVICE", "cpu")


def _int(name: str, default: int, floor: int = 0) -> int:
    try:
        return max(floor, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _flag(name: str, default: bool = True) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() \
        not in ("0", "false", "no", "off")


# --- generation / context budgeting ------------------------------------------- #

REPLY_TIMEOUT = _int("JARVS_REPLY_TIMEOUT", 600, floor=1)
# Stream the SSE response so the read timeout bounds the inter-chunk gap, not total
# wall-clock — a long but progressing completion never trips a spurious timeout.
STREAM_ENABLED = _flag("JARVS_STREAM", True)
# Send enable_thinking:False when the base model has no thinking mode, so a reasoning
# route doesn't just waste tokens on a scratchpad a non-thinking model can't produce.
THINKING_ENABLED = _flag("JARVS_THINKING", True)
# Context slot the backend was launched with, and how much of it to reserve for the
# reply / keep as headroom. Used to trim history before it overflows the slot.
SLOT_CTX = _int("JARVS_SLOT_CTX", 16384, floor=2048)
GEN_RESERVE = _int("JARVS_GEN_RESERVE", 2048, floor=256)
SAFETY_MARGIN = _int("JARVS_CTX_SAFETY_MARGIN", 1024, floor=0)

# Sampling defaults.
TEMPERATURE = float(_env("JARVS_TEMPERATURE", "0.7") or "0.7")


# --- external memory ---------------------------------------------------------- #

MEMORY_ENABLED = _flag("JARVS_MEMORY", True)
# How many memories to pull into the recall block, and the minimum cosine similarity
# a memory needs to be worth injecting (keeps irrelevant recollections out of head).
RECALL_K = _int("JARVS_RECALL_K", 5, floor=1)
RECALL_MIN = float(_env("JARVS_RECALL_MIN", "0.30") or "0.30")


# --- self-critique + meta-dreaming -------------------------------------------- #

# The critic grades each reply (observationally) and logs it; meta-dreaming distills
# the log into a sharper critic prompt. Both are best-effort and off the reply path.
CRITIC_ENABLED = _flag("JARVS_CRITIC", True)
METADREAM_ENABLED = _flag("JARVS_METADREAM", True)
# New critique lines before a meta pass fires, and distinct sessions before a focus
# area is injected into the critic (the anti-contamination promotion bar).
METADREAM_MIN_NEW = _int("JARVS_METADREAM_MIN_NEW", 16, floor=1)
METADREAM_PROMOTE = _int("JARVS_METADREAM_PROMOTE", 3, floor=1)

# Run an associative-consolidation ("dream") pass automatically every N completed
# turns (0 disables; the manual /dream command always works).
DREAM_EVERY = _int("JARVS_DREAM_EVERY", 12, floor=0)


# --- live web access ---------------------------------------------------------- #

# Master switch for any live web access, and whether background jobs may use it.
ALLOW_WEB = _flag("JARVS_ALLOW_WEB", True)
BACKGROUND_WEB = _flag("JARVS_BACKGROUND_WEB", True)
# Live web search on the REPLY path: when a turn needs current info, search and inject a
# [WEB] block before generating, so JARVS verifies instead of answering from its cutoff.
WEB_REPLY = _flag("JARVS_WEB_REPLY", True)
WEB_REPLY_RESULTS = _int("JARVS_WEB_REPLY_RESULTS", 5, floor=1)
WEB_REPLY_CHARS = _int("JARVS_WEB_REPLY_CHARS", 4000, floor=500)


# --- standing rules (durable user-set directives) ----------------------------- #

RULES_ENABLED = _flag("JARVS_RULES", True)


# --- system awareness (hardware / OS / geolocation) --------------------------- #

SYSTEM_SENSE = _flag("JARVS_SYSTEM_SENSE", True)
# IP-based geolocation (the router's public IP → a geo service). Respects ALLOW_WEB.
GEOLOCATION = _flag("JARVS_GEOLOCATION", True)
GEO_TTL = _int("JARVS_GEO_TTL", 21600, floor=60)   # cache seconds (location is stable)
GEO_URL = _env("JARVS_GEO_URL",
               "http://ip-api.com/json/?fields=status,message,city,regionName,"
               "country,lat,lon,timezone,isp,query")


# --- background jobs (autonomous deferred work) ------------------------------- #

BACKGROUND_ENABLED = _flag("JARVS_BACKGROUND", True)
BACKGROUND_MAX_PENDING = _int("JARVS_BACKGROUND_MAX_PENDING", 4, floor=1)
BACKGROUND_RETRIES = _int("JARVS_BACKGROUND_RETRIES", 1, floor=0)
BACKGROUND_RETRY_BACKOFF = _int("JARVS_BACKGROUND_RETRY_BACKOFF", 3, floor=0)
BACKGROUND_CONTEXT_CHARS = _int("JARVS_BACKGROUND_CONTEXT_CHARS", 6000, floor=500)


# --- autonomous coding (the foundry) ------------------------------------------ #
# A SECOND, specialist backend for code: Qwen3-Coder-Next over the same
# OpenAI-compatible wire protocol. Leaving JARVS_CODER_BASE_URL empty reuses the
# base model's endpoint and swaps only the model id; pointing it elsewhere gives the
# coder its own server (and therefore its own context slot and timeout budget).
# When the coder endpoint is unreachable the foundry falls back to the base backend
# so a build request degrades to a slower coder instead of failing outright.
CODER_ENABLED = _flag("JARVS_CODER", True)
CODER_BASE_URL = _env("JARVS_CODER_BASE_URL", "")
CODER_MODEL = _env("JARVS_CODER_MODEL", "Qwen3-Coder-Next")
CODER_API_KEY = _env("JARVS_CODER_API_KEY", "")
CODER_TIMEOUT = _int("JARVS_CODER_TIMEOUT", 600, floor=30)
# Code files are long; the reserve that bounds a chat reply would truncate them.
CODER_MAX_TOKENS = _int("JARVS_CODER_MAX_TOKENS", 8192, floor=1024)
CODER_TEMPERATURE = float(_env("JARVS_CODER_TEMPERATURE", "0.2") or "0.2")
PROJECTS_DIR = Path(_env("JARVS_PROJECTS_DIR", str(DATA_DIR / "projects")))

# Bounds on one build: how much it may write and how hard it may try to fix itself.
CODER_MAX_FILES = _int("JARVS_CODER_MAX_FILES", 8, floor=1)
CODER_MAX_FILE_CHARS = _int("JARVS_CODER_MAX_FILE_CHARS", 60000, floor=2000)
CODER_MAX_REPAIRS = _int("JARVS_CODER_MAX_REPAIRS", 2, floor=0)
CODER_MAX_FIXES = _int("JARVS_CODER_MAX_FIXES", 3, floor=1)
CODER_MAX_PENDING = _int("JARVS_CODER_MAX_PENDING", 2, floor=1)

# Verification: run the project's own checks after writing it. JARVS_CODER_EXEC=0
# disables all subprocess execution (the foundry then only writes files).
CODER_EXEC = _flag("JARVS_CODER_EXEC", True)
CODER_TEST_TIMEOUT = _int("JARVS_CODER_TEST_TIMEOUT", 180, floor=10)

# --- the duet: Qwen and Claude coding together -------------------------------- #
# When the Claude bridge is ON, a foundry build stops being Qwen alone and becomes a
# bounded dialogue: Qwen3-Coder-Next plans, writes and repairs (local, unlimited),
# while Claude reviews the design, diagnoses stubborn failures, and critiques the
# finished project — read-only, through the bridge's own gates and token ledger.
# Cheap local tokens do the bulk; expensive Claude tokens go where they change the
# outcome. Off (or bridge off) => Qwen builds alone, exactly as before.
CODER_COLLAB = _flag("JARVS_CODER_COLLAB", True)
# How many Claude consultations ONE build may spend (plan review, failure diagnosis
# and the final critique all draw on this budget).
CODER_COLLAB_ROUNDS = _int("JARVS_CODER_COLLAB_ROUNDS", 2, floor=0)
# Review the design BEFORE a file is written — the cheapest possible moment to be
# told the plan won't work.
CODER_COLLAB_PLAN = _flag("JARVS_CODER_COLLAB_PLAN", True)


# --- routines (recurring scheduled tasks) ------------------------------------- #

ROUTINES_ENABLED = _flag("JARVS_ROUTINES", True)
# How often the scheduler wakes to check for due routines (seconds).
ROUTINE_TICK = _int("JARVS_ROUTINE_TICK", 30, floor=5)
# Default interval when a request says "repeating"/"recurring" without a period.
ROUTINE_DEFAULT_INTERVAL = _int("JARVS_ROUTINE_DEFAULT_INTERVAL", 86400, floor=60)

# --- Claude Code bridge (delegate coding to Claude, via the Agent SDK) --------- #
# Off by default. When on, coding requests are routed to Claude Code through the
# `claude-agent-sdk` (authenticated separately via `claude login` — JARVS never holds
# the credentials). Governance is on tokens, not dollars: a rolling 24h ledger warns
# as the cap is approached and refuses past it.

CLAUDE_BRIDGE_ENABLED = _flag("JARVS_CLAUDE_BRIDGE", False)
# Working directory Claude is scoped to (defaults to the repo root).
CLAUDE_BRIDGE_CWD = _env("JARVS_CLAUDE_BRIDGE_CWD", str(ROOT))
# Max agent turns per Claude call (caps how long one exchange can run).
CLAUDE_MAX_TURNS = _int("JARVS_CLAUDE_MAX_TURNS", 4, floor=1)
# Rolling-24h token cap across all Claude calls (0 == unlimited), and the fraction
# of it at which JARVS starts warning before a call.
CLAUDE_TOKEN_DAILY_CAP = _int("JARVS_CLAUDE_TOKEN_DAILY_CAP", 200_000, floor=0)
try:
    CLAUDE_TOKEN_WARN_AT = min(1.0, max(0.0, float(
        _env("JARVS_CLAUDE_TOKEN_WARN_AT", "0.8") or "0.8")))
except ValueError:
    CLAUDE_TOKEN_WARN_AT = 0.8
# Optional model override for Claude (empty == the SDK/CLI default).
CLAUDE_MODEL = _env("JARVS_CLAUDE_MODEL", "")
# Refuse to run if ANTHROPIC_API_KEY is set, so the subscription is used (not the
# paid API). Set to 0 only if you deliberately want API-key billing.
CLAUDE_REQUIRE_SUBSCRIPTION = _flag("JARVS_CLAUDE_REQUIRE_SUBSCRIPTION", True)


# --- self-model (JARVS's dossier about its own architecture) ------------------ #
# A curated set of docs describing how JARVS is built, embedded and cosine-searched,
# kept OUTSIDE episodic/semantic memory so it never leaks into ordinary recall. It is
# consulted only when the user asks JARVS about itself (intent gate), and only chunks
# above the relevance floor are injected (relevance gate). External self-documentation,
# not introspection of the model's weights.

SELF_MODEL_ENABLED = _flag("JARVS_SELF_MODEL", True)
SELF_DIR = Path(os.environ.get("JARVS_SELF_DIR", ROOT / "self"))
SELF_RECALL_K = _int("JARVS_SELF_RECALL_K", 4, floor=1)
# Floor is modest on purpose: the intent gate already confirms the user is asking about
# JARVS, so this only ranks/trims dossier chunks (calibrated against the MiniLM encoder,
# where a relevant self-question scores ~0.18–0.52 and an unrelated one stays < ~0.14).
try:
    SELF_RECALL_MIN = float(_env("JARVS_SELF_RECALL_MIN", "0.15") or "0.15")
except ValueError:
    SELF_RECALL_MIN = 0.15


# --- voice -------------------------------------------------------------------- #

TTS_BACKEND = _env("JARVS_TTS_BACKEND", "auto")  # auto | piper | say | off


def summary() -> dict:
    """A redacted view of the active config, for /status and smoke tests."""
    return {
        "base_url": BASE_URL,
        "model": MODEL,
        "coder_base_url": CODER_BASE_URL or BASE_URL,
        "coder_model": CODER_MODEL,
        "projects_dir": str(PROJECTS_DIR),
        "embed_model": EMBED_MODEL,
        "tts_backend": TTS_BACKEND,
        "data_dir": str(DATA_DIR),
        "voices_dir": str(VOICES_DIR),
        "api_key_set": bool(API_KEY and API_KEY != "sk-no-key-required"),
    }
