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


# --- routines (recurring scheduled tasks) ------------------------------------- #

ROUTINES_ENABLED = _flag("JARVS_ROUTINES", True)
# How often the scheduler wakes to check for due routines (seconds).
ROUTINE_TICK = _int("JARVS_ROUTINE_TICK", 30, floor=5)
# Default interval when a request says "repeating"/"recurring" without a period.
ROUTINE_DEFAULT_INTERVAL = _int("JARVS_ROUTINE_DEFAULT_INTERVAL", 86400, floor=60)

# --- voice -------------------------------------------------------------------- #

TTS_BACKEND = _env("JARVS_TTS_BACKEND", "auto")  # auto | piper | say | off


def summary() -> dict:
    """A redacted view of the active config, for /status and smoke tests."""
    return {
        "base_url": BASE_URL,
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "tts_backend": TTS_BACKEND,
        "data_dir": str(DATA_DIR),
        "voices_dir": str(VOICES_DIR),
        "api_key_set": bool(API_KEY and API_KEY != "sk-no-key-required"),
    }
