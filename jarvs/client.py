"""
client.py — the OpenAI-compatible backend client (JARVS stage0).

The base model is a FIXED foundation reached over an OpenAI-compatible endpoint
(llama.cpp serving the Gemma-4 Opus-Distill GGUF). This module is the ONLY place
that talks HTTP to it; everything else calls :func:`generate` /
:func:`generate_with_history`.

Adapted from JCN3690's ``9chat.research.py`` chat layer, trimmed to the pieces the
JARVS spine needs: a single ``chat_completion`` with an SSE-streaming path (so the
read timeout bounds the inter-chunk gap, not total generation time), reasoning-chain
stripping, and token counting via the backend's own ``/tokenize`` when it has one.
"""

from __future__ import annotations

import functools
import json
import re

import requests

from . import config

# --- reasoning-chain stripping ------------------------------------------------ #

_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?\s*>", re.IGNORECASE | re.DOTALL)
_DANGLING_THINK_RE = re.compile(
    r"^.*?</think(?:ing)?\s*>", re.IGNORECASE | re.DOTALL)
_OPEN_THINK_RE = re.compile(r"^\s*<think(?:ing)?\b[^>]*>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Strip leaked model chain-of-thought from visible/parsed content."""
    if not text:
        return text or ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    low = cleaned.lower()
    # A closing tag with no surviving opener means the opener was consumed upstream:
    # keep only what follows that final close.
    if "</think" in low and "<think" not in low:
        cleaned = _DANGLING_THINK_RE.sub("", cleaned, count=1)
    # A leading opener with no close is an all-reasoning completion; drop it rather
    # than surfacing the scratchpad. (No close means no answer followed.)
    elif _OPEN_THINK_RE.match(cleaned) and "</think" not in low:
        cleaned = ""
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# --- backend plumbing --------------------------------------------------------- #

_RESOLVED_MODEL: str | None = None
# Set on every generate* call to the real reason it failed, so a caller can show WHY
# instead of a generic "backend error" (socket timeout, HTTP 500, context overflow,
# or an empty reply where the reasoning chain ate the budget all look alike otherwise).
LAST_GENERATION_ERROR = ""


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if config.API_KEY:
        headers["Authorization"] = f"Bearer {config.API_KEY}"
    return headers


def resolve_served_model() -> str:
    """The model id to send. A concrete JARVS_MODEL is used verbatim; "auto" asks the
    backend what it is serving and takes the first id."""
    global _RESOLVED_MODEL
    if _RESOLVED_MODEL:
        return _RESOLVED_MODEL
    if config.MODEL and config.MODEL != "auto":
        _RESOLVED_MODEL = config.MODEL
        return _RESOLVED_MODEL
    response = requests.get(f"{config.BASE_URL}/models", headers=_headers(), timeout=30)
    response.raise_for_status()
    models = [m.get("id") for m in response.json().get("data", []) if m.get("id")]
    if not models:
        raise RuntimeError(f"No models served at {config.BASE_URL}/models")
    _RESOLVED_MODEL = models[0]
    return _RESOLVED_MODEL


def chat_completion(messages: list[dict], *, temperature: float | None = None,
                    timeout: int | None = None, model: str = "",
                    max_tokens: int | None = None, reason: bool = True,
                    stream: bool | None = None) -> dict:
    payload = {
        "model": model or resolve_served_model(),
        "messages": messages,
        "temperature": config.TEMPERATURE if temperature is None else temperature,
        "stream": bool(config.STREAM_ENABLED if stream is None else stream),
    }
    # Disable the thinking scratchpad when the caller doesn't want it, OR globally
    # when the backend model has no thinking mode — asking a non-thinking model to
    # think just yields a normal answer and never a <think>.
    if not reason or not config.THINKING_ENABLED:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    url = f"{config.BASE_URL.rstrip('/')}/chat/completions"
    timeout = config.REPLY_TIMEOUT if timeout is None else timeout
    if not payload["stream"]:
        response = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
        response.raise_for_status()
        return response.json()
    return _chat_completion_stream(url, payload, timeout)


def _chat_completion_stream(url: str, payload: dict, timeout: int) -> dict:
    """Consume an OpenAI-style SSE stream and rebuild a NON-streaming-shaped response,
    so the parse helpers below are unchanged. The point is the timeout semantics: the
    read timeout bounds the gap between chunks, not the whole generation."""
    payload = {**payload, "stream": True,
               # vLLM only emits a usage chunk when asked; harmless to llama.cpp.
               "stream_options": {"include_usage": True}}
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason, usage = "", {}
    with requests.post(url, json=payload, headers=_headers(), timeout=timeout,
                       stream=True) as response:
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if not line or line == "[DONE]":
                if line == "[DONE]":
                    break
                continue
            try:
                chunk = json.loads(line)
            except (ValueError, TypeError):
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0] or {}
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    message = {"role": "assistant", "content": "".join(content_parts)}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    return {"choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage}


# --- response parsing --------------------------------------------------------- #

def _first_choice(data: dict) -> dict:
    try:
        return data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return {}


def _reply_text(data: dict) -> str:
    return ((_first_choice(data).get("message") or {}).get("content") or "").strip()


def _finish_reason(data: dict) -> str:
    return _first_choice(data).get("finish_reason") or ""


def _prompt_tokens(data: dict) -> int:
    n = (data.get("usage") or {}).get("prompt_tokens")
    return int(n) if isinstance(n, int) else 0


# --- high-level generation ---------------------------------------------------- #

def generate(system_msg: str, user_input: str, reason: bool = True) -> str:
    """Single-shot system+user generation. Signature matches what meta-dreaming and
    the other bookkeeping passes expect: ``generate(system, user, reason) -> str``.
    Never raises — returns an error string the caller can surface or log."""
    try:
        data = chat_completion(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_input}],
            temperature=0.6, reason=reason,
        )
        return strip_reasoning(_reply_text(data))
    except Exception as e:  # noqa: BLE001 — a bookkeeping call must not crash a turn
        return f"Error connecting to the backend: {e}"


def generate_with_history(system_msg: str, messages: list[dict],
                          reason: bool = False,
                          temperature: float | None = None) -> str:
    """The user-facing reply path: a system head plus the running conversation.

    The hidden reasoning chain still counts against ``max_tokens``. We cap the reply
    at GEN_RESERVE so it can't overflow the slot, but if reasoning fills that cap the
    visible content comes back empty with finish_reason "length"; detect exactly that
    and retry once with a cap sized to the slot's real headroom rather than failing.
    """
    global LAST_GENERATION_ERROR
    LAST_GENERATION_ERROR = ""
    try:
        full = [{"role": "system", "content": system_msg}] + messages
        data = chat_completion(full, temperature=temperature,
                               max_tokens=config.GEN_RESERVE, reason=reason)
        reply = strip_reasoning(_reply_text(data))
        if not reply and _finish_reason(data) == "length":
            headroom = config.SLOT_CTX - _prompt_tokens(data) - config.SAFETY_MARGIN
            retry_cap = min(max(headroom, config.GEN_RESERVE), config.GEN_RESERVE * 4)
            if retry_cap > config.GEN_RESERVE:
                data = chat_completion(full, temperature=temperature,
                                       max_tokens=retry_cap, reason=reason)
                reply = strip_reasoning(_reply_text(data))
        if not reply.strip():
            LAST_GENERATION_ERROR = (
                "empty reply (the reasoning chain likely consumed the token budget)")
        return reply
    except requests.Timeout as e:
        LAST_GENERATION_ERROR = f"backend timeout after {config.REPLY_TIMEOUT}s: {e}"
    except requests.RequestException as e:
        LAST_GENERATION_ERROR = f"backend request failed: {e}"
    except Exception as e:  # noqa: BLE001
        LAST_GENERATION_ERROR = f"{type(e).__name__}: {e}"
    return ""


# --- token counting (for context-window budgeting) ---------------------------- #
# A chars/4 estimate under-counts real tokens on prose full of names/markdown, which
# lets the prompt overflow the slot and truncate the reply. Use the backend's own
# tokenizer when it exposes POST /tokenize (llama.cpp / vLLM at the server root);
# fall back to a *conservative* char estimate so we trim early rather than overflow.

_TOKENIZE_OK = True
_TOKENIZE_URL: str | None = None


def _tokenize_url() -> str:
    global _TOKENIZE_URL
    if _TOKENIZE_URL is None:
        root = config.BASE_URL.rsplit("/v1", 1)[0] if "/v1" in config.BASE_URL \
            else config.BASE_URL.rstrip("/")
        _TOKENIZE_URL = root.rstrip("/") + "/tokenize"
    return _TOKENIZE_URL


@functools.lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    if not text:
        return 0
    global _TOKENIZE_OK
    if _TOKENIZE_OK:
        try:
            r = requests.post(_tokenize_url(),
                              json={"content": text, "add_special": False},
                              headers=_headers(), timeout=30)
            r.raise_for_status()
            toks = r.json().get("tokens")
            if isinstance(toks, list):
                return len(toks)
        except Exception:
            _TOKENIZE_OK = False  # backend has no /tokenize; stop trying, use est
    # Conservative fallback: ~3.2 chars/token over-estimates vs the usual ~4, so we
    # trim a little early instead of overflowing the slot.
    return int(len(text) / 3.2) + 1


def health() -> tuple[bool, str]:
    """Best-effort reachability probe for /status and startup. Returns (ok, detail)."""
    try:
        model = resolve_served_model()
        return True, f"backend reachable · serving {model}"
    except Exception as e:  # noqa: BLE001
        return False, f"backend unreachable at {config.BASE_URL}: {e}"
