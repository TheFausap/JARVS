"""
client.py — the OpenAI-compatible backend client (JARVS stage0).

The base model is a FIXED foundation reached over an OpenAI-compatible endpoint
(llama.cpp serving the Qwen3.8-27B-Turbo-Brilliance GGUF). This module is the ONLY
place that talks HTTP to it; everything else calls :func:`generate` /
:func:`generate_with_history` / :func:`stream_chat`.

Adapted from JCN3690's ``9chat.research.py`` chat layer and ENGRAM2's
``turbo_brilliance`` integration: a single ``chat_completion`` with an SSE-streaming
path (so the read timeout bounds the inter-chunk gap, not total generation time),
task-specific ``{REASON:...}`` cognitive routing, ``cache_prompt`` management,
reasoning-chain and construct-border stripping, and token counting via the backend's
own ``/tokenize`` when it has one.
"""

from __future__ import annotations

import functools
import json
import os
import re
from typing import Any

import requests

from . import config, turbo_brilliance

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


# --- Turbo-Brilliance & {REASON:...} state ------------------------------------ #

_BACKEND_PROPS_FETCHED = False
_CACHED_BACKEND_PROPS: dict | None = None
TURBO_BRILLIANCE_DETECTED = False
SUPPORTED_REASON_MODES: tuple[str, ...] = turbo_brilliance.DEFAULT_REASON_MODES
TEACHER_CONSTRUCT_INSTRUCTION: str = turbo_brilliance.FALLBACK_TEACHER_INSTRUCTION


def _server_root_url() -> str:
    base = config.BASE_URL.rstrip("/")
    return base.rsplit("/v1", 1)[0] if base.endswith("/v1") else base


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if config.API_KEY:
        headers["Authorization"] = f"Bearer {config.API_KEY}"
    return headers


def _fetch_backend_props(force: bool = False) -> dict | None:
    """Fetch and cache llama.cpp ``GET /props`` (slot context + chat_template)."""
    global _BACKEND_PROPS_FETCHED, _CACHED_BACKEND_PROPS
    if _BACKEND_PROPS_FETCHED and not force:
        return _CACHED_BACKEND_PROPS
    _BACKEND_PROPS_FETCHED = True
    try:
        r = requests.get(f"{_server_root_url()}/props", headers=_headers(), timeout=3)
        if r.ok:
            data = r.json()
            if isinstance(data, dict):
                _CACHED_BACKEND_PROPS = data
                return data
    except Exception:  # noqa: BLE001
        pass
    return _CACHED_BACKEND_PROPS


def _detect_reason_template(force_probe: bool = False) -> tuple[bool, str]:
    """Determine whether ``{REASON:...}`` tag injection should be active.

    Priority:
    1. Explicit ``JARVS_REASON_TAGS=1/true/on`` or ``0/false/off``.
    2. Auto-detection from ``GET /props`` ``chat_template`` when probed.
    3. Default heuristic: enabled automatically when ``config.MODEL`` names a
       ``Turbo-Brilliance`` GGUF, otherwise ``False`` until probed.
    """
    global TURBO_BRILLIANCE_DETECTED, SUPPORTED_REASON_MODES, TEACHER_CONSTRUCT_INSTRUCTION
    raw_env = (config.REASON_TAGS or "auto").strip().lower()
    props = _CACHED_BACKEND_PROPS
    if props is None and force_probe:
        props = _fetch_backend_props(force=True)
    if isinstance(props, dict) and props.get("chat_template"):
        info = turbo_brilliance.inspect_chat_template(props.get("chat_template", ""))
        TURBO_BRILLIANCE_DETECTED = bool(info["detected"])
        SUPPORTED_REASON_MODES = info["modes"]
        if info["teacher_instruction"]:
            TEACHER_CONSTRUCT_INSTRUCTION = info["teacher_instruction"]
    if raw_env in ("1", "true", "yes", "on"):
        return True, "env"
    if raw_env in ("0", "false", "no", "off"):
        return False, "env"
    if TURBO_BRILLIANCE_DETECTED:
        return True, "auto (llama.cpp /props)"
    if "turbo-brilliance" in (config.MODEL or "").lower():
        return True, "auto (model name)"
    return False, "default (off)"


REASON_TAGS_ON, REASON_TAGS_SOURCE = _detect_reason_template(force_probe=False)

_INITIAL_PRESET_NAME, _INITIAL_PRESET = turbo_brilliance.resolve_reason_preset(
    config.REASON_PRESET or "default"
)
_INITIAL_TASKS = (
    _INITIAL_PRESET["tasks"]
    if _INITIAL_PRESET is not None
    else turbo_brilliance.REASON_PRESETS["default"]["tasks"]
)

REASON_TAG_MAIN = turbo_brilliance.format_reason_tag(
    config.REASON_TAG_MAIN or _INITIAL_TASKS["main"]
)
REASON_TAG_BACKGROUND = turbo_brilliance.format_reason_tag(
    config.REASON_TAG_BACKGROUND or _INITIAL_TASKS["background"]
)
REASON_TAG_CRITIC = turbo_brilliance.normalize_instruct_reason_tag(
    config.REASON_TAG_CRITIC or _INITIAL_TASKS["critic"]
)
REASON_TAG_DREAM = turbo_brilliance.normalize_instruct_reason_tag(
    config.REASON_TAG_DREAM or _INITIAL_TASKS["dream"]
)
REASON_TAG_CLAUDE = turbo_brilliance.normalize_instruct_reason_tag(
    config.REASON_TAG_CLAUDE or _INITIAL_TASKS["claude"]
)
REASON_TAG_ROUTER = turbo_brilliance.normalize_instruct_reason_tag(
    config.REASON_TAG_ROUTER or _INITIAL_TASKS["router"]
)
REASON_TAG_OFF = turbo_brilliance.normalize_instruct_reason_tag(
    config.REASON_TAG_OFF or _INITIAL_TASKS["off"]
)


def _normalize_cache_prompt_mode(raw: str) -> str:
    val = (raw or "auto").strip().lower()
    if val in ("0", "false", "no", "off"):
        return "off"
    if val in ("1", "true", "yes", "on"):
        return "on"
    return "auto"


CACHE_PROMPT_MODE = _normalize_cache_prompt_mode(config.CACHE_PROMPT)
REASON_PANEL_DETAIL = bool(config.REASON_PANEL_DETAIL)

ACTIVE_CONSTRUCT = (
    config.CONSTRUCT if config.CONSTRUCT in turbo_brilliance.CONSTRUCT_NAMES else ""
)
CONSTRUCT_BLEND = bool(config.CONSTRUCT_BLEND)
CONSTRUCT_INSTRUCT = bool(config.CONSTRUCT_INSTRUCT)
HOLODECK_ON = bool(config.HOLODECK)
HOLODECK_INSTRUCT = bool(config.HOLODECK_INSTRUCT)


def is_pure_construct_or_holodeck() -> bool:
    """True when JARVS is acting as a pure AI Construct persona or Star Trek
    Holodeck simulation rather than blending with JARVS's assistant identity."""
    return bool(HOLODECK_ON or (ACTIVE_CONSTRUCT and not CONSTRUCT_BLEND))


def reason_tag_for_task(
    task: str = "main",
    reason: bool | None = None,
    reason_tag: str | None = None,
) -> str:
    """Resolve the effective ``{REASON:...}`` tag for a JARVS pipeline ``task``.

    Tasks: ``'main'``, ``'background'``, ``'critic'``, ``'dream'``, ``'claude'``,
    ``'router'``, ``'off'``.
    """
    if reason_tag is not None and reason_tag.strip():
        return (
            turbo_brilliance.normalize_instruct_reason_tag(reason_tag)
            if reason is False
            else turbo_brilliance.format_reason_tag(reason_tag)
        )
    if reason is None:
        reason = config.THINKING_ENABLED
    t = (task or ("main" if reason else "off")).strip().lower()

    if t == "main":
        if HOLODECK_ON:
            use_instruct = HOLODECK_INSTRUCT or not reason
            return "{REASON:iholodeck}" if use_instruct else "{REASON:holodeck}"
        if ACTIVE_CONSTRUCT:
            use_instruct = CONSTRUCT_INSTRUCT or not reason
            prefix = "i" if use_instruct and not ACTIVE_CONSTRUCT.startswith("x") else ""
            return f"{{REASON:{prefix}{ACTIVE_CONSTRUCT}}}"
        if not reason:
            return turbo_brilliance.normalize_instruct_reason_tag(REASON_TAG_OFF)
        return turbo_brilliance.format_reason_tag(REASON_TAG_MAIN)

    if t == "background":
        if not reason:
            return turbo_brilliance.normalize_instruct_reason_tag(REASON_TAG_OFF)
        return turbo_brilliance.format_reason_tag(REASON_TAG_BACKGROUND or REASON_TAG_MAIN)

    if t in ("critic", "critique"):
        return turbo_brilliance.normalize_instruct_reason_tag(
            REASON_TAG_CRITIC or REASON_TAG_OFF
        )
    if t == "dream":
        return turbo_brilliance.normalize_instruct_reason_tag(
            REASON_TAG_DREAM or REASON_TAG_OFF
        )
    if t == "claude":
        return turbo_brilliance.normalize_instruct_reason_tag(
            REASON_TAG_CLAUDE or REASON_TAG_OFF
        )
    if t == "router":
        return turbo_brilliance.normalize_instruct_reason_tag(
            REASON_TAG_ROUTER or REASON_TAG_OFF
        )
    return turbo_brilliance.normalize_instruct_reason_tag(REASON_TAG_OFF)


def active_reason_preset() -> str:
    """Return the name of the matching preset in ``turbo_brilliance.REASON_PRESETS``
    if the current 7 task tags match one, or ``'custom'`` otherwise."""
    current = {
        "main": REASON_TAG_MAIN,
        "background": REASON_TAG_BACKGROUND,
        "critic": REASON_TAG_CRITIC,
        "dream": REASON_TAG_DREAM,
        "claude": REASON_TAG_CLAUDE,
        "router": REASON_TAG_ROUTER,
        "off": REASON_TAG_OFF,
    }
    for name, info in turbo_brilliance.REASON_PRESETS.items():
        if info.get("tasks") == current:
            return name
    return "custom"


def _apply_teacher_workaround(messages: list[dict], tag: str) -> None:
    """In DavidAU's GGUF chat_template, `'teacher': system_instruction__DAU` has
    a double-underscore typo that evaluates to undefined/empty. When `{REASON:teacher}`
    or `{REASON:iteacher}` is used and not already injected by `ACTIVE_CONSTRUCT`,
    prepend `TEACHER_CONSTRUCT_INSTRUCTION` to the system message in-place."""
    base_mode, _ = turbo_brilliance.parse_reason_tag_mode(tag)
    if base_mode != "teacher" or not TEACHER_CONSTRUCT_INSTRUCTION:
        return
    marker = "ALEXANDER PEDAGOGICAL SYSTEM"
    if messages and messages[0].get("role") == "system":
        sys_txt = messages[0].get("content") or ""
        if marker not in sys_txt and TEACHER_CONSTRUCT_INSTRUCTION not in sys_txt:
            messages[0]["content"] = (
                f"{TEACHER_CONSTRUCT_INSTRUCTION}\n\n{sys_txt}"
                if sys_txt else TEACHER_CONSTRUCT_INSTRUCTION
            )
    else:
        messages.insert(0, {"role": "system", "content": TEACHER_CONSTRUCT_INSTRUCTION})


def _with_reason_tag(
    messages: list[dict],
    reason: bool = True,
    *,
    task: str = "",
    reason_tag: str | None = None,
) -> list[dict]:
    """Prepend the resolved ``{REASON:...}`` tag to the last user message."""
    t_norm = (task or "").strip().lower()
    allow_panel_suffix = bool(
        REASON_PANEL_DETAIL
        and t_norm not in ("critic", "critique", "dream", "claude", "router", "off")
    )
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            existing = messages[idx].get("content") or ""
            inline_tag, cleaned = turbo_brilliance.extract_inline_reason_tag(existing)
            if inline_tag:
                out = [dict(m) for m in messages]
                if allow_panel_suffix:
                    suffix = turbo_brilliance.panel_detail_suffix(inline_tag)
                    if suffix and suffix.lower() not in cleaned.lower():
                        cleaned = f"{cleaned}\n\n{suffix}".strip() if cleaned else suffix
                out[idx]["content"] = f"{inline_tag}\n{cleaned}" if cleaned else inline_tag
                _apply_teacher_workaround(out, inline_tag)
                return out
            break

    active = bool(REASON_TAGS_ON or ACTIVE_CONSTRUCT or HOLODECK_ON or reason_tag)
    if not active:
        return messages
    tag = reason_tag_for_task(task=task, reason=reason, reason_tag=reason_tag)
    if not tag:
        return messages
    out = [dict(m) for m in messages]
    _apply_teacher_workaround(out, tag)
    for m in reversed(out):
        if m.get("role") == "user":
            existing = m.get("content") or ""
            if allow_panel_suffix:
                suffix = turbo_brilliance.panel_detail_suffix(tag)
                if suffix and suffix.lower() not in existing.lower():
                    existing = f"{existing}\n\n{suffix}".strip() if existing else suffix
            m["content"] = f"{tag}\n{existing}" if existing else tag
            return out
    out.append({"role": "user", "content": tag})
    return out


def _should_disable_thinking_kwarg(
    original_messages: list[dict],
    tagged_messages: list[dict],
    reason: bool,
) -> bool:
    """Determine whether ``chat_template_kwargs={"enable_thinking": False}``
    should be sent on the request payload."""
    if not config.THINKING_ENABLED:
        return True
    if tagged_messages is not original_messages:
        for m in reversed(tagged_messages):
            if m.get("role") == "user":
                tag, _ = turbo_brilliance.extract_inline_reason_tag(m.get("content") or "")
                if tag:
                    return turbo_brilliance.is_instruct_reason_tag(tag)
                break
    return not reason


def _resolve_cache_prompt_kwarg(
    original_messages: list[dict],
    tagged_messages: list[dict],
    task: str = "",
) -> bool | None:
    """Determine whether ``cache_prompt`` should be explicitly sent in the payload."""
    if CACHE_PROMPT_MODE == "off":
        return False
    if CACHE_PROMPT_MODE == "on":
        return True
    if tagged_messages is not original_messages:
        for m in reversed(tagged_messages):
            if m.get("role") == "user":
                tag, _ = turbo_brilliance.extract_inline_reason_tag(m.get("content") or "")
                if tag and turbo_brilliance.tag_modifies_system_prompt(tag):
                    t = (task or "").strip().lower()
                    if t in ("critic", "critique", "dream", "claude", "router", "off") or len(original_messages) <= 2:
                        return False
                break
    return None


def saved_reason_state() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of active Turbo-Brilliance settings."""
    return {
        "enabled": REASON_TAGS_ON,
        "preset": active_reason_preset(),
        "main": REASON_TAG_MAIN,
        "background": REASON_TAG_BACKGROUND,
        "critic": REASON_TAG_CRITIC,
        "dream": REASON_TAG_DREAM,
        "claude": REASON_TAG_CLAUDE,
        "router": REASON_TAG_ROUTER,
        "off": REASON_TAG_OFF,
        "cache_prompt": CACHE_PROMPT_MODE,
        "panel_detail": REASON_PANEL_DETAIL,
        "construct": ACTIVE_CONSTRUCT,
        "construct_blend": CONSTRUCT_BLEND,
        "construct_instruct": CONSTRUCT_INSTRUCT,
        "holodeck": HOLODECK_ON,
        "holodeck_instruct": HOLODECK_INSTRUCT,
    }


def save_reason_state(path: str | None = None) -> None:
    """Persist the current Turbo-Brilliance state to ``data/reason_state.json``."""
    target = path or os.path.join(str(config.DATA_DIR), "reason_state.json")
    try:
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        tmp = f"{target}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(saved_reason_state(), fh, indent=2)
        os.replace(tmp, target)
    except OSError:
        pass


def load_reason_state(path: str | None = None) -> bool:
    """Restore persisted Turbo-Brilliance state if ``data/reason_state.json`` exists."""
    global REASON_TAGS_ON, REASON_TAGS_SOURCE
    global REASON_TAG_MAIN, REASON_TAG_BACKGROUND, REASON_TAG_CRITIC
    global REASON_TAG_DREAM, REASON_TAG_CLAUDE, REASON_TAG_ROUTER, REASON_TAG_OFF
    global CACHE_PROMPT_MODE, REASON_PANEL_DETAIL
    global ACTIVE_CONSTRUCT, CONSTRUCT_BLEND, CONSTRUCT_INSTRUCT
    global HOLODECK_ON, HOLODECK_INSTRUCT
    target = path or os.path.join(str(config.DATA_DIR), "reason_state.json")
    try:
        with open(target, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not isinstance(saved, dict):
        return False
    if isinstance(saved.get("enabled"), bool):
        REASON_TAGS_ON = saved["enabled"]
        REASON_TAGS_SOURCE = "saved state"
    if isinstance(saved.get("main"), str) and saved["main"].strip():
        REASON_TAG_MAIN = turbo_brilliance.format_reason_tag(saved["main"])
    if isinstance(saved.get("background"), str) and saved["background"].strip():
        REASON_TAG_BACKGROUND = turbo_brilliance.format_reason_tag(saved["background"])
    if isinstance(saved.get("critic"), str) and saved["critic"].strip():
        REASON_TAG_CRITIC = turbo_brilliance.normalize_instruct_reason_tag(saved["critic"])
    if isinstance(saved.get("dream"), str) and saved["dream"].strip():
        REASON_TAG_DREAM = turbo_brilliance.normalize_instruct_reason_tag(saved["dream"])
    if isinstance(saved.get("claude"), str) and saved["claude"].strip():
        REASON_TAG_CLAUDE = turbo_brilliance.normalize_instruct_reason_tag(saved["claude"])
    if isinstance(saved.get("router"), str) and saved["router"].strip():
        REASON_TAG_ROUTER = turbo_brilliance.normalize_instruct_reason_tag(saved["router"])
    if isinstance(saved.get("off"), str) and saved["off"].strip():
        REASON_TAG_OFF = turbo_brilliance.normalize_instruct_reason_tag(saved["off"])
    if isinstance(saved.get("cache_prompt"), str) and saved["cache_prompt"].strip():
        CACHE_PROMPT_MODE = _normalize_cache_prompt_mode(saved["cache_prompt"])
    if isinstance(saved.get("panel_detail"), bool):
        REASON_PANEL_DETAIL = saved["panel_detail"]
    if isinstance(saved.get("construct"), str):
        c = saved["construct"].strip().lower()
        if not c or c in turbo_brilliance.CONSTRUCT_NAMES:
            ACTIVE_CONSTRUCT = c
    if isinstance(saved.get("construct_blend"), bool):
        CONSTRUCT_BLEND = saved["construct_blend"]
    if isinstance(saved.get("construct_instruct"), bool):
        CONSTRUCT_INSTRUCT = saved["construct_instruct"]
    if isinstance(saved.get("holodeck"), bool):
        HOLODECK_ON = saved["holodeck"]
    if isinstance(saved.get("holodeck_instruct"), bool):
        HOLODECK_INSTRUCT = saved["holodeck_instruct"]
    return True


# --- backend plumbing --------------------------------------------------------- #

_RESOLVED_MODEL: str | None = None
# Set on every generate* call to the real reason it failed, so a caller can show WHY
# instead of a generic "backend error" (socket timeout, HTTP 500, context overflow,
# or an empty reply where the reasoning chain ate the budget all look alike otherwise).
LAST_GENERATION_ERROR = ""


def resolve_served_model() -> str:
    """The model id to send. A concrete JARVS_MODEL is used verbatim; "auto" asks the
    backend what it is serving and takes the first id."""
    global _RESOLVED_MODEL, REASON_TAGS_ON, REASON_TAGS_SOURCE
    if not _BACKEND_PROPS_FETCHED and (config.REASON_TAGS or "auto").strip().lower() == "auto":
        detected_on, detected_src = _detect_reason_template(force_probe=True)
        if detected_on and not REASON_TAGS_ON:
            REASON_TAGS_ON = True
            REASON_TAGS_SOURCE = detected_src
    if _RESOLVED_MODEL:
        return _RESOLVED_MODEL
    if config.MODEL and config.MODEL != "auto":
        _RESOLVED_MODEL = config.MODEL
        return _RESOLVED_MODEL
    response = requests.get(f"{config.BASE_URL.rstrip('/')}/models",
                            headers=_headers(), timeout=30)
    response.raise_for_status()
    models = [m.get("id") for m in response.json().get("data", []) if m.get("id")]
    if not models:
        raise RuntimeError(f"No models served at {config.BASE_URL}/models")
    _RESOLVED_MODEL = models[0]
    return _RESOLVED_MODEL


def chat_completion(
    messages: list[dict],
    *,
    temperature: float | None = None,
    timeout: int | None = None,
    model: str = "",
    max_tokens: int | None = None,
    reason: bool = True,
    stream: bool | None = None,
    task: str = "",
    reason_tag: str | None = None,
) -> dict:
    resolved_model = model or resolve_served_model()
    tagged_messages = _with_reason_tag(messages, reason, task=task, reason_tag=reason_tag)
    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": tagged_messages,
        "temperature": config.TEMPERATURE if temperature is None else temperature,
        "stream": bool(config.STREAM_ENABLED if stream is None else stream),
    }
    if _should_disable_thinking_kwarg(messages, tagged_messages, reason):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    cache_kwarg = _resolve_cache_prompt_kwarg(messages, tagged_messages, task=task)
    if cache_kwarg is not None:
        payload["cache_prompt"] = cache_kwarg
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
        # requests defaults a charset-less text/event-stream to ISO-8859-1, which
        # mojibakes UTF-8 (a curly ’ becomes "â€™"). Force UTF-8 before decoding.
        response.encoding = "utf-8"
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

def generate(
    system_msg: str,
    user_input: str,
    reason: bool = False,
    *,
    task: str = "",
    reason_tag: str | None = None,
) -> str:
    """Single-shot system+user generation. Signature matches what meta-dreaming and
    the other bookkeeping passes expect: ``generate(system, user, reason) -> str``.
    Defaults to ``reason=False`` so internal JSON/rubric passes do not waste tokens
    in ``<think>`` unless configured via a task-specific mode or ``reason=True``.
    Never raises — returns an error string the caller can surface or log."""
    effective_task = task or ("background" if reason else "off")
    try:
        data = chat_completion(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_input}],
            temperature=0.6,
            reason=reason,
            task=effective_task,
            reason_tag=reason_tag,
        )
        cleaned = strip_reasoning(_reply_text(data))
        if not is_pure_construct_or_holodeck():
            cleaned = turbo_brilliance.strip_construct_borders(cleaned)
        return cleaned
    except Exception as e:  # noqa: BLE001 — a bookkeeping call must not crash a turn
        return f"Error connecting to the backend: {e}"


def generate_with_history(
    system_msg: str,
    messages: list[dict],
    reason: bool | None = None,
    temperature: float | None = None,
    *,
    task: str = "main",
    reason_tag: str | None = None,
) -> str:
    """The user-facing reply path: a system head plus the running conversation.

    The hidden reasoning chain still counts against ``max_tokens``. We cap the reply
    at GEN_RESERVE so it can't overflow the slot, but if reasoning fills that cap the
    visible content comes back empty with finish_reason "length"; detect exactly that
    and retry once with a cap sized to the slot's real headroom rather than failing.
    """
    global LAST_GENERATION_ERROR
    LAST_GENERATION_ERROR = ""
    if reason is None:
        reason = config.THINKING_ENABLED
    try:
        full = [{"role": "system", "content": system_msg}] + messages
        data = chat_completion(
            full,
            temperature=temperature,
            max_tokens=config.GEN_RESERVE,
            reason=reason,
            task=task,
            reason_tag=reason_tag,
        )
        reply = strip_reasoning(_reply_text(data))
        if not reply and _finish_reason(data) == "length":
            headroom = config.SLOT_CTX - _prompt_tokens(data) - config.SAFETY_MARGIN
            retry_cap = min(max(headroom, config.GEN_RESERVE), config.GEN_RESERVE * 4)
            if retry_cap > config.GEN_RESERVE:
                data = chat_completion(
                    full,
                    temperature=temperature,
                    max_tokens=retry_cap,
                    reason=reason,
                    task=task,
                    reason_tag=reason_tag,
                )
                reply = strip_reasoning(_reply_text(data))
        if not reply and reason:
            # Fallback retry with thinking disabled if reasoning consumed the budget
            data = chat_completion(
                full,
                temperature=temperature,
                max_tokens=config.GEN_RESERVE,
                reason=False,
                task="off",
            )
            reply = strip_reasoning(_reply_text(data))
        if not is_pure_construct_or_holodeck():
            reply = turbo_brilliance.strip_construct_borders(reply)
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
        _TOKENIZE_URL = _server_root_url() + "/tokenize"
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


def stream_chat(
    messages: list[dict],
    *,
    temperature: float | None = None,
    reason: bool | None = None,
    timeout: int | None = None,
    task: str = "main",
    reason_tag: str | None = None,
):
    """Yield visible content deltas from the backend as they arrive (for the web UI).

    Reasoning-chain deltas are dropped (the backend reports them separately as
    ``reasoning_content``), so the stream is already clean of the scratchpad. Raises
    on a connection error — the caller decides how to surface it."""
    if reason is None:
        reason = config.THINKING_ENABLED
    resolved_model = resolve_served_model()
    tagged_messages = _with_reason_tag(messages, reason, task=task, reason_tag=reason_tag)
    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": tagged_messages,
        "temperature": config.TEMPERATURE if temperature is None else temperature,
        "stream": True,
        "max_tokens": int(config.GEN_RESERVE),
        "stream_options": {"include_usage": True},
    }
    if _should_disable_thinking_kwarg(messages, tagged_messages, reason):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    cache_kwarg = _resolve_cache_prompt_kwarg(messages, tagged_messages, task=task)
    if cache_kwarg is not None:
        payload["cache_prompt"] = cache_kwarg
    url = f"{config.BASE_URL.rstrip('/')}/chat/completions"
    timeout = config.REPLY_TIMEOUT if timeout is None else timeout
    with requests.post(url, json=payload, headers=_headers(), timeout=timeout,
                       stream=True) as response:
        response.raise_for_status()
        # requests defaults a charset-less text/event-stream to ISO-8859-1, which
        # mojibakes UTF-8 (a curly ’ becomes "â€™"). Force UTF-8 before decoding.
        response.encoding = "utf-8"
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
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            if delta.get("content"):
                yield delta["content"]


_HEALTH_CACHE: tuple[float, bool, str] = (0.0, False, "")
_HEALTH_TTL = 4.0  # seconds; the dashboard polls /api/stats every ~3s


def health() -> tuple[bool, str]:
    """Reachability probe for /status and the dashboard. Actually pings the backend
    (a bare model-list GET) rather than trusting the configured id, and caches the
    result briefly so repeated polls don't each block. Returns (ok, detail)."""
    global _HEALTH_CACHE
    import time
    now = time.time()
    ts, ok, detail = _HEALTH_CACHE
    if now - ts < _HEALTH_TTL and ts:
        return ok, detail
    try:
        r = requests.get(f"{config.BASE_URL.rstrip('/')}/models",
                         headers=_headers(), timeout=2)
        r.raise_for_status()
        model = config.MODEL if config.MODEL and config.MODEL != "auto" \
            else resolve_served_model()
        ok, detail = True, f"backend reachable · serving {model}"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"backend unreachable at {config.BASE_URL}: {e}"
    _HEALTH_CACHE = (now, ok, detail)
    return ok, detail
