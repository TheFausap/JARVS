"""
claude_bridge.py — the governed conduit from JARVS (Gemma) to Claude Code.

JARVS never holds Anthropic credentials. This broker drives the ``claude`` CLI
through the Agent SDK (``claude-agent-sdk``), which is authenticated separately via
``claude login`` (your subscription). JARVS only hands the broker a request; the
broker is what talks to Claude.

Every exchange is a bounded, three-gate negotiation:

1. **Local pre-flight** — Gemma vets JARVS's own request (``generate_fn``) before a
   single Claude token is spent. A vague or out-of-scope ask is bounced here.
2. **Validity verdict** — Claude answers ``VERDICT: proceed|needs_clarification|
   reject`` *before* doing any work, running read-only (``permission_mode="plan"``)
   so it can inspect but not touch the tree.
3. **Authorized execution** — only if Gemma authorizes Claude's plan does the broker
   re-enter the same Claude session (``resume``) with write access
   (``permission_mode="acceptEdits"``) scoped to ``cwd``.

A rolling 24h **token ledger** governs usage: Gemma is warned as a cap is approached
and the request is refused past it. Governance is on tokens, not dollars (the
subscription path reports tokens, not per-request USD).

The same gates also drive a READ-ONLY mode, :meth:`ClaudeBridge.consult`, used by the
code foundry: Claude reviews a design, diagnoses a failure or critiques a finished
project without ever being offered write access. That is the Claude half of the
Qwen↔Claude duet — the local coder writes, Claude advises.

Design notes:
* The orchestration is synchronous and depends on an injected ``claude_fn`` so the
  gates are testable without the SDK installed; the default ``claude_fn`` wraps the
  async SDK call in :func:`asyncio.run`.
* The SDK import is lazy — importing this module never requires ``claude-agent-sdk``.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Callable

from . import config

# --- code-request intent detection (pure, testable) --------------------------- #

_CLAUDE_CODE_RE = re.compile(r"\bclaude\s*code\b", re.I)
_CLAUDE_RE = re.compile(r"\bclaude\b", re.I)
# A coding verb applied to a coding noun within a short window ("write a function",
# "fix the bug in parser.py"). Kept tight so ordinary chat isn't mistaken for a
# delegation request.
_CODE_ACTION_RE = re.compile(
    r"\b(?:write|implement|refactor|rewrite|debug|fix|patch|add|create|build|"
    r"optimi[sz]e)\b.{0,40}\b(?:code|coding|function|method|class|module|script|"
    r"test|tests|bug|feature|endpoint|api|patch|file|program|parser|regex)\b",
    re.I | re.S)
_CODE_WORD_RE = re.compile(
    r"\b(?:code|coding|program|implement|refactor|debug|fix|write|script|function|"
    r"bug|test)\b", re.I)


def is_code_request(text: str) -> bool:
    """True when the user is asking to delegate a coding task to Claude. High
    precision: an explicit 'claude code', a Claude mention paired with a coding
    word, or a bare code-action→code-noun phrase all qualify; a casual mention of
    Claude does not."""
    t = text or ""
    if _CLAUDE_CODE_RE.search(t):
        return True
    if _CLAUDE_RE.search(t) and _CODE_WORD_RE.search(t):
        return True
    return bool(_CODE_ACTION_RE.search(t))


# --- verdict parsing (pure) --------------------------------------------------- #

_VERDICT_RE = re.compile(
    r"verdict\s*:?\s*(proceed|needs[_\s-]*clarification|reject)", re.I)


def parse_verdict(reply: str) -> str:
    """Pull Claude's verdict out of its reply. Returns 'proceed',
    'needs_clarification', 'reject', or '' if none was stated."""
    m = _VERDICT_RE.search(reply or "")
    if not m:
        return ""
    v = m.group(1).lower()
    return "needs_clarification" if v.startswith("needs") else v


# --- usage normalisation (pure) ----------------------------------------------- #

def _g(obj, key, default=0):
    """Read ``key`` from a dict or an attribute off an object (ModelUsage may be
    either, depending on the SDK version)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def sum_usage(result_msg) -> dict:
    """Fold an Agent SDK ``ResultMessage`` into the ledger's token shape. Prefers
    ``model_usage`` (covers subagents) and falls back to the main-loop ``usage``."""
    tin = tout = tcache = 0
    mu = getattr(result_msg, "model_usage", None)
    if mu:
        for u in mu.values():
            tin += _g(u, "inputTokens")
            tout += _g(u, "outputTokens")
            tcache += _g(u, "cacheReadInputTokens") + _g(u, "cacheCreationInputTokens")
    else:
        u = getattr(result_msg, "usage", None) or {}
        tin = _g(u, "input_tokens")
        tout = _g(u, "output_tokens")
        tcache = _g(u, "cache_read_input_tokens") + _g(u, "cache_creation_input_tokens")
    return {"in": int(tin), "out": int(tout), "cache": int(tcache),
            "turns": int(getattr(result_msg, "num_turns", 0) or 0),
            "session": getattr(result_msg, "session_id", "") or ""}


class ClaudeBridge:
    """Governed JARVS→Claude coding bridge. One per session; thread-safe ledger."""

    def __init__(self, *, generate_fn: Callable[[str, str, bool], str],
                 ledger_path: str, cwd: str = ".", exchanges_path: str = "",
                 claude_fn: Callable[..., tuple[str, dict]] | None = None,
                 memory_fn: Callable[[str], None] | None = None,
                 on_event: Callable[[str, dict], None] | None = None):
        self._generate = generate_fn           # Gemma: pre-flight + authorize
        self._ledger_path = ledger_path
        self._exchanges_path = exchanges_path   # "" -> ledger_path + ".exchanges"
        self._cwd = cwd
        self._claude = claude_fn or self._default_claude
        self._memory_fn = memory_fn             # called with a distilled lesson/observation
        self._on_event = on_event
        self._lock = threading.RLock()
        self.ledger: list[dict] = self._load(self._ledger_path)   # token records
        self.exchanges: list[dict] = self._load(self._exch_path())  # persisted dialogues
        self.events: list[dict] = []             # recent events (ring), for REPL/UI

    # -- persistence ---------------------------------------------------------- #

    def _exch_path(self) -> str:
        return self._exchanges_path or (self._ledger_path + ".exchanges")

    @staticmethod
    def _load(path: str) -> list[dict]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

    @staticmethod
    def _save(path: str, data: list) -> None:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, path)
        except OSError:
            pass

    def _save_locked(self) -> None:
        self._save(self._ledger_path, self.ledger[-500:])

    def _save_exchanges_locked(self) -> None:
        self._save(self._exch_path(), self.exchanges[-40:])

    # -- events --------------------------------------------------------------- #

    def _emit(self, kind: str, **fields) -> None:
        ev = {"kind": kind, "t": time.time(), **fields}
        with self._lock:
            self.events.append(ev)
            del self.events[:-50]
        if self._on_event:
            try:
                self._on_event(kind, ev)
            except Exception:  # noqa: BLE001
                pass

    def drain_events(self) -> list[dict]:
        with self._lock:
            out, self.events = self.events, []
        return out

    # -- token governance ----------------------------------------------------- #

    def _tokens_since(self, seconds: float) -> int:
        cut = time.time() - seconds
        with self._lock:
            return sum(int(e.get("in", 0)) + int(e.get("out", 0))
                       for e in self.ledger if e.get("t", 0) >= cut)

    def budget_state(self) -> dict:
        """Rolling-24h token spend against the daily cap (0 cap == unlimited)."""
        day = self._tokens_since(86400)
        cap = config.CLAUDE_TOKEN_DAILY_CAP
        return {"day": day, "cap": cap,
                "frac": (day / cap) if cap else 0.0,
                "over": bool(cap and day >= cap)}

    def _warning(self, b: dict) -> str:
        if b["over"]:
            return (f"[BUDGET] JARVS is at {b['day']:,}/{b['cap']:,} tokens over the "
                    f"last 24h — at or over the cap.")
        if b["cap"] and b["frac"] >= config.CLAUDE_TOKEN_WARN_AT:
            return (f"[BUDGET] Heads up: {b['day']:,}/{b['cap']:,} tokens "
                    f"({b['frac']:.0%}) used in the last 24h. Keep this request tight.")
        return ""

    def _record(self, usage: dict) -> None:
        with self._lock:
            self.ledger.append({"t": time.time(),
                                "in": int(usage.get("in", 0)),
                                "out": int(usage.get("out", 0)),
                                "cache": int(usage.get("cache", 0)),
                                "turns": int(usage.get("turns", 0)),
                                "session": usage.get("session", "")})
            self._save_locked()

    # -- the dialogue (synchronous orchestration) ----------------------------- #

    def request_code(self, task: str, context: str = "") -> dict:
        """Run the three-gate negotiation for one coding task. Never raises — returns
        a result dict with a ``status`` the caller can surface. The finished exchange is
        persisted and a distilled lesson/observation is fed back to memory."""
        import uuid
        task = " ".join((task or "").split()).strip()[:1000]
        try:
            result = self._dialogue(task, context)
        except Exception as e:  # noqa: BLE001
            result = {"status": "error", "task": task, "turns": [],
                      "detail": f"{type(e).__name__}: {e}"}
            self._emit("error", task=task, detail=result["detail"])
        result["id"] = uuid.uuid4().hex[:8]
        result["t"] = time.time()
        result["tokens"] = sum(int(t.get("usage", {}).get("in", 0))
                               + int(t.get("usage", {}).get("out", 0))
                               for t in result.get("turns", []))
        with self._lock:
            self.exchanges.append(result)
            del self.exchanges[:-40]
            self._save_exchanges_locked()
        self._learn(result)
        return result

    def consult(self, task: str, context: str = "", cwd: str | None = None) -> dict:
        """A READ-ONLY consultation: Claude analyses, reviews or diagnoses, never edits.

        This is the bridge's half of the Qwen↔Claude duet. The foundry writes the code;
        Claude is asked to look at it — the design before a file exists, a failure the
        local model could not fix, or the finished project. It goes through the same
        governance as :meth:`request_code` (local pre-flight, budget gate, token
        ledger, persisted exchange), with two differences:

        * Claude is never offered write access — ``permission_mode='plan'`` throughout.
        * It is scoped to ``cwd`` (the foundry's project directory) when given, so
          Claude reads the code under discussion instead of JARVS's own tree.

        Never raises. Returns ``{"status": "ok"|"rejected_local"|"over_budget"|"error",
        "text": ..., "detail": ..., "tokens": ...}``.
        """
        import uuid
        task = " ".join((task or "").split()).strip()[:4000]
        try:
            result = self._consult(task, context, cwd)
        except Exception as e:  # noqa: BLE001
            result = {"status": "error", "task": task, "turns": [],
                      "detail": f"{type(e).__name__}: {e}", "text": ""}
            self._emit("error", task=task, detail=result["detail"])
        result["id"] = uuid.uuid4().hex[:8]
        result["t"] = time.time()
        result["kind"] = "consult"
        result["tokens"] = sum(int(t.get("usage", {}).get("in", 0))
                               + int(t.get("usage", {}).get("out", 0))
                               for t in result.get("turns", []))
        with self._lock:
            self.exchanges.append(result)
            del self.exchanges[:-40]
            self._save_exchanges_locked()
        return result

    def _consult(self, task: str, context: str, cwd: str | None) -> dict:
        self._emit("started", task=task[:120], mode="consult")

        # GATE 1 — local pre-flight, so a pointless review costs no Claude tokens.
        verdict = self._generate(
            "You are JARVS's gatekeeper for asking Claude to review a piece of work. "
            "Decide whether the question below is specific enough and worth Claude's "
            "subscription tokens. Reply on one line: 'PROCEED' or "
            "'REVISE: <one-line reason>'.",
            f"Question:\\n{task}\\n\\nKnown context:\\n{context or '(none)'}", False)
        if not verdict.strip().upper().startswith("PROCEED"):
            self._emit("rejected_local", task=task[:120], detail=verdict, mode="consult")
            return {"status": "rejected_local", "task": task, "turns": [],
                    "detail": verdict.strip(), "text": ""}

        # Budget gate — identical to an implementation request.
        b = self.budget_state()
        warn = self._warning(b)
        if b["over"]:
            self._emit("over_budget", task=task[:120], detail=warn, budget=b,
                       mode="consult")
            return {"status": "over_budget", "task": task, "turns": [],
                    "detail": warn, "text": "", "budget": b}

        prompt = ((warn + "\\n\\n") if warn else "") + (
            f"A local assistant (JARVS) is asking for a READ-ONLY review from Claude. "
            f"Analyse and advise — do not edit any file.\\n\\n"
            f"Question:\\n{task}\\n\\nContext:\\n{context or '(none)'}")
        reply, usage = self._call_claude(prompt, allow_write=False, resume=None, cwd=cwd)
        self._record(usage)
        turns = [{"role": "claude", "text": reply, "usage": usage}]
        self._emit("done", task=task[:120], detail=reply[:400],
                   budget=self.budget_state(), mode="consult")
        return {"status": "ok", "task": task, "turns": turns, "text": reply.strip(),
                "detail": reply.strip(), "budget": self.budget_state()}

    def _call_claude(self, prompt: str, *, allow_write: bool, resume: str | None,
                     cwd: str | None = None) -> tuple[str, dict]:
        """``self._claude`` with an optional directory scope.

        ``claude_fn`` is injected (and in tests is a plain two-arg callable), so a
        ``cwd`` kwarg is attempted first and dropped if the callable does not accept it.
        """
        if cwd:
            try:
                return self._claude(prompt, allow_write=allow_write, resume=resume,
                                    cwd=cwd)
            except TypeError:
                pass
        return self._claude(prompt, allow_write=allow_write, resume=resume)

    # -- learning (feed outcomes back into JARVS's memory) -------------------- #

    def _learn(self, result: dict) -> None:
        """Distill one exchange into a durable lesson/observation and hand it to
        memory, so recall surfaces JARVS's own missteps and Claude's findings next
        time. Best-effort; never raises."""
        if self._memory_fn is None:
            return
        status, task = result.get("status"), result.get("task", "")
        detail = (result.get("result") or result.get("detail") or "").strip()
        detail = " ".join(detail.split())[:800]
        lessons = {
            "rejected_local":
                f"[Claude-bridge lesson] JARVS held its own coding request back as "
                f"unfit to send: \"{task}\". Reason: {detail}. Specify requests like "
                f"this more concretely before delegating.",
            "needs_clarification":
                f"[Claude-bridge lesson] Claude needed clarification on \"{task}\": "
                f"{detail}. Include that detail up front next time.",
            "reject":
                f"[Claude-bridge lesson] Claude rejected \"{task}\": {detail}. Avoid "
                f"framing a request this way.",
            "held":
                f"[Claude-bridge lesson] JARVS reviewed Claude's plan for \"{task}\" "
                f"and held off: {detail}.",
            "over_budget":
                f"[Claude-bridge note] A coding request (\"{task}\") was refused on the "
                f"token budget. {detail}",
            "done":
                f"[Claude-bridge observation] Claude completed \"{task}\". What it found "
                f"/ changed: {detail}",
        }
        text = lessons.get(status)
        if not text:
            return
        try:
            self._memory_fn(text)
        except Exception:  # noqa: BLE001
            pass

    # -- web payload ---------------------------------------------------------- #

    def payload(self, limit: int = 12) -> dict:
        """The /api/claude view: budget meter + recent exchanges (trimmed)."""
        out = []
        for e in self.recent(limit):
            out.append({"id": e.get("id"), "t": e.get("t"), "status": e.get("status"),
                        "task": e.get("task", ""), "tokens": e.get("tokens", 0),
                        "detail": (e.get("result") or e.get("detail") or "")[:600]})
        return {"enabled": True, "budget": self.budget_state(), "exchanges": out}

    def _dialogue(self, task: str, context: str) -> dict:
        self._emit("started", task=task)

        # GATE 1 — local pre-flight (no Claude tokens spent).
        verdict = self._generate(
            "You are JARVS's gatekeeper for delegating work to Claude Code. Decide if "
            "this coding request is specific enough, in scope, and safe to send. Reply "
            "on one line: 'PROCEED' or 'REVISE: <one-line reason>'.",
            f"Request:\n{task}\n\nKnown context:\n{context or '(none)'}", False)
        if not verdict.strip().upper().startswith("PROCEED"):
            self._emit("rejected_local", task=task, detail=verdict)
            return {"status": "rejected_local", "task": task, "turns": [],
                    "detail": verdict.strip()}

        # Budget gate.
        b = self.budget_state()
        warn = self._warning(b)
        if b["over"]:
            self._emit("over_budget", task=task, detail=warn, budget=b)
            return {"status": "over_budget", "task": task, "turns": [],
                    "detail": warn, "budget": b}

        # GATE 2 — Claude's validity verdict (read-only).
        turns: list[dict] = []
        prompt = ((warn + "\n\n") if warn else "") + (
            f"A local assistant (JARVS) is delegating a coding task.\n\n"
            f"Task:\n{task}\n\nContext:\n{context or '(none)'}")
        reply, u1 = self._claude(prompt, allow_write=False, resume=None)
        self._record(u1)
        turns.append({"role": "claude", "text": reply, "usage": u1})
        decision = parse_verdict(reply)
        if decision != "proceed":
            status = decision or "no_verdict"
            self._emit(status, task=task, detail=reply[:400])
            return {"status": status, "task": task, "turns": turns,
                    "detail": reply.strip(), "budget": self.budget_state()}

        # GATE 3 — Gemma authorizes, then Claude executes (write, same session).
        authorize = self._generate(
            "Claude reviewed a coding task and is ready to implement it. Based on its "
            "assessment below, reply 'AUTHORIZE' to let it make changes, or "
            "'HOLD: <one-line reason>' to stop.", reply, False)
        if not authorize.strip().upper().startswith("AUTHORIZE"):
            self._emit("held", task=task, detail=authorize)
            return {"status": "held", "task": task, "turns": turns,
                    "detail": authorize.strip(), "budget": self.budget_state()}

        follow = ("Proceed with the implementation. Make the changes and then report "
                  "concisely what you changed and why.")
        result, u2 = self._claude(follow, allow_write=True, resume=u1.get("session"))
        self._record(u2)
        turns.append({"role": "claude", "text": result, "usage": u2})
        self._emit("done", task=task, detail=result[:400], budget=self.budget_state())
        return {"status": "done", "task": task, "turns": turns,
                "result": result.strip(), "budget": self.budget_state()}

    # -- the real Claude call (async SDK, wrapped) ---------------------------- #

    def _default_claude(self, prompt: str, *, allow_write: bool,
                        resume: str | None, cwd: str | None = None) -> tuple[str, dict]:
        import asyncio
        return asyncio.run(self._sdk_call(prompt, allow_write, resume, cwd))

    async def _sdk_call(self, prompt: str, allow_write: bool,
                        resume: str | None, cwd: str | None = None) -> tuple[str, dict]:
        # Refuse to run against the paid API when we were told to use the
        # subscription — the Agent SDK bills the API if ANTHROPIC_API_KEY is set.
        if config.CLAUDE_REQUIRE_SUBSCRIPTION and os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is set but JARVS_CLAUDE_REQUIRE_SUBSCRIPTION is on; "
                "refusing so the subscription (not the paid API) is used. Unset the key "
                "or set JARVS_CLAUDE_REQUIRE_SUBSCRIPTION=0.")

        from claude_agent_sdk import (  # lazy: module import never needs the SDK
            ClaudeSDKClient, ClaudeAgentOptions, ResultMessage, AssistantMessage,
            TextBlock)

        opts_kwargs = dict(
            cwd=cwd or self._cwd,
            max_turns=config.CLAUDE_MAX_TURNS,
            permission_mode="acceptEdits" if allow_write else "plan",
            allowed_tools=(["Read", "Grep", "Glob", "Write", "Edit", "MultiEdit"]
                           if allow_write else ["Read", "Grep", "Glob"]),
            system_prompt=(
                "You are Claude Code, invoked by a local assistant named JARVS to help "
                "with coding. Unless you have been authorized to implement, FIRST line "
                "of your reply must be 'VERDICT: proceed', 'VERDICT: needs_clarification' "
                "or 'VERDICT: reject', followed by concerns or questions, and you must "
                "not edit files."))
        if config.CLAUDE_MODEL:
            opts_kwargs["model"] = config.CLAUDE_MODEL
        if resume:
            opts_kwargs["resume"] = resume
        options = ClaudeAgentOptions(**opts_kwargs)

        text: list[str] = []
        usage: dict = {}
        async with ClaudeSDKClient(options=options) as claude:
            await claude.query(prompt)
            async for msg in claude.receive_response():
                if isinstance(msg, AssistantMessage):
                    text += [b.text for b in msg.content if isinstance(b, TextBlock)]
                elif isinstance(msg, ResultMessage):
                    usage = sum_usage(msg)
        return "\n".join(t for t in text if t).strip(), usage

    # -- introspection -------------------------------------------------------- #

    def recent(self, limit: int = 10) -> list[dict]:
        with self._lock:
            return [dict(x) for x in self.exchanges[-limit:]][::-1]

    def status(self) -> str:
        b = self.budget_state()
        cap = (f"{b['day']:,}/{b['cap']:,} tok/24h" if b["cap"]
               else f"{b['day']:,} tok/24h (no cap)")
        return (f"claude bridge: {len(self.exchanges)} exchange(s) this session · {cap}"
                + (" · OVER CAP" if b["over"] else ""))
