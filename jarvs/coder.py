"""
coder.py — autonomous code generation (the foundry).

Give JARVS a TOPIC ("a terminal habit tracker", "a tiny static site generator") and it
builds a real project on disk, unattended:

    plan      one call to the coder model turns the topic into a concrete design —
              name, language, entry point, file list with a purpose per file, and the
              command that proves it works.
    write     every file is generated in full, in dependency order, with the design
              and the already-written files in context so the pieces agree.
    verify    the project is actually run: a syntax/compile check, then the test
              command (pytest / npm test / …) inside the project directory.
    repair    failures go straight back to the model with the offending file and the
              raw error, and the loop repeats — bounded by JARVS_CODER_MAX_REPAIRS.

Everything is bounded and sandboxed: writes are confined to JARVS_PROJECTS_DIR (path
traversal rejected), subprocesses run from an allowlist of interpreters with a timeout
and no shell, and the whole run can be disarmed with JARVS_CODER_EXEC=0.

The coding specialist is Qwen3-Coder-Next, reached over the same OpenAI-compatible
protocol as the base model (see :class:`CoderBackend`), with an automatic fallback to
the base backend if the coder endpoint is down.

Runs off the conversation path like :mod:`jarvs.background`: queue a topic, a daemon
thread builds it, results surface as events (REPL, /api/code, transcript).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Callable

import requests

from . import config
from .client import strip_reasoning

# --- trigger detection (pure, testable) --------------------------------------- #

# A build verb aimed at a software object. Both halves are required so ordinary
# requests ("explain how a parser works") never start a build.
_BUILD_VERB = re.compile(
    r"\b(?:build|write|create|generate|implement|code|develop|make|scaffold|"
    r"prototype|put together|whip up)\b", re.I)
_BUILD_OBJECT = re.compile(
    r"\b(?:app|application|program|script|tool|cli|command[- ]line|project|module|"
    r"library|package|game|bot|server|api|parser|scraper|crawler|dashboard|"
    r"simulator|simulation|generator|calculator|tracker|converter|client|"
    r"framework|engine|demo|prototype|plugin|utility|utilities|website|site)\b", re.I)
_CODE_PHRASE = re.compile(
    r"\b(?:write|generate|build|some|the)\s+(?:me\s+)?(?:the\s+)?code\b"
    r"|\bcode\s+(?:me|up|that|this|a|an)\b|\bscaffold\b|\bboilerplate\b"
    r"|\bfrom scratch\b|\brepo(?:sitory)?\b", re.I)
# Questions about building are not requests to build.
_QUESTION = re.compile(
    r"^\s*(?:what|how|why|when|where|who|which|is|are|was|were|do|does|did|"
    r"can|could|would|should|explain|tell me)\b.*\?\s*$", re.I | re.S)

# The foundry builds NEW projects from a topic. Work on code that already exists is the
# Claude bridge's job (it runs in the user's tree), so these disqualify a request here:
# a named source file, a definite reference to an existing artifact ("the parser", "our
# client"), or a maintenance verb leading the sentence ("refactor the module").
_CODE_FILE_RE = re.compile(
    r"\b[\w./-]+\.(?:py|js|mjs|cjs|ts|tsx|jsx|go|rs|java|rb|c|cpp|h|sh)\b", re.I)
_EXISTING_REF_RE = re.compile(
    r"\b(?:the|this|that|our|my|existing|current)\s+(?:[\w-]+\s+){0,1}"
    r"(?:module|file|class|function|method|repo|repository|codebase|project|script|"
    r"parser|server|client|api|endpoint|test|tests|code)\b", re.I)
_MAINTENANCE_LEAD_RE = re.compile(
    r"^\s*(?:please\s+)?(?:fix|refactor|debug|patch|optimi[sz]e|update|change|"
    r"rename|move|remove|clean\s+up|improve|rewrite|migrate|add)\b", re.I)


def is_code_request(text: str) -> bool:
    """High-precision test for 'go build this for me'.

    Needs a build verb plus a software object, or an explicit code phrase — and never
    fires on a question, so 'how would you build a parser?' stays a conversation.
    """
    t = (text or "").strip()
    if len(t) < 8 or _QUESTION.match(t):
        return False
    if (_CODE_FILE_RE.search(t) or _EXISTING_REF_RE.search(t)
            or _MAINTENANCE_LEAD_RE.match(t)):
        return False           # about existing code — the bridge's job, not ours
    return bool(_CODE_PHRASE.search(t)
                or (_BUILD_VERB.search(t) and _BUILD_OBJECT.search(t)))


_CODE_MARKER = re.compile(r"\[\[CODE:\s*(.+?)\]\]", re.I | re.S)


def extract_code_markers(reply: str) -> tuple[str, list[str]]:
    """Pull ``[[CODE: topic]]`` markers out of a reply (the model deferring a build)."""
    topics = [m.strip() for m in _CODE_MARKER.findall(reply or "") if m.strip()]
    clean = _CODE_MARKER.sub("", reply or "")
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, topics


# --- JSON / file-block parsing (pure) ----------------------------------------- #

def _json_object(raw: str) -> dict | None:
    """Parse a JSON object out of a model reply, tolerating prose and code fences."""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", (raw or "").strip(),
                  flags=re.MULTILINE).strip()
    for candidate in (text, _between(text, "{", "}")):
        if not candidate:
            continue
        try:
            val = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(val, dict):
            return val
    return None


def _between(text: str, open_ch: str, close_ch: str) -> str:
    start, end = text.find(open_ch), text.rfind(close_ch)
    return text[start:end + 1] if start != -1 and end > start else ""


# `=== FILE: path ===` (or ### / ---) followed by one fenced block.
_FILE_BLOCK_RE = re.compile(
    r"^[ \t]*(?:===+|###+|---+|\*\*\*+)[ \t]*FILE:[ \t]*(?P<path>[^\s=]+)[ \t]*"
    r"(?:===+|###+|---+|\*\*\*+)?[ \t]*\r?\n"
    r"[ \t]*```[a-zA-Z0-9_+\-]*[ \t]*\r?\n(?P<body>.*?)\r?\n[ \t]*```",
    re.MULTILINE | re.DOTALL)
# A bare fenced block is accepted only when exactly one file is expected.
_BARE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+\-]*[ \t]*\r?\n(?P<body>.*?)\r?\n[ \t]*```",
                            re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+\-]*\s*$")


def extract_files(text: str, single_path: str = "") -> list[tuple[str, str]]:
    """Parse ``FILE: path`` + fenced-code blocks out of a model reply.

    Returns [(relative_path, content)]. Content is de-fenced and the trailing
    truncation marker (a model that ran out of budget mid-file) is flagged by a
    missing closing fence only at the very end of the reply.
    """
    out: list[tuple[str, str]] = []
    for m in _FILE_BLOCK_RE.finditer(text or ""):
        path = _clean_rel_path(m.group("path"))
        body = m.group("body")
        if path:
            out.append((path, body.strip("\n")))
    if out or not single_path:
        return out
    stripped = text or ""
    if stripped.lstrip().startswith("```"):        # reply is just the code
        stripped = stripped.strip()
        stripped = stripped[stripped.find("\n") + 1:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
        return [(single_path, stripped.strip("\n"))]
    blocks = [b.group("body") for b in _BARE_BLOCK_RE.finditer(text or "")]
    if len(blocks) == 1:
        return [(single_path, blocks[0].strip("\n"))]
    return []


def _clean_rel_path(raw: str) -> str:
    """Normalise a model-emitted path into a safe relative one; '' when unusable.

    Models write ``./src/x.py``, ``/src/x.py``, ``src\\x.py`` — all accepted and folded
    into the project. Anything that still escapes after normalisation is rejected
    (:func:`safe_join` re-checks containment, so this is defence in depth, not the
    only gate).
    """
    p = (raw or "").strip().strip("`\"'").replace("\\", "/")
    p = re.sub(r"^[\s/]+", "", p)                   # /src/x.py -> src/x.py
    p = re.sub(r"^(?:\./)+", "", p)                 # ./src/x.py -> src/x.py
    p = os.path.normpath(p).replace(os.sep, "/")
    if not p or p == "." or p == ".." or p.startswith("../"):
        return ""
    return p


def safe_join(root: str, rel: str) -> str | None:
    """Resolve ``rel`` inside ``root``. Returns None if it would escape (or is junk)."""
    rel = _clean_rel_path(rel)
    if not rel:
        return None
    root_abs = os.path.abspath(root)
    full = os.path.abspath(os.path.join(root_abs, rel))
    if full != root_abs and not full.startswith(root_abs + os.sep):
        return None
    return full


# --- command sandbox ---------------------------------------------------------- #

# Only these interpreters may be executed, never through a shell.
_ALLOWED_PROGRAMS = {"python", "python3", "pytest", "py.test", "node", "npm",
                     "npx", "go", "cargo", "make", "deno", "bun"}
# Reject anything that smells like chaining, redirection, or network/system access.
_DENY_TOKENS = {";", "|", "&", ">", "<", "$(", "`", "&&", "||", "*", "?",
                "rm", "del", "curl", "wget", "sudo", "su", "sh", "bash", "zsh",
                "eval", "exec", "install", "publish", "deploy", "git", "ssh",
                "chmod", "chown", "mkfs", "dd"}


def sanitize_command(cmd: str) -> list[str] | None:
    """Turn a model-proposed test command into a safe argv list, or None if unsafe."""
    cmd = (cmd or "").strip()
    if not cmd or len(cmd) > 200:
        return None
    if any(bad in cmd for bad in _DENY_TOKENS):
        return None
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return None
    if not argv:
        return None
    program = os.path.basename(argv[0]).lower()
    if program.endswith(".exe"):
        program = program[:-4]
    if program not in _ALLOWED_PROGRAMS:
        return None
    return [sys.executable if program in ("python", "python3") and
            os.path.dirname(argv[0]) == "" else argv[0]] + argv[1:]


# --- the coder backend -------------------------------------------------------- #

class CoderBackend:
    """Qwen3-Coder-Next over the OpenAI-compatible wire, with a fallback.

    ``complete(system, user) -> str`` never raises: on failure it returns an
    ``"error: …"`` string, and if the coder endpoint is a different server it retries
    once against the base backend so a build still completes (slower, weaker, but not
    dead).
    """

    def __init__(self, timeout: int | None = None):
        self.base_url = (config.CODER_BASE_URL or config.BASE_URL).rstrip("/")
        self.fallback_url = config.BASE_URL.rstrip("/") if config.CODER_BASE_URL else ""
        self.model = config.CODER_MODEL
        self.timeout = timeout or config.CODER_TIMEOUT
        self.last_error = ""
        self.used_fallback = False

    def _headers(self) -> dict:
        key = config.CODER_API_KEY or config.API_KEY
        return {"Content-Type": "application/json",
                **({"Authorization": f"Bearer {key}"} if key else {})}

    def _post(self, url: str, system: str, user: str,
              max_tokens: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": config.CODER_TEMPERATURE,
            "max_tokens": int(max_tokens or config.CODER_MAX_TOKENS),
            "stream": False,
        }
        if not config.THINKING_ENABLED:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        r = requests.post(f"{url}/chat/completions", json=payload,
                          headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        content = ((r.json().get("choices") or [{}])[0].get("message") or {}).get(
            "content") or ""
        return strip_reasoning(content).strip()

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        self.last_error = ""
        try:
            return self._post(self.base_url, system, user, max_tokens)
        except Exception as e:  # noqa: BLE001 — a build must not die on one bad call
            self.last_error = f"{type(e).__name__}: {e}"
        if self.fallback_url and self.fallback_url != self.base_url:
            try:
                out = self._post(self.fallback_url, system, user, max_tokens)
                self.used_fallback = True
                self.last_error = ""
                return out
            except Exception as e:  # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"
        return f"error: {self.last_error or 'coder backend unavailable'}"

    def health(self) -> tuple[bool, str]:
        """(ok, detail) — a cheap reachability probe for /status and the dashboard."""
        try:
            r = requests.get(f"{self.base_url}/models", headers=self._headers(),
                             timeout=3)
            r.raise_for_status()
            return True, f"coder reachable at {self.base_url} · {self.model}"
        except Exception as e:  # noqa: BLE001
            return False, f"coder unreachable at {self.base_url}: {e}"

    @staticmethod
    def failed(text: str) -> bool:
        t = (text or "").strip().lower()
        return not t or t.startswith("error:")


# --- prompts ------------------------------------------------------------------ #

_PLAN_SYSTEM = """You are a senior software engineer drafting a build plan for a small,
self-contained software project. The user will give you only a TOPIC — you decide the
scope, the language, and the file layout.

Rules:
- Choose the smallest design that is genuinely COMPLETE and RUNNABLE. Working beats
  ambitious. Prefer the standard library.
- 2 to {max_files} files. One of them must be the entry point. Add a test file when the
  language has a test runner available (pytest for Python, node's built-in test runner
  for JavaScript).
- Every file needs a one-line purpose; file paths are relative and use forward slashes.
- `test_command` is the command that proves the project works (e.g. "python -m pytest -q"
  or "node --test"). It runs in the project directory with no shell. If nothing can
  verify the project, use an empty string.
- `language` is one of: python, javascript, typescript, go, rust, html, other.

Output ONLY a JSON object (no prose, no fences):
{{"name": "short project name", "slug": "directory-name", "language": "python",
 "entry": "main.py", "summary": "two sentences on what it does and how it is run",
 "test_command": "python -m pytest -q",
 "files": [{{"path": "main.py", "purpose": "CLI entry point and argument parsing"}}]}}"""

_PLAN_USER = """TOPIC:
{topic}

Design a complete, runnable project for this topic. Reply with the JSON object only."""

_FILE_SYSTEM = """You are a senior software engineer writing ONE file of a project whose
design is already fixed. Write the complete file — real, working, self-contained code.
No placeholders, no TODOs, no "…rest of implementation…", no ellipses. The code must run
as written and must agree with the design and with the files already written.

Reply with EXACTLY this shape and nothing else:

=== FILE: {path} ===
```{lang}
<the complete file contents>
```

Do not add commentary before or after the block."""

_FILE_USER = """PROJECT: {name} — {summary}
Language: {language} · entry point: {entry}

DESIGN (all files):
{files}

{written}Write the file `{path}` now.
Purpose: {purpose}"""

_REPAIR_SYSTEM = """You are fixing a broken file in a project you wrote. You get the file
as it stands and the exact error the checks produced. Return the COMPLETE corrected file
— not a diff, not a fragment, no placeholders.

Reply with EXACTLY this shape and nothing else:

=== FILE: {path} ===
```{lang}
<the complete corrected file>
```"""

_REVISE_SYSTEM = """You are a senior software engineer revising your build plan after a
peer review.
Another engineer has read your design and raised points. Adopt the ones that make the
project better and keep the design small, complete and runnable.

Output ONLY the revised JSON object with the same schema (no prose, no fences):
{{"name": str, "slug": str, "language": str, "entry": str, "summary": str,
 "test_command": str, "files": [{{"path": str, "purpose": str}}]}}

Keep at most {max_files} files. If you disagree with a point, you may ignore it, but
the result must still be a complete project."""

_REVISE_USER = """TOPIC:
{topic}

CURRENT DESIGN:
{plan}

PEER REVIEW:
{advice}

Reply with the revised JSON object only."""

_DUET_BRIEF = {
    "plan": """Review this design for a small, self-contained software project BEFORE any
code is written. The project must run as specified and be complete.

TOPIC: {topic}

DESIGN:
{plan}

Reply with concrete, actionable revisions — at most 10 lines. If the design is sound,
reply with a single line: LGTM.""",
    "failure": """A local coding model wrote the project below but its checks still fail after
{rounds} repair attempt(s). Read the code in this directory and the failing output, then
diagnose the ROOT CAUSE.

PROJECT: {name} — {summary}
FILES: {files}

FAILING CHECK OUTPUT:
```
{error}
```

Reply with the root cause and the concrete change that fixes it — at most 15 lines,
naming the specific file(s). Do not edit anything.""",
    "critique": """Review the finished project below. It builds and its checks pass. Look for
correctness gaps, edge cases it gets wrong, and anything that will break in normal use.

PROJECT: {name} — {summary}
FILES: {files}

Reply with the defects worth fixing — at most 10 lines, naming the specific file(s). If
you find nothing worth changing, reply with a single line: LGTM.""",
}

_REPAIR_USER = """PROJECT: {name} — {summary}
FILE: {path}
Purpose: {purpose}

CURRENT FILE CONTENTS:
```{lang}
{content}
```

FAILING CHECK OUTPUT:
```
{error}
```

Rewrite `{path}` so the project builds and the checks pass. If the failure is caused by
another file's interface, adapt THIS file to that interface rather than renaming things
you cannot see.
{advice}"""


# --- the foundry -------------------------------------------------------------- #

_LANG_FENCE = {"python": "python", "javascript": "javascript", "typescript":
               "typescript", "go": "go", "rust": "rust", "html": "html",
               "other": "text"}
# Extensions we can syntax-check without a build step.
_CHECKABLE = {".py": "python", ".js": "javascript", ".mjs": "javascript",
              ".cjs": "javascript", ".ts": "typescript"}


class CodeFoundry:
    """Queue of autonomous build jobs, each drained by a single worker daemon."""

    def __init__(self, *, store_path: str,
                 generate_fn: Callable[[str, str], str] | None = None,
                 review_fn: Callable[[str, str | None], str] | None = None,
                 memory_fn: Callable[[str], None] | None = None,
                 on_event: Callable[[str, dict], None] | None = None,
                 projects_dir: str | None = None):
        self._generate = generate_fn or CoderBackend().complete
        # The duet: one bounded, read-only consultation with the peer model (Claude).
        # ``review_fn(brief, cwd) -> str``; '' means it declined or is unavailable, and
        # the build simply continues as a solo run.
        self._review = review_fn
        self._memory_fn = memory_fn
        self._on_event = on_event
        self._store_path = store_path
        self._projects_dir = projects_dir or str(config.PROJECTS_DIR)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self.projects: list[dict] = self._load()
        self.events: list[dict] = []
        # Consultations spent per build (project id -> count), capped by
        # CODER_COLLAB_ROUNDS so one build can't drain a subscription on its own.
        self._claude_used: dict[str, int] = {}

    # -- persistence ---------------------------------------------------------- #

    def _load(self) -> list[dict]:
        try:
            with open(self._store_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

    def _save_locked(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._store_path) or ".", exist_ok=True)
            tmp = f"{self._store_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.projects, fh, indent=2)
            os.replace(tmp, self._store_path)
        except OSError:
            pass

    def _update(self, project_id: str, **fields) -> dict:
        with self._lock:
            p = next((x for x in self.projects if x.get("id") == project_id), None)
            if p is None:
                return {}
            p.update(fields)
            self._save_locked()
            return dict(p)

    # -- events --------------------------------------------------------------- #

    def _emit(self, kind: str, project: dict) -> None:
        ev = {"kind": kind, "id": project.get("id"), "topic": project.get("topic", ""),
              "name": project.get("name") or project.get("topic", ""),
              "status": project.get("status"), "progress": project.get("progress", ""),
              "path": project.get("path", ""), "t": time.time()}
        with self._lock:
            self.events.append(ev)
            del self.events[:-50]
        if self._on_event:
            try:
                self._on_event(kind, project)
            except Exception:  # noqa: BLE001
                pass

    def drain_events(self) -> list[dict]:
        with self._lock:
            out, self.events = self.events, []
        return out

    # -- queue ---------------------------------------------------------------- #

    def create(self, topic: str, name: str = "") -> dict | None:
        """Record a build WITHOUT starting the worker (so a caller can run it inline)."""
        topic = " ".join((topic or "").split()).strip()[:500]
        if len(topic) < 4:
            return None
        with self._lock:
            active = [p for p in self.projects
                      if p.get("status") in ("queued", "running")]
            if len(active) >= config.CODER_MAX_PENDING:
                return None
            for p in active:  # the same build already pending — refuse, don't duplicate
                if p.get("topic", "").casefold() == topic.casefold():
                    return None
            project = {"id": uuid.uuid4().hex[:8], "topic": topic, "name": name,
                       "slug": "", "path": "", "language": "", "summary": "",
                       "status": "queued", "progress": "queued", "plan": {},
                       "files": [], "checks": [], "verified": False,
                       "repairs": 0, "created": time.time(), "finished": None,
                       "result": "", "error": ""}
            self.projects.append(project)
            del self.projects[:-30]
            self._save_locked()
        self._emit("queued", project)
        return dict(project)

    def queue(self, topic: str, name: str = "") -> dict | None:
        """Record a build and hand it to the worker thread (the normal path)."""
        project = self.create(topic, name)
        if project is not None:
            self._ensure_worker()
        return project

    def _claim(self) -> dict | None:
        with self._lock:
            for p in self.projects:
                if p.get("status") == "queued":
                    p["status"] = "running"
                    p["started"] = time.time()
                    self._save_locked()
                    return dict(p)
        return None

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._worker,
                                            name="jarvs-coder", daemon=True)
            self._thread.start()

    # -- the build pipeline --------------------------------------------------- #

    # -- the duet (peer review inside the build) ------------------------------ #

    @property
    def duet_enabled(self) -> bool:
        """True when a peer model is available AND collaboration is switched on."""
        return self._review is not None and config.CODER_COLLAB

    def _consult(self, pid: str, kind: str, brief: str, cwd: str | None = None) -> str:
        """One bounded consultation with the peer model. '' when it declines.

        ``CODER_COLLAB_ROUNDS`` caps how many of these ONE build may spend, so a
        pathological project cannot quietly drain a subscription.
        """
        if self._review is None or not config.CODER_COLLAB:
            return ""
        used = self._claude_used.get(pid, 0)
        if used >= config.CODER_COLLAB_ROUNDS:
            return ""
        self._claude_used[pid] = used + 1
        self._update(pid, progress=f"claude {kind}")
        try:
            reply = (self._review(brief, cwd) or "").strip()
        except Exception:  # noqa: BLE001 — a declined review must not fail the build
            reply = ""
        if not reply:
            self._claude_used[pid] = used          # a declined call costs nothing
            return ""
        self._note(pid, "peer", kind, reply)
        return reply

    @staticmethod
    def _is_lgtm(text: str) -> bool:
        """Did the reviewer effectively say 'nothing to change'?"""
        t = (text or "").strip().lower()
        if not t:
            return True
        if t.startswith("lgtm") or t.startswith("verdict: lgtm"):
            return True
        return len(t) < 12 and "lgtm" in t

    def _note(self, pid: str, role: str, kind: str, text: str) -> None:
        """Append one turn of the Qwen↔Claude dialogue to the project record."""
        with self._lock:
            p = next((x for x in self.projects if x.get("id") == pid), None)
            if p is None:
                return
            notes = list(p.get("collab") or [])
            notes.append({"role": role, "kind": kind, "text": text[:4000],
                          "t": time.time()})
            p["collab"] = notes[-12:]
            self._save_locked()

    def _revise_plan(self, pid: str, plan: dict, topic: str, advice: str) -> dict:
        """Hand the peer's review of the design back to the coder and take the revision."""
        self._update(pid, progress="revising the design")
        raw = self._generate(
            _REVISE_SYSTEM.format(max_files=config.CODER_MAX_FILES),
            _REVISE_USER.format(topic=topic,
                                plan=json.dumps(plan, indent=2)[:6000],
                                advice=advice[:4000]))
        revised = _json_object(raw)
        if not revised or not isinstance(revised.get("files"), list):
            return plan                     # an unusable revision keeps the original
        files = []
        for f in revised["files"][:config.CODER_MAX_FILES]:
            if not isinstance(f, dict):
                continue
            path = _clean_rel_path(str(f.get("path", "")))
            if path:
                files.append({"path": path, "purpose": str(f.get("purpose", ""))[:300]})
        if not files:
            return plan
        revised["files"] = files
        revised["entry"] = _clean_rel_path(str(revised.get("entry") or "")) or files[0]["path"]
        revised["name"] = str(revised.get("name") or plan.get("name") or topic)[:80]
        revised["summary"] = str(revised.get("summary") or plan.get("summary") or "")[:600]
        revised["test_command"] = str(revised.get("test_command")
                                      or plan.get("test_command") or "")[:200]
        revised["language"] = str(revised.get("language")
                                  or plan.get("language") or "other").lower()
        self._note(pid, "coder", "revised design",
                   json.dumps({k: revised[k] for k in
                               ("name", "language", "entry", "files")}, indent=2))
        return revised

    def build(self, project: dict) -> dict:
        """Plan → write → verify → repair. Returns the finished project dict.

        With a peer model attached (the bridge on) the pipeline becomes a dialogue:
        Claude reviews the design before a file is written, diagnoses any failure the
        local repairs could not fix, and critiques the result once it passes.
        """
        pid = project["id"]
        self._claude_used[pid] = 0
        try:
            plan = self._plan(project)
            if plan is None:
                return self._finish(pid, error="could not turn that topic into a plan")
            slug = self._slug(plan.get("slug") or plan.get("name") or
                              project["topic"], pid)
            root = safe_join(self._projects_dir, slug)
            if root is None:
                return self._finish(pid, error="rejected project path (unsafe slug)")
            os.makedirs(root, exist_ok=True)
            self._update(pid, name=plan.get("name") or project["topic"], slug=slug,
                         path=root, language=plan.get("language") or "other",
                         summary=plan.get("summary") or "", plan=plan,
                         progress="planning done")

            # DUET 1 — the design is reviewed before anything is written: catching a
            # bad plan here costs one review instead of a whole project.
            if self.duet_enabled and config.CODER_COLLAB_PLAN:
                brief = _DUET_BRIEF["plan"].format(
                    topic=project["topic"], plan=self._design_block(plan))
                advice = self._consult(pid, "design review", brief)
                if advice and not self._is_lgtm(advice):
                    plan = self._revise_plan(pid, plan, project["topic"], advice)
                    self._update(pid, plan=plan,
                                 name=plan.get("name") or project["topic"],
                                 language=plan.get("language") or "other",
                                 summary=plan.get("summary") or "")

            written = self._write_files(pid, plan, root)
            if not written:
                return self._finish(pid, error="no files were generated")

            checks = self._verify(plan, root)
            self._update(pid, checks=checks, verified=checks.get("ok", False),
                         progress="verifying")

            rounds = 0
            while not checks.get("ok") and rounds < config.CODER_MAX_REPAIRS:
                if not any(c.get("runnable") for c in checks.get("checks", [])):
                    break                      # nothing that can be run → can't repair
                rounds += 1
                self._update(pid, repairs=rounds, progress=f"repairing {rounds}")
                fixed = self._repair(pid, plan, root, checks)
                if not fixed:
                    break
                checks = self._verify(plan, root)
                self._update(pid, checks=checks, verified=checks.get("ok", False))

            # DUET 2 — still failing: let the peer read the code and diagnose it, then
            # hand that diagnosis to the coder as the repair brief. This is the case
            # where a second model actually changes the outcome.
            checks, rounds = self._collaborate(pid, plan, root, checks, rounds)

            self._write_report(pid, plan, root, checks, rounds)
            self._tidy(root)                   # no __pycache__/.pytest_cache in the result
            self._update(pid, repairs=rounds, progress="done")
            return self._finish(pid)
        except Exception as e:  # noqa: BLE001 — one bad build must not kill the worker
            return self._finish(pid, error=f"{type(e).__name__}: {e}")

    def _collaborate(self, pid: str, plan: dict, root: str, checks: dict,
                     rounds: int) -> tuple[dict, int]:
        """The peer's remaining interventions: diagnose a failure, then critique a pass.

        Returns the (possibly updated) checks and repair count. Every branch is bounded
        by ``CODER_COLLAB_ROUNDS`` and every repair it triggers is a normal repair round,
        so a build with a peer attached can never loop longer than one without it.
        """
        if not self.duet_enabled:
            return checks, rounds

        # 1. Still failing after the local repairs → ask for a diagnosis, then repair
        #    with it as the brief.
        if not checks.get("ok") and any(c.get("runnable")
                                        for c in checks.get("checks", [])):
            brief = _DUET_BRIEF["failure"].format(
                name=plan.get("name", ""), summary=plan.get("summary", ""),
                rounds=rounds, files=self._design_block(plan),
                error=checks.get("log", "")[:3000])
            advice = self._consult(pid, "failure diagnosis", brief, cwd=root)
            if advice:
                rounds += 1
                self._update(pid, repairs=rounds,
                             progress=f"repairing {rounds} (peer diagnosis)")
                if self._repair(pid, plan, root, checks, advice=advice):
                    checks = self._verify(plan, root)
                    self._update(pid, checks=checks,
                                 verified=checks.get("ok", False))

        # 2. Passing → one critique of the finished project; act on it only if the peer
        #    named something concrete.
        if checks.get("ok"):
            brief = _DUET_BRIEF["critique"].format(
                name=plan.get("name", ""), summary=plan.get("summary", ""),
                files=self._design_block(plan))
            advice = self._consult(pid, "final critique", brief, cwd=root)
            if advice and not self._is_lgtm(advice):
                rounds += 1
                self._update(pid, repairs=rounds,
                             progress=f"repairing {rounds} (peer critique)")
                snapshot = self._snapshot(root, plan)
                if self._repair(pid, plan, root, checks, advice=advice):
                    after = self._verify(plan, root)
                    if after.get("ok"):
                        checks = after
                        self._update(pid, checks=checks, verified=True)
                    else:
                        # A critique is advice, not gospel. Undo the files (not just
                        # the verdict) — a passing build we cannot reproduce on disk is
                        # a lie, so roll the code back and keep the build we verified.
                        self._restore(pid, root, snapshot)
                        self._verify(plan, root)
                        self._update(pid, progress="peer critique reverted")
                        self._note(pid, "coder", "critique reverted",
                                   "the peer's suggested change broke the checks and "
                                   "was discarded; the verified version was restored")
        return checks, rounds

    def _plan(self, project: dict) -> dict | None:
        self._update(project["id"], progress="planning")
        raw = self._generate(_PLAN_SYSTEM.format(max_files=config.CODER_MAX_FILES),
                             _PLAN_USER.format(topic=project["topic"]))
        plan = _json_object(raw)
        if not plan or not isinstance(plan.get("files"), list):
            return None
        files = []
        for f in plan["files"][:config.CODER_MAX_FILES]:
            if not isinstance(f, dict):
                continue
            path = _clean_rel_path(str(f.get("path", "")))
            if path:
                files.append({"path": path,
                              "purpose": str(f.get("purpose", ""))[:300]})
        if not files:
            return None
        entry = _clean_rel_path(str(plan.get("entry") or "")) or files[0]["path"]
        plan["files"] = files
        plan["entry"] = entry
        plan["name"] = str(plan.get("name") or project["topic"])[:80]
        plan["summary"] = str(plan.get("summary") or "")[:600]
        plan["test_command"] = str(plan.get("test_command") or "")[:200]
        plan["language"] = str(plan.get("language") or "other").lower()
        return plan

    def _design_block(self, plan: dict) -> str:
        return "\n".join(f"- {f['path']}: {f['purpose']}" for f in plan["files"])

    def _lang(self, plan: dict) -> str:
        return _LANG_FENCE.get(plan.get("language", "other"), "text")

    def _write_files(self, pid: str, plan: dict, root: str) -> list[dict]:
        """Generate each file in design order, keeping earlier files in context."""
        written: list[dict] = []
        total = len(plan["files"])
        for i, spec in enumerate(plan["files"], start=1):
            path = spec["path"]
            self._update(pid, progress=f"writing {i}/{total}: {path}")
            context = self._written_block(plan, root, written, exclude=path)
            user = _FILE_USER.format(
                name=plan["name"], summary=plan["summary"] or project_topic(plan),
                language=plan["language"], entry=plan["entry"],
                files=self._design_block(plan), written=context, path=path,
                purpose=spec["purpose"] or "core implementation")
            raw = self._generate(_FILE_SYSTEM.format(path=path, lang=self._lang(plan)),
                                 user)
            blocks = extract_files(raw, single_path=path)
            target = next((b for b in blocks if b[0] == path), None) or (
                blocks[0] if blocks else None)
            if not target or not target[1].strip():
                continue                       # a file that failed to generate is skipped
            content = target[1][:config.CODER_MAX_FILE_CHARS]
            if not self._write(root, target[0], content):
                continue
            written.append({"path": target[0], "chars": len(content)})
            self._update(pid, files=written)
        return written

    def _written_block(self, plan: dict, root: str, written: list[dict],
                       exclude: str = "", limit: int = 6000) -> str:
        if not written:
            return "FILES ALREADY WRITTEN: none yet.\n\n"
        parts = ["FILES ALREADY WRITTEN (match their real interfaces exactly):"]
        for w in written:
            if w["path"] == exclude:
                continue
            try:
                with open(os.path.join(root, _clean_rel_path(w["path"]) or ""),
                          "r", encoding="utf-8", errors="replace") as fh:
                    body = fh.read()[:limit]
            except OSError:
                body = "(unreadable)"
            parts.append(f"\n--- {w['path']} ---\n{body}")
        return "\n".join(parts) + "\n\n"

    @staticmethod
    def _snapshot(root: str, plan: dict) -> dict[str, str]:
        """Every file's current contents, so a change can be undone afterwards."""
        snap: dict[str, str] = {}
        for spec in plan.get("files", []):
            full = safe_join(root, spec["path"])
            if not full or not os.path.isfile(full):
                continue
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    snap[spec["path"]] = fh.read()
            except OSError:
                continue
        return snap

    def _restore(self, pid: str, root: str, snap: dict[str, str]) -> None:
        """Put the files back after a peer's suggestion broke the build."""
        for rel, content in snap.items():
            self._write(root, rel, content)
        with self._lock:
            p = next((x for x in self.projects if x.get("id") == pid), None)
            if p is None:
                return
            files = [dict(f) for f in (p.get("files") or [])]
            for f in files:
                if f.get("path") in snap:
                    f["chars"] = len(snap[f["path"]])
            p["files"] = files
            self._save_locked()

    @staticmethod
    def _write(root: str, rel: str, content: str) -> bool:
        full = safe_join(root, rel)
        if full is None:
            return False
        try:
            os.makedirs(os.path.dirname(full) or root, exist_ok=True)
            tmp = f"{full}.jarvs-tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(content.rstrip("\n") + "\n")
            os.replace(tmp, full)
            return True
        except OSError:
            return False

    def _verify(self, plan: dict, root: str) -> dict:
        """Run the checks that prove the project works. Never raises."""
        checks: list[dict] = []
        if not config.CODER_EXEC:
            return {"ok": False, "log": "verification disabled (JARVS_CODER_EXEC=0)",
                    "checks": [], "runnable": False}
        sources = self._source_files(root)
        syntax = self._syntax_check(plan, root, sources)
        if syntax:
            checks.append(syntax)
        test = self._test_check(plan, root)
        if test:
            checks.append(test)
        log = "\n".join(f"$ {c['name']}\n{c['output']}" for c in checks).strip()
        runnable = any(c.get("runnable") for c in checks)
        ok = bool(checks) and all(c.get("ok") for c in checks)
        return {"ok": ok, "log": log[:8000], "checks": checks, "runnable": runnable}

    @staticmethod
    def _source_files(root: str) -> list[str]:
        out = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", "node_modules", ".git")]
            for fn in filenames:
                if os.path.splitext(fn)[1] in _CHECKABLE:
                    out.append(os.path.relpath(os.path.join(dirpath, fn), root)
                               .replace(os.sep, "/"))
        return sorted(out)

    def _run(self, argv: list[str], root: str) -> dict:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        env["CI"] = "1"
        try:
            proc = subprocess.run(argv, cwd=root, env=env, timeout=config.CODER_TEST_TIMEOUT,
                                  capture_output=True, text=True, shell=False)
            return {"ok": proc.returncode == 0, "code": proc.returncode,
                    "output": ((proc.stdout or "") + (proc.stderr or ""))[:4000]}
        except FileNotFoundError:
            return {"ok": False, "code": None, "output": f"not available: {argv[0]}"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "code": None,
                    "output": f"timed out after {config.CODER_TEST_TIMEOUT}s"}

    def _syntax_check(self, plan: dict, root: str, sources: list[str]) -> dict | None:
        lang = plan.get("language", "other")
        if lang in ("python", "other"):
            py = [s for s in sources if s.endswith(".py")]
            if not py:
                return None
            res = self._run([sys.executable, "-m", "py_compile", *py], root)
            return {"name": "syntax (python -m py_compile)", **res, "runnable": True}
        if lang in ("javascript", "typescript"):
            js = [s for s in sources if os.path.splitext(s)[1] in
                  (".js", ".mjs", ".cjs")]
            if not js:
                return None
            res = self._run(["node", "--check", *js], root)
            res["runnable"] = not res.get("output", "").startswith("not available:")
            return {"name": "syntax (node --check)", **res}
        return None

    def _test_check(self, plan: dict, root: str) -> dict | None:
        argv = sanitize_command(plan.get("test_command") or "")
        if argv is None and plan.get("language") in ("python", "other"):
            # No usable command from the model: fall back to pytest when tests exist.
            tests_dir = os.path.join(root, "tests")
            has_tests = os.path.isdir(tests_dir) or any(
                f.startswith("test_") or f.endswith("_test.py")
                for _, _, fs in os.walk(root) for f in fs)
            argv = [sys.executable, "-m", "pytest", "-q"] if has_tests else None
        if not argv:
            return None
        res = self._run(argv, root)
        unavailable = res.get("output", "").startswith("not available:")
        res["runnable"] = not unavailable
        return {"name": "tests (" + " ".join(os.path.basename(a) for a in argv) + ")",
                **res}

    def _repair(self, pid: str, plan: dict, root: str, checks: dict,
                advice: str = "") -> list[str]:
        """Feed the failure back to the model and rewrite the implicated files.

        ``advice`` is an optional second opinion (the peer model's diagnosis of the
        failure, or its critique of the finished project) appended to the brief — the
        coder still writes the code, the peer only steers.
        """
        error = checks.get("log", "")[:4000]
        targets = self._blame(plan, root, error)
        fixed: list[str] = []
        for rel in targets[:config.CODER_MAX_FIXES]:
            full = safe_join(root, rel)
            if full is None or not os.path.isfile(full):
                continue
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()[:12000]
            except OSError:
                continue
            purpose = next((f["purpose"] for f in plan["files"]
                            if f["path"] == rel), "")
            advice_block = ""
            if advice:
                advice_block = ("\nPEER REVIEW (a second model read the code and "
                                "the failure above; weigh it heavily, but only "
                                "make changes that are correct):\n"
                                + advice.strip()[:3000])
            user = _REPAIR_USER.format(
                name=plan["name"], summary=plan["summary"], path=rel,
                purpose=purpose, lang=self._lang(plan), content=content,
                error=error, advice=advice_block)
            raw = self._generate(_REPAIR_SYSTEM.format(path=rel, lang=self._lang(plan)),
                                 user)
            if CoderBackend.failed(raw):
                continue
            blocks = extract_files(raw, single_path=rel)
            target = next((b for b in blocks if b[0] == rel), None) or (
                blocks[0] if blocks else None)
            if not target or not target[1].strip():
                continue
            if self._write(root, target[0], target[1][:config.CODER_MAX_FILE_CHARS]):
                fixed.append(target[0])
                # Keep the recorded file sizes honest — the report is generated from
                # these, and a repaired file is a different size than the broken one.
                with self._lock:
                    p = next((x for x in self.projects if x.get("id") == pid), None)
                    if p is not None:
                        files = [dict(f) for f in (p.get("files") or [])]
                        entry = next((f for f in files if f.get("path") == target[0]),
                                     None)
                        size = len(target[1][:config.CODER_MAX_FILE_CHARS])
                        if entry:
                            entry["chars"] = size
                        else:
                            files.append({"path": target[0], "chars": size})
                        p["files"] = files
                        self._save_locked()
                self._update(pid, progress=f"repaired {target[0]}")
        return fixed

    @staticmethod
    def _blame(plan: dict, root: str, error: str) -> list[str]:
        """Which files does the error mention? (design order breaks ties)."""
        known = [f["path"] for f in plan.get("files", [])]
        if not known:
            return []
        mentioned = {m for path in known
                     for m in re.findall(re.escape(path), error or "")}
        named = re.findall(r"[\w./\\-]+\.(?:py|js|mjs|cjs|ts|go|rs)", error or "")
        for n in named:
            n = _clean_rel_path(n)
            if n in known:
                mentioned.add(n)
        ordered = [p for p in known if p in mentioned]
        # Nothing identifiable: repair the entry point first, then the rest.
        return ordered or known[:config.CODER_MAX_FIXES]

    @staticmethod
    def _tidy(root: str) -> None:
        """Drop the caches the checks leave behind — the deliverable is source code."""
        for dirpath, dirnames, _ in os.walk(root, topdown=False):
            for d in list(dirnames):
                if d in ("__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"):
                    shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)

    def _write_report(self, pid: str, plan: dict, root: str, checks: dict,
                      rounds: int) -> None:
        project = self._update(pid) or {}
        files = project.get("files") or []
        lines = [
            f"# {plan.get('name') or project.get('topic')}",
            "",
            f"**Topic:** {project.get('topic', '')}",
            f"**Language:** {plan.get('language', 'other')} · "
            f"**entry:** {plan.get('entry', '')}",
            "",
            plan.get("summary") or "",
            "",
            "## Files",
            "",
            *(f"- `{f['path']}` ({f['chars']} chars)" for f in files),
            "",
            "## Verification",
            "",
            f"- result: **{'PASS' if checks.get('ok') else 'FAIL'}**",
            f"- repair rounds: {rounds}",
        ]
        for c in checks.get("checks", []):
            lines.append(f"- {c['name']}: {'ok' if c.get('ok') else 'failed'}"
                         + ("" if c.get("ok") else f"\n\n```\n{c.get('output','')}\n```"))
        if not checks.get("checks"):
            lines.append(f"- {checks.get('log') or 'no checks could be run'}")
        collab = project.get("collab") or []
        if collab:
            lines += ["", "## Collaboration (Qwen \u2194 Claude)", ""]
            for turn in collab:
                who = ("Claude (review)" if turn.get("role") == "peer"
                       else config.CODER_MODEL + " (coder)")
                lines += [f"### {who} \u2014 {turn.get('kind', '')}", "",
                          (turn.get("text") or "").strip()[:2500], ""]
        duo = " with review from Claude" if collab else ""
        lines += ["", "_Built autonomously by JARVS \u2014 coder: "
                  f"{config.CODER_MODEL}{duo}._", ""]
        self._write(root, "JARVS_REPORT.md", "\n".join(lines))

    # -- worker --------------------------------------------------------------- #

    def _finish(self, pid: str, error: str = "") -> dict:
        with self._lock:
            p = next((x for x in self.projects if x.get("id") == pid), None)
            if p is None:
                return {}
            p["finished"] = time.time()
            if error:
                p["status"] = "failed"
                p["error"] = error[:1000]
                p["progress"] = "failed"
            else:
                p["status"] = "completed"
                p["error"] = ""
                p["result"] = self._result_text(p)
            self._save_locked()
            done = dict(p)
        if not error:
            if self._memory_fn:
                try:
                    self._memory_fn(
                        f"[Code project {done['id']}] {done.get('name') or done['topic']}"
                        f" — built from the topic \"{done['topic']}\" at {done.get('path')}."
                        f"\n{done.get('result', '')}")
                except Exception:  # noqa: BLE001
                    pass
        self._emit("failed" if error else "completed", done)
        return done

    @staticmethod
    def _result_text(project: dict) -> str:
        files = project.get("files") or []
        checks = project.get("checks") or {}
        verdict = ("checks passed" if checks.get("ok")
                   else ("could not be verified" if not checks.get("checks")
                         else "checks still failing"))
        peers = [t for t in (project.get("collab") or []) if t.get("role") == "peer"]
        duet = f" · {len(peers)} Claude review(s)" if peers else ""
        return (f"{len(files)} file(s), {project.get('language') or '?'} · {verdict}"
                + (f" after {project.get('repairs')} repair round(s)"
                   if project.get("repairs") else "")
                + duet
                + f"\n{project.get('path', '')}")

    def _worker(self) -> None:
        while True:
            project = self._claim()
            if not project:
                with self._lock:
                    if any(p.get("status") == "queued" for p in self.projects):
                        continue
                    self._thread = None
                return
            self._emit("started", project)
            self.build(project)

    # -- introspection -------------------------------------------------------- #

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(p) for p in self.projects][::-1]

    def recent(self, limit: int = 10) -> list[dict]:
        return self.list()[:limit]

    def get(self, project_id: str) -> dict | None:
        with self._lock:
            p = next((x for x in self.projects if x.get("id") == project_id), None)
            return dict(p) if p else None

    def remove(self, project_id: str) -> bool:
        with self._lock:
            p = next((x for x in self.projects if x.get("id") == project_id), None)
            if p is None:
                return False
            self.projects = [x for x in self.projects if x.get("id") != project_id]
            self._save_locked()
        if p.get("path") and os.path.isdir(p["path"]):
            shutil.rmtree(p["path"], ignore_errors=True)
        return True

    def counts(self) -> dict:
        with self._lock:
            c = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
            for p in self.projects:
                c[p.get("status", "queued")] = c.get(p.get("status", "queued"), 0) + 1
            return c

    def status(self) -> str:
        c = self.counts()
        return (f"coder: {c['queued']} queued, {c['running']} building, "
                f"{c['completed']} built, {c['failed']} failed "
                f"({config.CODER_MODEL}, exec {'on' if config.CODER_EXEC else 'off'})")

    @staticmethod
    def _slug(name: str, pid: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:40]
        return f"{slug or 'project'}-{pid}"


def project_topic(plan: dict) -> str:
    """Fallback summary line when the model gave none."""
    return f"{len(plan.get('files', []))} files, entry {plan.get('entry', '')}"
