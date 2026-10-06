"""
background.py — autonomous deferred work (the agent-like layer).

JARVS can accept a task now and finish it later, off the conversation path. A
:class:`BackgroundManager` keeps a small persisted queue and a single worker daemon
that drains it: each job gathers its own context (optionally doing one live web
search / URL fetch), runs a bounded analysis through the model, stores the result in
episodic memory, and emits a status event. Transient backend timeouts are retried
with backoff; anything else fails the job.

This is the JARVS counterpart to JCN3690's background-thought system, rewritten as a
self-contained object (no module globals). Triggering is kept deliberately simple for
now: the user asks for deferred work in plain language (``should_auto_queue``) or via
an explicit command; model-emitted ``[[THINK:]]`` markers are also supported.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from typing import Callable

from . import config, web_access

# --- trigger detection (pure, testable) --------------------------------------- #

_AUTO_PATTERNS = {
    "background": re.compile(
        r"\b(?:in the background|background (?:task|work|thinking|analysis|research))\b", re.I),
    "deliberate": re.compile(
        r"\b(?:take your time|no rush|don['’]?t need (?:a )?(?:quick|immediate) answer)\b", re.I),
    "completion": re.compile(
        r"\b(?:let me know|notify me|tell me|ping me|report back).{0,60}\b(?:when|once).{0,30}"
        r"\b(?:finished|done|ready|complete|completed)\b", re.I | re.S),
    # An actionable "go do a piece of work" verb. On its own this is just a request;
    # paired with a background/deliberate/completion signal it means "do it async".
    "actionable": re.compile(
        r"\b(?:look into|research|dig into|investigate|work on|compile|"
        r"summari[sz]e|compare|find out|keep an eye on|monitor)\b", re.I),
}

_THINK_MARKER = re.compile(r"\[\[THINK:\s*(.+?)\]\]", re.I | re.S)


def background_signals(user_input: str) -> set[str]:
    return {name for name, pat in _AUTO_PATTERNS.items() if pat.search(user_input or "")}


def should_auto_queue(user_input: str) -> bool:
    """High-precision test for a user asking JARVS to work on something in the
    background. Requires the explicit 'background' phrasing plus one more signal, or
    two of the softer signals, so an ordinary request is never deferred by mistake."""
    s = background_signals(user_input)
    if "background" in s and len(s) >= 1:
        return True
    return len(s) >= 2


def extract_think_markers(reply: str) -> tuple[str, list[str]]:
    """Pull any ``[[THINK: task]]`` markers out of a reply. Returns the cleaned reply
    (markers removed) and the list of task strings the model deferred."""
    tasks = [m.strip() for m in _THINK_MARKER.findall(reply or "") if m.strip()]
    clean = _THINK_MARKER.sub("", reply or "")
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, tasks


_NONACTIONABLE = re.compile(
    r"^\s*(?:ok|okay|sure|done|thanks?|yes|no|nothing|n/?a|none)\s*[.!]?\s*$", re.I)


class BackgroundManager:
    def __init__(self, *, generate_fn: Callable[[str, str, bool], str],
                 store_path: str, persona: str = "",
                 memory_fn: Callable[[str], None] | None = None,
                 on_event: Callable[[str, dict], None] | None = None):
        self._generate = generate_fn
        self._store_path = store_path
        self._persona = persona
        self._memory_fn = memory_fn           # called with the finished result text
        self._on_event = on_event
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self.jobs: list[dict] = self._load()
        self.events: list[dict] = []          # recent events (ring), for REPL/UI flush

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
                json.dump(self.jobs, fh, indent=2)
            os.replace(tmp, self._store_path)
        except OSError:
            pass

    # -- events --------------------------------------------------------------- #

    def _emit(self, kind: str, job: dict) -> None:
        ev = {"kind": kind, "id": job.get("id"), "task": job.get("task", ""),
              "status": job.get("status"), "t": time.time()}
        with self._lock:
            self.events.append(ev)
            del self.events[:-50]
        if self._on_event:
            try:
                self._on_event(kind, job)
            except Exception:  # noqa: BLE001
                pass

    def drain_events(self) -> list[dict]:
        """Pop and return buffered events (the REPL flushes these before a prompt)."""
        with self._lock:
            out, self.events = self.events, []
        return out

    # -- queue ---------------------------------------------------------------- #

    def queue(self, task: str, context: str = "") -> dict | None:
        task = " ".join((task or "").split()).strip()[:700]
        if len(task) < 8 or _NONACTIONABLE.match(task):
            return None
        with self._lock:
            active = [j for j in self.jobs if j.get("status") in ("queued", "running")]
            if len(active) >= config.BACKGROUND_MAX_PENDING:
                return None
            for j in active:  # dedup near-identical pending work
                if j.get("task", "").casefold() == task.casefold():
                    return dict(j)
            job = {"id": uuid.uuid4().hex[:8], "task": task,
                   "context": (context or "")[:config.BACKGROUND_CONTEXT_CHARS],
                   "status": "queued", "created": time.time(),
                   "result": "", "error": "", "attempts": 0}
            self.jobs.append(job)
            del self.jobs[:-40]               # bound the store
            self._save_locked()
        self._emit("queued", job)
        self._ensure_worker()
        return dict(job)

    def _claim(self) -> dict | None:
        with self._lock:
            for j in self.jobs:
                if j.get("status") == "queued":
                    j["status"] = "running"
                    j["started"] = time.time()
                    self._save_locked()
                    return dict(j)
        return None

    def _ensure_worker(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._thread = threading.Thread(target=self._worker,
                                            name="jarvs-background", daemon=True)
            self._thread.start()
            return True

    # -- web gathering (runs on the worker thread) ---------------------------- #

    def _web_context(self, task: str) -> tuple[str, str]:
        if not (config.ALLOW_WEB and config.BACKGROUND_WEB) or not web_access.DDGS_AVAILABLE:
            return "", ""
        blocks: list[str] = []
        for url in web_access.extract_urls(task)[:2]:
            try:
                f = web_access.fetch_url_content(url)
                blocks.append(f"URL: {f['url']}\nTitle: {f['title']}\n"
                              f"Extracted text:\n{f['text']}")
            except Exception as e:  # noqa: BLE001
                blocks.append(f"URL: {url}\nFetch failed: {e}")
        query = ""
        if web_access.needs_web_search(task) or blocks:
            query = task[:200]
            results = web_access.web_search(query)
            if (results.strip()
                    and not results.startswith(("Search failed:", "Search unavailable:"))
                    and results != "Search returned no results."):
                blocks.append(f"Live web search — query: {query}\n{results}")
            else:
                query = ""
        if not blocks:
            return "", ""
        block = ("--- LIVE WEB CONTEXT (gathered just now for this task) ---\n"
                 + "\n\n".join(blocks))
        return block[:config.BACKGROUND_CONTEXT_CHARS], query

    # -- execution ------------------------------------------------------------ #

    def _run_job(self, job: dict) -> str:
        web_block, query = self._web_context(job["task"])
        if query:
            self._set_field(job["id"], web_used=True, web_query=query)
        if web_block:
            web_clause = ("You have LIVE WEB RESULTS below, gathered just now for this "
                          "task; ground your analysis in them and say plainly where "
                          "the material is thin.")
        else:
            web_clause = ("You have no shell, files, or live web for this task; reason "
                          "only over the captured context, and say so if it lacks what "
                          "the task needs rather than inventing facts.")
        persona = (self._persona + "\n\n") if self._persona else ""
        system = (f"{persona}[BOUNDED BACKGROUND TASK]\n"
                  f"Complete one task using the context below. {web_clause} Return a "
                  f"finished, useful result — not a plan or a promise. Do not claim you "
                  f"performed actions outside this reasoning call.")
        captured = job.get("context") or "(No stored context was available.)"
        if web_block:
            captured = f"{captured}\n\n{web_block}"
        user = f"Task:\n{job['task']}\n\nCaptured context:\n{captured}"
        return (self._generate(system, user, True) or "").strip()

    @staticmethod
    def _failed(result: str) -> bool:
        r = (result or "").strip().lower()
        return not r or r.startswith("error connecting to the backend")

    def _worker(self) -> None:
        while True:
            job = self._claim()
            if not job:
                with self._lock:
                    if any(j.get("status") == "queued" for j in self.jobs):
                        continue
                    self._thread = None
                return
            self._emit("started", job)
            try:
                result = self._run_job(job)
                if self._failed(result):
                    self._handle_failure(job["id"], result or "empty result")
                    continue
                finished = self._finish(job["id"], result=result)
                if self._memory_fn:
                    try:
                        self._memory_fn(f"[Background task {job['id']}] {job['task']}\n{result}")
                    except Exception:  # noqa: BLE001
                        pass
                self._emit("completed", finished)
            except Exception as e:  # noqa: BLE001
                self._handle_failure(job["id"], str(e))

    def _handle_failure(self, job_id: str, error: str) -> None:
        transient = any(t in error.lower() for t in ("timed out", "timeout", "connection"))
        if config.BACKGROUND_RETRIES > 0 and transient:
            with self._lock:
                for j in self.jobs:
                    if j.get("id") == job_id and int(j.get("attempts", 0)) < config.BACKGROUND_RETRIES:
                        j["attempts"] = int(j.get("attempts", 0)) + 1
                        j["status"] = "queued"
                        j["error"] = ""
                        j.pop("started", None)
                        self._save_locked()
                        requeued = dict(j)
                        break
                else:
                    requeued = None
            if requeued is not None:
                self._emit("retrying", requeued)
                if config.BACKGROUND_RETRY_BACKOFF > 0:
                    time.sleep(config.BACKGROUND_RETRY_BACKOFF)
                return
        self._emit("failed", self._finish(job_id, error=error))

    def _finish(self, job_id: str, result: str = "", error: str = "") -> dict:
        with self._lock:
            job = next((j for j in self.jobs if j.get("id") == job_id), None)
            if job is None:
                return {}
            job["finished"] = time.time()
            job["context"] = ""               # no longer needed; can be large
            if error:
                job["status"] = "failed"
                job["error"] = error[:1000]
            else:
                job["status"] = "completed"
                job["result"] = result[:12000]
            self._save_locked()
            return dict(job)

    def _set_field(self, job_id: str, **fields) -> None:
        with self._lock:
            for j in self.jobs:
                if j.get("id") == job_id:
                    j.update(fields)
                    self._save_locked()
                    return

    # -- introspection -------------------------------------------------------- #

    def recent(self, limit: int = 10) -> list[dict]:
        with self._lock:
            return [dict(j) for j in self.jobs[-limit:]][::-1]

    def counts(self) -> dict:
        with self._lock:
            c = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
            for j in self.jobs:
                c[j.get("status", "queued")] = c.get(j.get("status", "queued"), 0) + 1
            return c

    def status(self) -> str:
        c = self.counts()
        web = "web on" if (config.ALLOW_WEB and config.BACKGROUND_WEB
                           and web_access.DDGS_AVAILABLE) else "web off"
        return (f"background: {c['queued']} queued, {c['running']} running, "
                f"{c['completed']} done, {c['failed']} failed ({web})")
