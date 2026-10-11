"""
routines.py — recurring scheduled tasks (the capability JARVS used to only *claim*).

A :class:`RoutineScheduler` keeps a persisted list of recurring routines and a daemon
that, when one is due, enqueues a :class:`~jarvs.background.BackgroundManager` job for
it — so each run does the same bounded analysis / live web research as any background
task, and its result surfaces as a digest. This is the JARVS port of JCN3690's
self-authored "routines", deliberately deferred until now.

A plain-language scheduling request ("add daily searches about X and Y") is turned into
structured routines by one bounded LLM extraction pass (:meth:`plan_and_create`), so the
assistant's "I've scheduled that" is backed by a real routine instead of being a
confabulation.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Callable

from . import config
from .assoc_memory import _extract_json_array

# --- interval parsing + intent detection (pure) ------------------------------- #

_UNIT_SEC = {"min": 60, "minute": 60, "hour": 3600, "hr": 3600, "day": 86400,
             "week": 604800}
_EVERY_RE = re.compile(r"every\s+(\d+)\s*(min(?:ute)?s?|hours?|hrs?|days?|weeks?)", re.I)

_SCHEDULE_RE = re.compile(
    r"\b(?:schedul\w*|recurring|repeat\w*|periodically|regularly|"
    r"every\s+(?:day|morning|hour|week|\d+\s*(?:min\w*|hours?|days?|weeks?))|"
    r"(?:twice|once)\s+(?:a|per)\s+(?:day|week|hour)|daily|hourly|weekly|"
    r"each\s+(?:day|morning|hour|week))\b", re.I)


def parse_interval(text: str, default: int | None = None) -> int:
    """Map a phrase to seconds. 'daily'→86400, 'every 6 hours'→21600, etc."""
    t = (text or "").lower()
    m = _EVERY_RE.search(t)
    if m:
        n = max(1, int(m.group(1)))
        unit = m.group(2).rstrip("s")
        for key, secs in _UNIT_SEC.items():
            if unit.startswith(key[:3]):
                return n * secs
    if "twice a day" in t or "twice per day" in t:
        return 43200
    if "hourly" in t or "every hour" in t:
        return 3600
    if "weekly" in t or "every week" in t or "once a week" in t:
        return 604800
    if "daily" in t or "every day" in t or "each day" in t or "each morning" in t \
            or "every morning" in t or "a day" in t:
        return 86400
    return default if default is not None else config.ROUTINE_DEFAULT_INTERVAL


def is_schedule_request(text: str) -> bool:
    return bool(_SCHEDULE_RE.search(text or ""))


# --- clock-time anchoring ----------------------------------------------------- #

_NAMED_TIMES = {"midnight": (0, 0), "morning": (8, 0), "noon": (12, 0),
                "afternoon": (14, 0), "evening": (18, 0), "night": (21, 0)}
_HHMM_RE = re.compile(r"\b(?:at\s+)?([01]?\d|2[0-3]):([0-5]\d)\s*(am|pm)?\b", re.I)
_AMPM_RE = re.compile(r"\b(1[0-2]|0?\d)\s*(am|pm)\b", re.I)


def parse_time_of_day(text: str) -> tuple[int, int] | None:
    """Parse a wall-clock time from a phrase: '7am', '08:00', 'at 19:30', 'every
    morning'. Returns (hour, minute) in 24h, or None. Deliberately strict so an
    interval like 'every 6 hours' is NOT read as 6:00."""
    t = (text or "").lower()
    m = _HHMM_RE.search(t)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if m.group(3) == "pm" and h < 12:
            h += 12
        elif m.group(3) == "am" and h == 12:
            h = 0
        return (h % 24, mn)
    m = _AMPM_RE.search(t)
    if m:
        h = int(m.group(1))
        if m.group(2).lower() == "pm" and h < 12:
            h += 12
        elif m.group(2).lower() == "am" and h == 12:
            h = 0
        return (h % 24, 0)
    for name, hm in _NAMED_TIMES.items():
        if re.search(rf"\b{name}\b", t):
            return hm
    return None


def _next_at(hour: int, minute: int, interval_sec: int) -> float:
    """Next epoch at local hour:minute that is strictly in the future, stepping by
    interval_sec (so daily anchors to that time each day, weekly each week)."""
    now = datetime.now().astimezone()
    cand = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    step = timedelta(seconds=max(60, interval_sec))
    while cand <= now:
        cand += step
    return cand.timestamp()


_MANAGE_VERB_RE = re.compile(
    r"\b(reschedul\w*|change|move|adjust|update|set|pause|resume|unpause|cancel|"
    r"delete|remove|stop|disable|enable|turn (?:off|on))\b", re.I)
_ROUTINE_NOUN_RE = re.compile(
    r"\b(routine|routines|schedule[d]?|digest|recurring|daily|hourly|weekly|"
    r"search(?:es)?|updates?|frequenc\w*|interval)\b", re.I)
_TIME_WORD_RE = re.compile(
    r"\b(\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm)|morning|noon|afternoon|evening|night|"
    r"midnight|earlier|later)\b", re.I)


def is_routine_command(text: str) -> bool:
    """A request to MANAGE an existing routine (change time/frequency, pause, cancel)
    — as opposed to creating one. A management verb plus a routine noun or a time."""
    t = text or ""
    return bool(_MANAGE_VERB_RE.search(t)
                and (_ROUTINE_NOUN_RE.search(t) or _TIME_WORD_RE.search(t)))


def human_interval(secs: int) -> str:
    for label, s in (("week", 604800), ("day", 86400), ("hour", 3600), ("minute", 60)):
        if secs % s == 0:
            n = secs // s
            return f"every {n} {label}{'s' if n != 1 else ''}" if n != 1 else f"{label}ly" \
                if label != "day" else "daily"
    return f"every {secs}s"


def schedule_label(routine: dict) -> str:
    base = human_interval(int(routine.get("interval_sec", 86400)))
    if routine.get("at_hour") is not None:
        return f"{base} at {routine['at_hour']:02d}:{routine.get('at_minute', 0):02d}"
    return base


_PLAN_SYSTEM = """You convert a user's scheduling request into structured recurring \
routines for an assistant that can run one web-research task per routine on a timer.

Output ONLY a JSON array (no prose). Each element:
{"description": "<=6-word label", "task": "a self-contained instruction to research \
and summarize, phrased so it works every time it runs", "interval": "hourly|daily|\
weekly|every N hours|every N days"}

Make one element per DISTINCT topic the user named. If the request is not about \
scheduling recurring work, output []."""


_ROUTE_SYSTEM = """You manage an assistant's recurring routines. The current routines \
are (id · description · schedule):
{routines}

The user's message may create a new routine OR change an existing one. Output ONLY a \
JSON array of actions (no prose). Each action is one of:
{{"action":"create","description":"<=6 words","task":"research instruction","interval":"hourly|daily|weekly|every N hours|every N days","time":"HH:MM or null"}}
{{"action":"reschedule","id":"<existing id>","interval":"<new frequency or null>","time":"<new HH:MM or null>"}}
{{"action":"pause","id":"<existing id>"}}
{{"action":"resume","id":"<existing id>"}}
{{"action":"remove","id":"<existing id>"}}

Rules:
- To change WHEN or HOW OFTEN an existing routine runs, use "reschedule" with its id — \
never "create" a second one.
- Identify which routine the user means by matching its description.
- Use only ids from the list above. Times are 24-hour HH:MM.
- If nothing should change, output []."""


class RoutineScheduler:
    def __init__(self, *, background, store_path: str,
                 generate_fn: Callable[[str, str, bool], str],
                 on_event: Callable[[str, dict], None] | None = None):
        self._bg = background
        self._store_path = store_path
        self._generate = generate_fn
        self._on_event = on_event
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.routines: list[dict] = self._load()
        self.events: list[dict] = []

    # -- persistence ---------------------------------------------------------- #

    def _load(self) -> list[dict]:
        try:
            with open(self._store_path) as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

    def _save_locked(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._store_path) or ".", exist_ok=True)
            tmp = f"{self._store_path}.tmp"
            with open(tmp, "w") as fh:
                json.dump(self.routines, fh, indent=2)
            os.replace(tmp, self._store_path)
        except OSError:
            pass

    def _emit(self, kind: str, routine: dict) -> None:
        ev = {"kind": kind, "id": routine.get("id"),
              "description": routine.get("description", ""), "t": time.time()}
        with self._lock:
            self.events.append(ev)
            del self.events[:-50]
        if self._on_event:
            try:
                self._on_event(kind, routine)
            except Exception:  # noqa: BLE001
                pass

    def drain_events(self) -> list[dict]:
        with self._lock:
            out, self.events = self.events, []
        return out

    # -- CRUD ----------------------------------------------------------------- #

    def add(self, description: str, task: str, interval_sec: int,
            first_delay: int | None = None,
            at: tuple[int, int] | None = None) -> dict:
        now = time.time()
        interval_sec = max(60, int(interval_sec))
        at_hour, at_minute = (at if at else (None, None))
        if at:
            next_run = _next_at(at_hour, at_minute, interval_sec)
        elif first_delay is not None:
            next_run = now + first_delay
        else:
            next_run = now + interval_sec
        routine = {
            "id": uuid.uuid4().hex[:8],
            "description": " ".join((description or task).split())[:80],
            "task": " ".join((task or "").split())[:500],
            "interval_sec": interval_sec,
            "at_hour": at_hour,
            "at_minute": at_minute,
            "enabled": True,
            "created": now,
            "last_run": None,
            "next_run": next_run,
            "runs": 0,
        }
        with self._lock:
            self.routines.append(routine)
            del self.routines[:-40]
            self._save_locked()
        self._emit("created", routine)
        return dict(routine)   # the scheduler loop (started via start()) picks it up

    def plan_and_create(self, user_input: str) -> list[dict]:
        """Extract recurring routines from a plain-language request via one LLM pass,
        then create them. Returns the created routines (empty on nothing/failure)."""
        try:
            raw = self._generate(_PLAN_SYSTEM, user_input, False)
        except Exception:  # noqa: BLE001
            return []
        items = _extract_json_array(raw) or []
        created = []
        for it in items:
            if not isinstance(it, dict):
                continue
            task = str(it.get("task", "")).strip()
            if len(task) < 8:
                continue
            interval = parse_interval(str(it.get("interval", "")),
                                      default=config.ROUTINE_DEFAULT_INTERVAL)
            created.append(self.add(it.get("description", task), task, interval))
        return created

    def reschedule(self, routine_id: str, interval_sec: int | None = None,
                   at: tuple[int, int] | None = None, clear_time: bool = False) -> bool:
        """Change a routine's frequency and/or its anchored clock time, recomputing
        the next run. Either argument may be omitted to leave it unchanged."""
        with self._lock:
            for r in self.routines:
                if r.get("id") != routine_id:
                    continue
                if interval_sec:
                    r["interval_sec"] = max(60, int(interval_sec))
                if at is not None:
                    r["at_hour"], r["at_minute"] = at
                elif clear_time:
                    r["at_hour"] = r["at_minute"] = None
                if r.get("at_hour") is not None:
                    r["next_run"] = _next_at(r["at_hour"], r["at_minute"], r["interval_sec"])
                else:
                    r["next_run"] = time.time() + r["interval_sec"]
                self._save_locked()
                self._emit("rescheduled", dict(r))
                return True
        return False

    def handle_request(self, user_input: str) -> list[tuple[str, str]]:
        """Route a plain-language routine request to concrete actions via one bounded
        LLM pass that sees the current routines, then apply them. Handles both creating
        a routine and managing an existing one (reschedule / pause / resume / remove).
        Returns a list of (action, label) for surfacing. Best-effort; never raises."""
        try:
            listing = "\n".join(
                f"- {r['id']} · {r['description']} ({schedule_label(r)})"
                for r in self.list()) or "(none yet)"
            raw = self._generate(_ROUTE_SYSTEM.format(routines=listing), user_input, False)
        except Exception:  # noqa: BLE001
            return []
        actions = _extract_json_array(raw) or []
        valid = {r["id"] for r in self.list()}
        applied: list[tuple[str, str]] = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            act = str(a.get("action", "")).strip().lower()
            rid = a.get("id")
            interval = (parse_interval(str(a["interval"]), default=0) or None
                        if a.get("interval") else None)
            at = parse_time_of_day(str(a["time"])) if a.get("time") else None
            if act == "create":
                task = str(a.get("task", "")).strip()
                if len(task) >= 8:
                    r = self.add(a.get("description", task), task,
                                 interval or config.ROUTINE_DEFAULT_INTERVAL, at=at)
                    applied.append(("created", r["description"]))
            elif rid in valid and act == "reschedule" and (interval or at):
                if self.reschedule(rid, interval_sec=interval, at=at):
                    applied.append(("rescheduled", rid))
            elif rid in valid and act in ("pause", "disable"):
                self.set_enabled(rid, False); applied.append(("paused", rid))
            elif rid in valid and act in ("resume", "enable"):
                self.set_enabled(rid, True); applied.append(("resumed", rid))
            elif rid in valid and act in ("remove", "delete", "cancel"):
                self.remove(rid); applied.append(("removed", rid))
        return applied

    def remove(self, routine_id: str) -> bool:
        with self._lock:
            n = len(self.routines)
            self.routines = [r for r in self.routines if r.get("id") != routine_id]
            changed = len(self.routines) != n
            if changed:
                self._save_locked()
        return changed

    def set_enabled(self, routine_id: str, enabled: bool) -> bool:
        with self._lock:
            for r in self.routines:
                if r.get("id") == routine_id:
                    r["enabled"] = bool(enabled)
                    self._save_locked()
                    return True
        return False

    def run_now(self, routine_id: str) -> bool:
        with self._lock:
            r = next((x for x in self.routines if x.get("id") == routine_id), None)
        return self._fire(r) if r else False

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.routines]

    # -- scheduling ----------------------------------------------------------- #

    def _fire(self, routine: dict) -> bool:
        """Enqueue a background job for this routine and advance its schedule."""
        if self._bg is None:
            return False
        job = self._bg.queue(routine["task"], context=f"(Recurring routine: "
                             f"{routine['description']})")
        now = time.time()
        with self._lock:
            for r in self.routines:
                if r.get("id") == routine["id"]:
                    r["last_run"] = now
                    # re-anchor to the clock time if one is set, else interval from now
                    if r.get("at_hour") is not None:
                        r["next_run"] = _next_at(r["at_hour"], r["at_minute"],
                                                 r["interval_sec"])
                    else:
                        r["next_run"] = now + r["interval_sec"]
                    r["runs"] = int(r.get("runs", 0)) + 1
                    self._save_locked()
                    fired = dict(r)
                    break
            else:
                fired = routine
        self._emit("fired", fired)
        return job is not None

    def due(self) -> list[dict]:
        now = time.time()
        with self._lock:
            return [dict(r) for r in self.routines
                    if r.get("enabled") and now >= float(r.get("next_run", 0))]

    def _loop(self) -> None:
        while not self._stop.wait(config.ROUTINE_TICK):
            for r in self.due():
                try:
                    self._fire(r)
                except Exception:  # noqa: BLE001
                    pass

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop,
                                            name="jarvs-routines", daemon=True)
            self._thread.start()

    def start(self) -> None:
        """Start the scheduler (fires any routine already past due on the next tick)."""
        self._ensure_thread()

    # -- introspection -------------------------------------------------------- #

    def status(self) -> str:
        with self._lock:
            active = sum(1 for r in self.routines if r.get("enabled"))
            return f"routines: {active} active / {len(self.routines)} total"
