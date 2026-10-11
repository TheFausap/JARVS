"""
chat.py — the JARVS reply loop (the spine).

A :class:`JarvsSession` holds the persona head, the running conversation, and the
voice, and turns a user line into a spoken reply. It is deliberately small: the
heavier subsystems (external memory, meta-dreaming) attach at the marked seams
rather than living here, so the spine stays readable and testable on its own.

Run the REPL with:  python -m jarvs.chat
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid

from . import client, config, system_sense, web_access
from .background import BackgroundManager, should_auto_queue, extract_think_markers
from .claude_bridge import ClaudeBridge, is_code_request
from .rules import RulesStore, is_rule_request, implies_web_verify
from .routines import (
    RoutineScheduler, is_schedule_request, is_routine_command, schedule_label,
)
from .critic import Critic
from .memory import Memory, is_remember_request, remember_from_request
from .meta_dreaming import MetaDreamer
from .self_model import SelfModel, is_self_query
from .voice import JarvsVoice

# The persona head. JARVS is a calm, precise, lightly wry assistant — competent and
# unhurried, never obsequious. Kept short on purpose: durable behaviour will come
# from the external stack (memory, consolidated adaptations), not a bloated prompt.
DEFAULT_SYSTEM = (
    "You are JARVS, a personal AI assistant in the vein of JARVIS: calm, precise, "
    "and quietly witty. You address the user respectfully and get to the point. "
    "You are concise by default and expand only when asked or when the task needs "
    "it. You never invent facts; when you are unsure, you say so plainly. Your "
    "replies are spoken aloud, so you write clean prose — no markdown tables or "
    "code fences unless the user is clearly reading, not listening."
)


def distill_search_query(text: str, generate_fn) -> str:
    """Turn a conversational message into a tight web-search query — just the key
    entities/terms — so a rambling ask ("I admire your stubbornness, but can you look
    online for Gemma 3 and Gemma 4?") searches for "Gemma 3 Gemma 4", not "stubbornness".
    Falls back to the cleaned raw text if the model returns nothing."""
    raw = " ".join((text or "").split())
    try:
        q = (generate_fn(
            "Rewrite the user's message as a concise web-search query: keep only the key "
            "entities and terms, drop pleasantries and meta-commentary, no punctuation or "
            "quotes, at most 12 words. Output only the query.", raw, False) or "").strip()
    except Exception:  # noqa: BLE001
        q = ""
    q = " ".join(q.split()).strip(" \"'")[:200]
    return q or raw[:200]


class JarvsSession:
    """One conversation. Thread-unsafe by design — one session per user loop."""

    def __init__(self, *, system: str = DEFAULT_SYSTEM,
                 voice: JarvsVoice | None = None, speak: bool = True,
                 memory: Memory | None = None):
        self.system = system
        self.history: list[dict] = []          # [{"role": "...", "content": "..."}]
        self.voice = voice if voice is not None else JarvsVoice(log=self._log)
        self.speak = speak
        # External memory: durable across sessions, degradable to a no-op. The dream
        # pass borrows the bookkeeping generate(system, user) path.
        self.memory = memory if memory is not None else Memory(
            generate_fn=lambda s, u: client.generate(s, u, reason=False))

        # Self-critique + meta-dreaming. The session id is the diversity axis
        # meta-dreaming promotes across, so one conversation can't reshape the critic.
        self.session_id = uuid.uuid4().hex[:12]
        self.meta_dreamer: MetaDreamer | None = None
        self.critic: Critic | None = None
        if config.METADREAM_ENABLED:
            self.meta_dreamer = MetaDreamer(
                log_path=os.path.join(str(config.DATA_DIR), "critique_log.jsonl"),
                store_path=os.path.join(str(config.DATA_DIR), "meta_focus.json"),
                generate_fn=client.generate,
                known_names=("jarvs",),
                promote_min_pairs=config.METADREAM_PROMOTE,
                distill_min_new=config.METADREAM_MIN_NEW)
        if config.CRITIC_ENABLED:
            focus_fn = self.meta_dreamer.focus_directive if self.meta_dreamer else None
            self.critic = Critic(
                log_path=os.path.join(str(config.DATA_DIR), "critique_log.jsonl"),
                session_id=self.session_id,
                generate_fn=client.generate,
                focus_fn=focus_fn)

        # Autonomous background work (deferred tasks + optional web research).
        self.background: BackgroundManager | None = None
        if config.BACKGROUND_ENABLED:
            self.background = BackgroundManager(
                generate_fn=client.generate,
                store_path=os.path.join(str(config.DATA_DIR), "background_jobs.json"),
                persona=self.system,
                memory_fn=lambda t: self.memory.remember_fact(t, source="background"))

        # Recurring routines: a scheduler that fires background jobs on a timer. Needs
        # the background manager to run anything, so it is gated on it existing.
        self.routines: RoutineScheduler | None = None
        if config.ROUTINES_ENABLED and self.background is not None:
            self.routines = RoutineScheduler(
                background=self.background,
                store_path=os.path.join(str(config.DATA_DIR), "routines.json"),
                generate_fn=client.generate)
            self.routines.start()

        # Governed conduit for delegating coding to Claude Code (off by default).
        # JARVS never holds the credentials; the Agent SDK authenticates separately.
        self.claude_bridge: ClaudeBridge | None = None
        if config.CLAUDE_BRIDGE_ENABLED:
            self.claude_bridge = ClaudeBridge(
                generate_fn=client.generate,
                ledger_path=os.path.join(str(config.DATA_DIR), "claude_ledger.json"),
                exchanges_path=os.path.join(str(config.DATA_DIR),
                                            "claude_exchanges.json"),
                cwd=config.CLAUDE_BRIDGE_CWD,
                # Feed each outcome back as a durable fact so JARVS learns from its own
                # missteps and from Claude's observations about the codebase.
                memory_fn=lambda t: self.memory.remember_fact(t, source="claude_bridge"))

        # Standing rules: durable user-set directives injected every turn (and the
        # actionable web-verify flag the reply path consults).
        self.rules: RulesStore | None = None
        if config.RULES_ENABLED:
            self.rules = RulesStore(
                os.path.join(str(config.DATA_DIR), "rules.json"))

        # Self-model: JARVS's dossier about its own architecture. Needs the memory
        # embedder; consulted only when the user asks JARVS about itself.
        self.self_model: SelfModel | None = None
        if config.SELF_MODEL_ENABLED and self.memory.enabled:
            try:
                self.self_model = SelfModel(
                    str(config.SELF_DIR), self.memory._embed,
                    recall_k=config.SELF_RECALL_K, min_sim=config.SELF_RECALL_MIN)
            except Exception:  # noqa: BLE001 — never block startup on the dossier
                self.self_model = None

        # Turn counter driving the automatic dream (associative consolidation) cadence.
        self._turns_since_dream = 0

    # -- logging seam ---------------------------------------------------------- #

    @staticmethod
    def _log(msg: str) -> None:
        print(f"  · {msg}", file=sys.stderr)

    # -- history budgeting ----------------------------------------------------- #

    def _budget(self) -> int:
        """Tokens available for the system head + history, leaving room for the reply
        and a safety margin inside the backend's context slot."""
        return config.SLOT_CTX - config.GEN_RESERVE - config.SAFETY_MARGIN

    def _fit_history(self) -> list[dict]:
        """Keep the most recent turns that fit the budget alongside the system head.
        Drops from the OLDEST end so the live context is preserved — the external
        memory layer (step 3) will later recover what gets dropped here."""
        budget = self._budget() - client.count_tokens(self.system)
        kept: list[dict] = []
        for msg in reversed(self.history):
            cost = client.count_tokens(msg["content"]) + 4  # ~role/format overhead
            if budget - cost < 0 and kept:
                break
            budget -= cost
            kept.append(msg)
        kept.reverse()
        return kept

    # -- the reply path -------------------------------------------------------- #

    def build_system(self, user_input: str) -> str:
        """Assemble the system head for this turn: the persona plus any relevant
        recollections. (Meta-dreaming's focus directive goes to the CRITIC's prompt,
        not here — it sharpens how replies are judged, not how they are written.)"""
        head = self.system
        recall = self.memory.recall_block(user_input)
        if recall:
            head += "\n\n" + recall
        # Ground any host/time/location question in real OS telemetry (geolocation
        # only when the question is actually about location, so we don't ping the
        # geo service for a plain "what's the CPU load?").
        if config.SYSTEM_SENSE and system_sense.needs_system_info(user_input):
            head += "\n\n" + system_sense.summary_block(
                include_geo=system_sense.needs_location(user_input))
        # When the user asks JARVS about itself, add the relevant dossier chunks.
        if self.self_model is not None and is_self_query(user_input):
            block = self.self_model.self_block(user_input)
            if block:
                head += "\n\n" + block
        # Durable standing directives the user has set.
        if self.rules is not None:
            rules_block = self.rules.block()
            if rules_block:
                head += "\n\n" + rules_block
        # Live web: tell JARVS it CAN search (so it never denies having internet), and
        # when the turn needs current facts, fetch them and inject a [WEB] block to
        # ground the reply instead of answering from the training cutoff.
        if config.ALLOW_WEB and config.WEB_REPLY and web_access.DDGS_AVAILABLE:
            head += ("\n\n[CAPABILITIES] A live web-search tool is run FOR you by the "
                     "system before you reply; when it runs, the results appear below "
                     "under [WEB]. You cannot start a search yourself, run one in the "
                     "background, or 'stand by' for one — a search either already ran "
                     "this turn (a [WEB] block is present) or it did not. Rules: never "
                     "say you lack internet access; never claim you are searching or have "
                     "searched unless a [WEB] block is present; if a [WEB] block is "
                     "present, base your answer on it and cite the source URLs; if the "
                     "question needs current facts and there is NO [WEB] block, say you "
                     "don't have live results for it rather than inventing an answer.")
            lower_bar = bool(self.rules and self.rules.web_verify)
            query = ""
            if web_access.should_search(user_input, lower_bar=lower_bar):
                query = user_input
            elif web_access.is_search_followup(user_input):
                # "did you find something?" — the model may have promised a search last
                # turn; run it now for the previous question and present real results.
                query = web_access.last_searchable(self.history)
            if query:
                web_block = self._web_block(query)
                if web_block:
                    head += "\n\n" + web_block
        return head

    def _web_block(self, query: str) -> str:
        """Distil ``query`` into clean search terms, run a live web search, and render a
        ``[WEB]`` block (or "" if unavailable/empty). Best-effort — never raises into the
        reply path."""
        search_q = distill_search_query(query, client.generate)
        try:
            results = web_access.web_search(search_q,
                                            max_results=config.WEB_REPLY_RESULTS)
        except Exception:  # noqa: BLE001
            return ""
        if (not results.strip()
                or results.startswith(("Search failed:", "Search unavailable:"))
                or results == "Search returned no results."):
            return ""
        return (f"[WEB]\nLive web results fetched just now (search: \"{search_q}\"). "
                "Ground your answer in them, cite sources by URL, and say plainly if they "
                "do not cover the question:\n" + results[:config.WEB_REPLY_CHARS])

    def reply(self, user_input: str, learn: bool = True) -> str:
        """Run one turn: record the user line, generate, record + return the reply.
        Does not speak — use :meth:`reply_and_speak`. With ``learn=False`` the
        post-reply critique/meta-dreaming pass is skipped (used by tests)."""
        user_input = (user_input or "").strip()
        if not user_input:
            return ""
        self.history.append({"role": "user", "content": user_input})
        messages = self._fit_history()
        reply = client.generate_with_history(self.build_system(user_input), messages)
        if not reply:
            reply = (f"[backend error] {client.LAST_GENERATION_ERROR}"
                     if client.LAST_GENERATION_ERROR else "[no reply]")
        else:
            reply = self._dispatch_background(user_input, reply)
            self.history.append({"role": "assistant", "content": reply})
            # Persist the exchange to external memory so it survives this session.
            self.memory.remember_exchange(user_input, reply)
            # Critique + meta-dreaming + the dream cadence run OFF the reply path
            # (background thread) so they never add latency to the turn.
            if learn and self._wants_learning():
                threading.Thread(target=self.run_learning,
                                 args=(user_input, reply), daemon=True).start()
        return reply

    def _wants_learning(self) -> bool:
        return bool(self.critic or self.meta_dreamer or config.DREAM_EVERY > 0)

    def _dispatch_background(self, user_input: str, reply: str) -> str:
        """Queue any deferred work this turn implies — model-emitted ``[[THINK:]]``
        markers and plain-language 'do it in the background' requests — and return the
        reply with the markers stripped so they are never shown, spoken, or stored."""
        # Explicit "remember this" — pin a durable fact off the reply path. Kept
        # independent of the background subsystem so it works even when that is off.
        if self.memory.enabled and is_remember_request(user_input):
            threading.Thread(target=self._remember_from_request,
                             args=(user_input, list(self.history)), daemon=True).start()
        # "From now on, …" — adopt a durable standing rule (independent of background).
        if self.rules is not None and is_rule_request(user_input):
            threading.Thread(target=self._adopt_rule,
                             args=(user_input,), daemon=True).start()
        if self.background is None:
            return reply
        clean, tasks = extract_think_markers(reply)
        for task in tasks:
            self.background.queue(task)
        # A scheduling OR routine-management request is routed to the scheduler (one
        # bounded LLM pass, off the reply path): it creates a routine or changes an
        # existing one. Otherwise, a plain "do this in the background" queues one job.
        if self.routines is not None and (is_schedule_request(user_input)
                                          or is_routine_command(user_input)):
            threading.Thread(target=self.routines.handle_request,
                             args=(user_input,), daemon=True).start()
        elif self.claude_bridge is not None and is_code_request(user_input):
            # Delegate coding to Claude off the reply path; the dialogue + token
            # governance surface as bridge events (drained before the next prompt).
            threading.Thread(
                target=self.claude_bridge.request_code,
                args=(user_input, self.memory.recall_block(user_input)),
                daemon=True).start()
        elif should_auto_queue(user_input):
            self.background.queue(user_input,
                                  context=self.memory.recall_block(user_input))
        return clean

    def _remember_from_request(self, user_input: str, history: list[dict]) -> None:
        """Store what the user asked JARVS to remember (an inline clause, or the
        previous reply). Runs on a daemon thread; best-effort."""
        try:
            remember_from_request(
                user_input, history, client.generate,
                lambda t: self.memory.remember_fact(t, source="user"))
        except Exception:  # noqa: BLE001
            pass

    def _adopt_rule(self, user_input: str) -> None:
        """Distil a 'from now on …' request into a crisp standing directive, store it,
        and flip the web-verify flag when the rule is about checking the internet. Runs
        on a daemon thread; best-effort."""
        if self.rules is None:
            return
        try:
            directive = client.generate(
                "Rewrite the user's instruction as ONE concise standing rule for an "
                "assistant to follow, imperative voice, at most 30 words. Output only "
                "the rule.", user_input, False).strip()
            directive = " ".join(directive.split())[:400] or user_input
            self.rules.add(directive)
            if implies_web_verify(user_input) or implies_web_verify(directive):
                self.rules.set_web_verify(True)
        except Exception:  # noqa: BLE001
            pass

    def run_learning(self, user_input: str, reply: str) -> dict:
        """Grade the reply and, when enough new critiques have accumulated, run a
        meta-dreaming pass that may sharpen the critic. Synchronous and best-effort —
        reply() dispatches it on a background thread; tests call it directly."""
        result: dict = {"critique": {}, "distilled": 0, "dreamed": None}
        if self.critic is not None:
            try:
                result["critique"] = self.critic.critique(user_input, reply)
            except Exception:  # noqa: BLE001
                pass
        if self.meta_dreamer is not None:
            try:
                if self.meta_dreamer.should_distill():
                    result["distilled"] = self.meta_dreamer.distill()
            except Exception:  # noqa: BLE001
                pass
        # Automatic associative-consolidation ("dream") cadence.
        self._turns_since_dream += 1
        if config.DREAM_EVERY > 0 and self._turns_since_dream >= config.DREAM_EVERY:
            self._turns_since_dream = 0
            try:
                result["dreamed"] = self.memory.dream()
            except Exception:  # noqa: BLE001
                pass
        return result

    def stream_reply(self, user_input: str, learn: bool = True, speak: bool = False):
        """Generator variant of :meth:`reply` for the web UI: yields visible content
        deltas as they arrive, then records the full reply, persists it, and dispatches
        the critique/meta-dreaming pass. On a backend error it yields nothing and the
        caller can read :attr:`client.LAST_GENERATION_ERROR`."""
        user_input = (user_input or "").strip()
        if not user_input:
            return
        self.history.append({"role": "user", "content": user_input})
        messages = self._fit_history()
        full = [{"role": "system", "content": self.build_system(user_input)}] + messages
        parts: list[str] = []
        client.LAST_GENERATION_ERROR = ""
        try:
            for delta in client.stream_chat(full):
                parts.append(delta)
                yield delta
        except Exception as e:  # noqa: BLE001
            client.LAST_GENERATION_ERROR = f"{type(e).__name__}: {e}"
            return
        reply = client.strip_reasoning("".join(parts)).strip()
        if not reply:
            client.LAST_GENERATION_ERROR = (client.LAST_GENERATION_ERROR
                                            or "empty reply")
            return
        reply = self._dispatch_background(user_input, reply)
        self.history.append({"role": "assistant", "content": reply})
        self.memory.remember_exchange(user_input, reply)
        if speak and self.speak:
            self.voice.speak(reply)
        if learn and self._wants_learning():
            threading.Thread(target=self.run_learning,
                             args=(user_input, reply), daemon=True).start()

    def reply_and_speak(self, user_input: str, reveal=None) -> str:
        """Reply, then voice it. With a ``reveal`` callback the transcript is revealed
        in time with the audio (see JarvsVoice.speak_synced); otherwise it just speaks."""
        reply = self.reply(user_input)
        if reply and not reply.startswith("[") and self.speak:
            if reveal is not None:
                self.voice.speak_synced(reply, reveal)
            else:
                self.voice.speak(reply)
        return reply

    def reset(self) -> None:
        self.history.clear()
        self.voice.stop()


# --- CLI REPL ----------------------------------------------------------------- #

_BANNER = "JARVS online. Type a message, or /help for commands."
_HELP = (
    "/voice [on|off]      toggle or set voice output\n"
    "/remember <fact>     store a durable semantic fact\n"
    "/memory              memory stats\n"
    "/dream               weave associative links across stored memories\n"
    "/critic              self-critique stats (last verdict + pending log lines)\n"
    "/metadream [run]     meta-dreaming focus areas; 'run' forces a distill pass\n"
    "/background <task>   queue a deferred task (runs off-conversation, may use web)\n"
    "/jobs                recent background jobs + their results\n"
    "/routines [rm|off|on|run <id>]   recurring scheduled tasks\n"
    "/claude <task>       delegate a coding task to Claude Code (token-governed)\n"
    "/system              live host telemetry (CPU/GPU/mem/time/location)\n"
    "/self [reload]       JARVS's self-architecture dossier (stats; reload re-reads it)\n"
    "/rules [add <text>|rm <id>|web on|off]   standing rules (durable directives)\n"
    "/status              backend + voice + memory status\n"
    "/reset               clear the conversation (memory persists)\n"
    "/help                this list\n"
    "/quit, /exit         leave"
)


def dispatch_command(session: JarvsSession, line: str) -> str:
    """Run a slash command and RETURN its text output. Shared by the REPL and the web
    UI so both expose exactly the same commands. ``quit``/``exit`` are REPL-only and
    handled by the caller (:func:`_handle_command`), never here."""
    cmd, _, arg = line[1:].partition(" ")
    cmd, arg = cmd.strip().lower(), arg.strip()   # keep arg case for /remember
    out: list[str] = []
    p = out.append
    if cmd == "help":
        p(_HELP)
    elif cmd == "reset":
        session.reset()
        p("  · conversation cleared (memory persists)")
    elif cmd == "status":
        ok, detail = client.health()
        p(f"  · {detail}")
        p(f"  · {session.voice.status()}")
        p(f"  · {session.memory.status()}")
        if session.meta_dreamer is not None:
            snap = session.meta_dreamer.snapshot()
            p(f"  · critic: on · meta-dreaming: {snap['promoted_areas']} promoted / "
              f"{snap['total_areas']} focus areas")
        else:
            p(f"  · critic: {'on' if session.critic else 'off'} · meta-dreaming: off")
        if session.background is not None:
            p(f"  · {session.background.status()}")
        if session.routines is not None:
            p(f"  · {session.routines.status()}")
        if session.claude_bridge is not None:
            p(f"  · {session.claude_bridge.status()}")
        if config.SYSTEM_SENSE:
            p(f"  · {system_sense.status()}")
        if session.self_model is not None:
            p(f"  · {session.self_model.stats()}")
        if session.rules is not None:
            p(f"  · {session.rules.status()}")
        p(f"  · model={config.MODEL}  slot_ctx={config.SLOT_CTX}  "
          f"turns={len(session.history)}")
    elif cmd == "voice":
        if arg.lower() in ("on", "off"):
            session.voice.set_enabled(arg.lower() == "on")
        else:
            session.voice.set_enabled(not session.voice.enabled)
        p(f"  · {session.voice.status()}")
    elif cmd == "memory":
        p(f"  · {session.memory.status()}")
    elif cmd == "remember":
        if not arg:
            p("  · usage: /remember <fact to store>")
        else:
            mid = session.memory.remember_fact(arg)
            p(f"  · stored fact {mid[:8]}" if mid
              else "  · memory unavailable — nothing stored")
    elif cmd == "dream":
        p(f"  · {session.memory.dream()}")
    elif cmd == "critic":
        if session.meta_dreamer is None:
            p("  · critic: on, meta-dreaming off" if session.critic
              else "  · critic: off")
        else:
            snap = session.meta_dreamer.snapshot()
            p(f"  · critic: {'on' if session.critic else 'off'}  "
              f"pending log lines: {snap['pending_lines']}/"
              f"{snap['distill_min_new']} until next meta pass")
    elif cmd == "metadream":
        if session.meta_dreamer is None:
            p("  · meta-dreaming is disabled (JARVS_METADREAM=0)")
        else:
            if arg.lower() == "run":
                added = session.meta_dreamer.distill()
                p(f"  · distilled {added} focus area(s)")
            snap = session.meta_dreamer.snapshot()
            p(f"  · focus areas: {snap['total_areas']} total, "
              f"{snap['promoted_areas']} promoted (>= {snap['promote_min_pairs']} "
              f"sessions); {snap['pending_lines']} critique lines pending")
            for a in snap["areas"][:5]:
                star = "★" if a["pairs"] >= snap["promote_min_pairs"] else " "
                p(f"    {star} {a['area']} (sessions={a['pairs']}, hits={a['hits']})"
                  + (f": {a['specific']}" if a["specific"] else ""))
    elif cmd == "routines":
        if session.routines is None:
            p("  · routines are disabled (needs background work enabled)")
        else:
            sub, _, rid = arg.partition(" ")
            sub, rid = sub.strip().lower(), rid.strip()
            if sub in ("rm", "off", "on", "run") and rid:
                if sub == "rm":
                    ok = session.routines.remove(rid)
                elif sub == "run":
                    ok = session.routines.run_now(rid)
                else:
                    ok = session.routines.set_enabled(rid, sub == "on")
                p(f"  · {'done' if ok else 'no routine '+rid}")
            p(f"  · {session.routines.status()}")
            now = time.time()
            for r in session.routines.list():
                when = max(0, int(float(r["next_run"]) - now))
                state = "on " if r["enabled"] else "off"
                p(f"    [{state}] {r['id']} · {r['description']} "
                  f"({schedule_label(r)}, "
                  f"next in {when // 3600}h{(when % 3600) // 60}m, runs={r['runs']})")
    elif cmd == "system":
        p(system_sense.summary_block(include_geo=True))
    elif cmd == "rules":
        if session.rules is None:
            p("  · standing rules are disabled (JARVS_RULES=0)")
        else:
            sub, _, rest = arg.partition(" ")
            sub, rest = sub.strip().lower(), rest.strip()
            if sub == "add" and rest:
                r = session.rules.add(rest)
                if r and implies_web_verify(rest):
                    session.rules.set_web_verify(True)
                p(f"  · added rule {r['id']}" if r else "  · not added (too short/dupe)")
            elif sub == "rm" and rest:
                p(f"  · {'removed' if session.rules.remove(rest) else 'no rule '+rest}")
            elif sub == "web" and rest.lower() in ("on", "off"):
                session.rules.set_web_verify(rest.lower() == "on")
                p(f"  · web-verify {rest.lower()}")
            p(f"  · {session.rules.status()}")
            for r in session.rules.list():
                p(f"    [{r['id']}] {r['text']}")
    elif cmd == "self":
        if session.self_model is None:
            p("  · self-model is disabled (JARVS_SELF_MODEL=0 or memory off)")
        else:
            if arg.lower() == "reload":
                session.self_model.load()
                p("  · reloaded dossier")
            p(f"  · {session.self_model.stats()}")
    elif cmd == "background":
        if session.background is None:
            p("  · background work is disabled (JARVS_BACKGROUND=0)")
        elif not arg:
            p("  · usage: /background <task to work on>")
        else:
            job = session.background.queue(arg)
            p(f"  · queued job {job['id']} — I'll work on it and report back"
              if job else "  · not queued (queue full, duplicate, or too short)")
    elif cmd == "jobs":
        if session.background is None:
            p("  · background work is disabled")
        else:
            p(f"  · {session.background.status()}")
            for j in session.background.recent(8):
                p(f"    [{j['status']}] {j['id']} · {j['task'][:60]}")
                if j["status"] == "completed" and j.get("result"):
                    p(f"        → {j['result'][:200]}"
                      + ("…" if len(j['result']) > 200 else ""))
                elif j["status"] == "failed" and j.get("error"):
                    p(f"        ✗ {j['error'][:160]}")
    elif cmd == "claude":
        if session.claude_bridge is None:
            p("  · the Claude bridge is disabled (JARVS_CLAUDE_BRIDGE=0)")
        elif not arg:
            p("  · usage: /claude <coding task to delegate to Claude Code>")
            p(f"  · {session.claude_bridge.status()}")
        else:
            threading.Thread(
                target=session.claude_bridge.request_code,
                args=(arg, session.memory.recall_block(arg)),
                daemon=True).start()
            p("  · delegating to Claude — I'll report the verdict and result back")
    else:
        p(f"  · unknown command: /{cmd} (try /help)")
    return "\n".join(out)


def _handle_command(session: JarvsSession, line: str) -> bool:
    """REPL wrapper around :func:`dispatch_command`: handle quit, print the output."""
    cmd = line[1:].partition(" ")[0].strip().lower()
    if cmd in ("quit", "exit", "q"):
        raise SystemExit(0)
    if cmd == "dream":
        print("  · weaving associative links…")
    elif cmd == "metadream" and line[1:].partition(" ")[2].strip().lower() == "run":
        print("  · running a distill pass…")
    print(dispatch_command(session, line))
    return True


def _flush_background(session: JarvsSession) -> None:
    """Print any background-job / routine events that arrived since the last prompt."""
    if session.routines is not None:
        for ev in session.routines.drain_events():
            if ev["kind"] == "created":
                print(f"\n  ⟡ routine scheduled: {ev['description']}")
            elif ev["kind"] == "rescheduled":
                print(f"\n  ⟡ routine updated: {ev['description']}")
            elif ev["kind"] == "fired":
                print(f"\n  ⟡ routine fired: {ev['description']} (running now)")
    if session.claude_bridge is not None:
        for ev in session.claude_bridge.drain_events():
            task = (ev.get("task") or "")[:60]
            if ev["kind"] == "started":
                print(f"\n  ⟡ claude: working on — {task}")
            elif ev["kind"] == "done":
                print(f"\n  ⟡ claude: done — {task}")
                if ev.get("detail"):
                    print(f"    {ev['detail']}")
            elif ev["kind"] == "rejected_local":
                print(f"\n  ⟡ claude: not sent (JARVS held it back) — {task}")
                if ev.get("detail"):
                    print(f"    {ev['detail'][:200]}")
            elif ev["kind"] in ("needs_clarification", "reject", "no_verdict", "held"):
                print(f"\n  ⟡ claude: paused ({ev['kind']}) — {task}")
                if ev.get("detail"):
                    print(f"    {ev['detail'][:200]}")
            elif ev["kind"] == "over_budget":
                print(f"\n  ⟡ claude: refused (token budget) — {task}")
                if ev.get("detail"):
                    print(f"    {ev['detail']}")
            elif ev["kind"] == "error":
                print(f"\n  ⟡ claude: error — {task}: {ev.get('detail', '')[:160]}")
    if session.background is None:
        return
    for ev in session.background.drain_events():
        if ev["kind"] == "queued":
            print(f"\n  ⟡ background job {ev['id']} queued: {ev['task'][:70]}")
        elif ev["kind"] == "completed":
            job = next((j for j in session.background.recent(40)
                        if j["id"] == ev["id"]), None)
            res = (job or {}).get("result", "")
            print(f"\n  ⟡ background job {ev['id']} complete — {ev['task'][:60]}")
            if res:
                print(f"    {res[:400]}" + ("…" if len(res) > 400 else ""))
        elif ev["kind"] == "failed":
            print(f"\n  ⟡ background job {ev['id']} failed: {ev['task'][:60]}")


def main() -> None:
    session = JarvsSession()
    print(_BANNER)
    ok, detail = client.health()
    print(f"  · {detail}")
    print(f"  · {session.voice.status()}")
    print(f"  · {session.memory.status()}")
    if session.background is not None:
        print(f"  · {session.background.status()}")
    if session.routines is not None:
        print(f"  · {session.routines.status()}")
    if session.claude_bridge is not None:
        print(f"  · {session.claude_bridge.status()}")
    if session.self_model is not None:
        print(f"  · {session.self_model.stats()}")
    if session.rules is not None:
        print(f"  · {session.rules.status()}")
    while True:
        _flush_background(session)
        try:
            line = input("\nyou › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            try:
                _handle_command(session, line)
            except SystemExit:
                break
            continue
        reply = session.reply_and_speak(line)
        print(f"\nJARVS › {reply}")


if __name__ == "__main__":
    main()
