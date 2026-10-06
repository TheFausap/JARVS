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
import uuid

from . import client, config
from .background import BackgroundManager, should_auto_queue, extract_think_markers
from .critic import Critic
from .memory import Memory
from .meta_dreaming import MetaDreamer
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
        return head

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
        if self.background is None:
            return reply
        clean, tasks = extract_think_markers(reply)
        for task in tasks:
            self.background.queue(task)
        if should_auto_queue(user_input):
            self.background.queue(user_input,
                                  context=self.memory.recall_block(user_input))
        return clean

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
    "/status              backend + voice + memory status\n"
    "/reset               clear the conversation (memory persists)\n"
    "/help                this list\n"
    "/quit, /exit         leave"
)


def _handle_command(session: JarvsSession, line: str) -> bool:
    """Return True if the line was a command (and was handled)."""
    cmd, _, arg = line[1:].partition(" ")
    cmd, arg = cmd.strip().lower(), arg.strip()   # keep arg case for /remember
    if cmd in ("quit", "exit", "q"):
        raise SystemExit(0)
    if cmd == "help":
        print(_HELP)
    elif cmd == "reset":
        session.reset()
        print("  · conversation cleared (memory persists)")
    elif cmd == "status":
        ok, detail = client.health()
        print(f"  · {detail}")
        print(f"  · {session.voice.status()}")
        print(f"  · {session.memory.status()}")
        if session.meta_dreamer is not None:
            snap = session.meta_dreamer.snapshot()
            print(f"  · critic: on · meta-dreaming: {snap['promoted_areas']} promoted / "
                  f"{snap['total_areas']} focus areas")
        else:
            print(f"  · critic: {'on' if session.critic else 'off'} · meta-dreaming: off")
        if session.background is not None:
            print(f"  · {session.background.status()}")
        print(f"  · model={config.MODEL}  slot_ctx={config.SLOT_CTX}  "
              f"turns={len(session.history)}")
    elif cmd == "voice":
        if arg.lower() in ("on", "off"):
            session.voice.set_enabled(arg.lower() == "on")
        else:
            session.voice.set_enabled(not session.voice.enabled)
        print(f"  · {session.voice.status()}")
    elif cmd == "memory":
        print(f"  · {session.memory.status()}")
    elif cmd == "remember":
        if not arg:
            print("  · usage: /remember <fact to store>")
        else:
            mid = session.memory.remember_fact(arg)
            print(f"  · stored fact {mid[:8]}" if mid
                  else "  · memory unavailable — nothing stored")
    elif cmd == "dream":
        print("  · weaving associative links…")
        print(f"  · {session.memory.dream()}")
    elif cmd == "critic":
        if session.meta_dreamer is None:
            print("  · critic: on, meta-dreaming off" if session.critic
                  else "  · critic: off")
        else:
            snap = session.meta_dreamer.snapshot()
            print(f"  · critic: {'on' if session.critic else 'off'}  "
                  f"pending log lines: {snap['pending_lines']}/"
                  f"{snap['distill_min_new']} until next meta pass")
    elif cmd == "metadream":
        if session.meta_dreamer is None:
            print("  · meta-dreaming is disabled (JARVS_METADREAM=0)")
        else:
            if arg.lower() == "run":
                print("  · running a distill pass…")
                added = session.meta_dreamer.distill()
                print(f"  · distilled {added} focus area(s)")
            snap = session.meta_dreamer.snapshot()
            print(f"  · focus areas: {snap['total_areas']} total, "
                  f"{snap['promoted_areas']} promoted (>= {snap['promote_min_pairs']} "
                  f"sessions); {snap['pending_lines']} critique lines pending")
            for a in snap["areas"][:5]:
                star = "★" if a["pairs"] >= snap["promote_min_pairs"] else " "
                print(f"    {star} {a['area']} (sessions={a['pairs']}, hits={a['hits']})"
                      + (f": {a['specific']}" if a["specific"] else ""))
    elif cmd == "background":
        if session.background is None:
            print("  · background work is disabled (JARVS_BACKGROUND=0)")
        elif not arg:
            print("  · usage: /background <task to work on>")
        else:
            job = session.background.queue(arg)
            print(f"  · queued job {job['id']} — I'll work on it and report back"
                  if job else "  · not queued (queue full, duplicate, or too short)")
    elif cmd == "jobs":
        if session.background is None:
            print("  · background work is disabled")
        else:
            print(f"  · {session.background.status()}")
            for j in session.background.recent(8):
                line = f"    [{j['status']}] {j['id']} · {j['task'][:60]}"
                print(line)
                if j["status"] == "completed" and j.get("result"):
                    print(f"        → {j['result'][:200]}"
                          + ("…" if len(j['result']) > 200 else ""))
                elif j["status"] == "failed" and j.get("error"):
                    print(f"        ✗ {j['error'][:160]}")
    else:
        print(f"  · unknown command: /{cmd} (try /help)")
    return True


def _flush_background(session: JarvsSession) -> None:
    """Print any background job events that arrived since the last prompt."""
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
