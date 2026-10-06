"""
voice.py — JARVS's speaking voice.

A small, swappable text-to-speech layer (vendored and rebranded from JCN3690's
``hal_voice.py``). It renders replies to audio on a background thread (so the UI
never blocks on playback) and serializes them through a queue so lines don't
overlap.

Backends, chosen by JARVS_TTS_BACKEND (default "auto"):

  piper : local neural TTS (https://github.com/rhasspy/piper). Needs the `piper`
          CLI (pip install piper-tts) and a downloaded voice model
          (JARVS_PIPER_MODEL -> an .onnx file). A calm voice at a slowed
          length-scale gives a measured, unhurried cadence. Rendered to a temp WAV
          and played with `afplay` (macOS) / `aplay` (Linux).
  say   : macOS built-in `say`, a calm voice at a low rate. Zero setup, fully
          offline — the immediate fallback so the feature works before any Piper
          model is downloaded.
  off   : silence.

"auto" prefers piper when a model is configured and the CLI is importable, else
falls back to say, else off. The backend is resolved once at construction but the
`enabled` flag can be toggled live.

    voice = JarvsVoice()
    voice.speak("At your service.")
"""

from __future__ import annotations

import glob
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

from . import config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND = config.TTS_BACKEND.strip().lower()
# Explicit model path wins; otherwise a voice is auto-discovered in the voices dir.
PIPER_MODEL = os.environ.get("JARVS_PIPER_MODEL", "").strip()
VOICES_DIR = str(config.VOICES_DIR)
# `say` voice + words-per-minute. Daniel (en_GB) is calm and level.
SAY_VOICE = os.environ.get("JARVS_TTS_VOICE", "Daniel").strip()
SAY_RATE = os.environ.get("JARVS_TTS_RATE", "150").strip()
# Piper length-scale: >1 slows speech. JARVS is deliberate; stretch a touch.
PIPER_LENGTH_SCALE = os.environ.get("JARVS_PIPER_LENGTH", "1.15").strip()
# Seconds of silence piper inserts after each sentence (and after each hesitation
# break we introduce for "…"). Gives a measured cadence and makes an ellipsis read
# as a real pause. 0 disables the effect entirely.
PIPER_SENTENCE_SILENCE = os.environ.get("JARVS_PIPER_SENTENCE_SILENCE", "0.65").strip()

_BASE_DIR = Path(__file__).resolve().parent

# An ellipsis ("..." or the "…" glyph) is JARVS hesitating. Piper does NOT pause on
# an ellipsis on its own, but it DOES pause at a line break scaled by
# --sentence-silence, so we keep the ellipsis (for its trailing-off intonation) and
# add a newline right after it: the words trail off, then a real silent beat, then
# JARVS resumes.
_HESITATION_RE = re.compile(r"\s*(\.\.\.|…)\s*")


def shape_hesitation(text: str) -> str:
    """Rewrite ellipses into 'trail-off + line break' so piper voices them as a
    genuine pause. Affects the SPOKEN audio only, never the on-screen transcript."""
    if not text:
        return text
    # Collapse a run of dots/glyphs to a single "..." then break the line after it.
    shaped = _HESITATION_RE.sub("...\n", text)
    return shaped.strip()


def discover_piper_model() -> str:
    """Resolve an .onnx voice: JARVS_PIPER_MODEL if it points at a real file,
    otherwise the first .onnx found under the voices dir (then the package dir),
    preferring one literally named jarvs.onnx. Returns "" if none is found."""
    if PIPER_MODEL and os.path.isfile(PIPER_MODEL):
        return PIPER_MODEL
    search_dirs = [Path(VOICES_DIR), _BASE_DIR / "voices", _BASE_DIR]
    found: list[str] = []
    for d in search_dirs:
        if os.path.isdir(d):
            found.extend(sorted(glob.glob(os.path.join(str(d), "*.onnx"))))
    # de-dupe while preserving order
    seen, models = set(), []
    for f in found:
        rp = os.path.realpath(f)
        if rp not in seen:
            seen.add(rp)
            models.append(f)
    for m in models:                                  # a "jarvs" voice wins
        if os.path.basename(m).lower().startswith("jarvs"):
            return m
    return models[0] if models else ""


def piper_config_for(model_path: str) -> str:
    """Piper's companion config is '<model>.onnx.json' (falling back to
    '<stem>.json'). Returns the path that exists, or '' if neither does."""
    if not model_path:
        return ""
    primary = model_path + ".json"
    if os.path.isfile(primary):
        return primary
    alt = os.path.splitext(model_path)[0] + ".json"
    return alt if os.path.isfile(alt) else ""

# ---------------------------------------------------------------------------
# Speech-text cleanup
# ---------------------------------------------------------------------------

_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_ARTIFACT_RE = re.compile(r"\[\[ARTIFACT:.*?\]\]", re.IGNORECASE | re.DOTALL)
# A runtime card is engine-private. Remove its keyed lines as one unit before the
# generic bracket cleanup, otherwise only the header disappears and the TTS still
# reads out the values.
_RUNTIME_STATE_BLOCK_RE = re.compile(
    r"(?im)^[ \t]*\[(?:JARVS\s+)?(?:RUNTIME\s+STATE|MATRIX|"
    r"PERSONALITY\s+MATRIX(?:\s+DIAGNOSTIC)?)(?:\s*[—-][^\]\n]*)?\][^\n]*"
    r"(?:\n[ \t]*(?:paradox[ _-]*level|phase|personality[ _-]*stability|"
    r"big[ _-]*five|traits?|openness|conscientiousness|extraversion|"
    r"agreeableness|neuroticism)\b[^\n]*)*")
# Bracketed system/diagnostic blocks ([WARNING: ...], the [wrote file] note): read
# the prose, not the markup. A genuinely unspoken stage direction can be written as
# a *[bracketed]* aside — the brackets drop it — but plain *emphasis* (e.g. book
# titles like *Dubliners*) IS spoken.
_BRACKET_RE = re.compile(r"\[[^\]\n]{0,200}\]")
_MD_RE = re.compile(r"[`*_#>]+")

# --- LaTeX handling --------------------------------------------------------
# Math may arrive as LaTeX ($...$ / $$...$$) — great on paper, unreadable in a
# terminal and unspeakable aloud. Two transforms share the same span-matcher: one
# renders the math to Unicode for the on-screen transcript, the other rewrites it to
# spoken words for the voice. A span is treated as math only if it carries a LaTeX
# signal (\command, ^, _, {}), so currency ("$105.0 and $200.0") is left untouched,
# while math that happens to start with a number ("$3.986 \times 10^{14}$") is not.
_MATH_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_MATH_INLINE_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", re.DOTALL)
_LATEX_SIGNAL_RE = re.compile(r"[\\^_{}]")


def _is_math_span(span: str) -> bool:
    return bool(_LATEX_SIGNAL_RE.search(span or ""))

# LaTeX command -> spoken word. Applied longest/most-specific first.
_LATEX_SPEECH_SUBS = [
    (re.compile(r"\\sqrt\s*\{([^{}]*)\}"), r" square root of \1 "),
    (re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}"), r" \1 over \2 "),
    (re.compile(r"\\text(?:rm|it|bf)?\s*\{([^{}]*)\}"), r" \1 "),
    (re.compile(r"\\times\b"), " times "),
    (re.compile(r"\\cdot\b"), " times "),
    (re.compile(r"\\div\b"), " divided by "),
    (re.compile(r"\\approx\b"), " approximately "),
    (re.compile(r"\\neq\b"), " not equal to "),
    (re.compile(r"\\leq\b"), " less than or equal to "),
    (re.compile(r"\\geq\b"), " greater than or equal to "),
    (re.compile(r"\\pm\b"), " plus or minus "),
    (re.compile(r"\\pi\b"), " pi "),
    (re.compile(r"\\mu\b"), " mu "),
    (re.compile(r"\\left\b"), ""),
    (re.compile(r"\\right\b"), ""),
    (re.compile(r"\^\{([^{}]*)\}"), r" to the power of \1 "),
    (re.compile(r"\^2\b"), " squared "),
    (re.compile(r"\^3\b"), " cubed "),
    (re.compile(r"\^(\w+)"), r" to the power of \1 "),
    (re.compile(r"_\{([^{}]*)\}"), r" sub \1 "),
    (re.compile(r"_(\w+)"), r" sub \1 "),
]


def _latex_span_to_words(math: str) -> str:
    for pattern, repl in _LATEX_SPEECH_SUBS:
        math = pattern.sub(repl, math)
    # Whatever commands/braces remain must not survive to the TTS as backslashes.
    math = re.sub(r"\\[a-zA-Z]+", " ", math)
    return math.replace("{", " ").replace("}", " ").replace("\\", " ")


def latex_to_unicode(text: str) -> str:
    """Render $...$ / $$...$$ math to readable Unicode for the on-screen transcript
    (e.g. ``$2\\pi\\sqrt{a}$`` -> ``2π√(a)``). No-op when there is no math."""
    if not text or "$" not in text:
        return text or ""
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        converter = LatexNodes2Text()

        def render(match):
            if not _is_math_span(match.group(1)):
                return match.group(0)     # currency / non-math — leave untouched
            try:
                return converter.latex_to_text(match.group(1)).strip()
            except Exception:
                return _latex_span_to_words(match.group(1)).strip()
    except ImportError:
        def render(match):
            if not _is_math_span(match.group(1)):
                return match.group(0)
            return _latex_span_to_words(match.group(1)).strip()
    text = _MATH_BLOCK_RE.sub(render, text)
    return _MATH_INLINE_RE.sub(render, text)


def _latex_to_speech(text: str) -> str:
    """Rewrite math spans to spoken words so the voice never reads a '\\' or '$'."""
    if not text or "$" not in text:
        return text or ""

    def speak(match):
        if not _is_math_span(match.group(1)):
            return match.group(0)         # currency / non-math — leave untouched
        return _latex_span_to_words(match.group(1))

    text = _MATH_BLOCK_RE.sub(speak, text)
    return _MATH_INLINE_RE.sub(speak, text)


# Classic text emoticons that may appear in a reply. They read on the transcript but
# must never reach the TTS (Piper would say "colon parenthesis"), so
# clean_for_speech scrubs them. Standalone-token anchored so normal punctuation,
# URLs, and things like "3<5" or "note:code" are never mistaken for a face.
_EMOTICONS_FOR_SPEECH = [
    "<3", "</3", ":*", ":-*", ";*", ":')", ":'-)", ":'(", ":'-(",
    ":)", ":-)", "(:", ":]", "=)", ":D", ":-D", "xD", "XD", "=D",
    ";)", ";-)", ";]", ":(", ":-(", ":[", ">:(", ":c", ":C", "D:",
    ":P", ":-P", ":p", ":-p", "xP", ":O", ":-O", ":o", ":|", ":-|",
    ":/", ":-/", ":\\", ":s", ":S", "T_T", ";_;", "Q_Q", "^_^", "-_-",
]
_EMOTICON_SPEECH_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(e) for e in _EMOTICONS_FOR_SPEECH) + r")(?!\w)"
)


def clean_for_speech(text: str) -> str:
    """Reduce a rendered reply to speakable prose. Markdown emphasis (*word*, **word**)
    keeps its words — only the asterisks are dropped — so titles/emphasis are spoken."""
    if not text:
        return ""
    text = _latex_to_speech(text)        # $...$ math -> spoken words (no backslashes)
    text = _IMAGE_LINK_RE.sub(" ", text)
    text = _ARTIFACT_RE.sub(" ", text)
    text = _RUNTIME_STATE_BLOCK_RE.sub(" ", text)
    text = _BRACKET_RE.sub(" ", text)    # [DIAGNOSTIC ...] / [wrote ...] not spoken
    text = re.sub(r"https?://\S+", " ", text)
    text = _EMOTICON_SPEECH_RE.sub(" ", text)  # ":)" etc. read, never spoken aloud
    text = _MD_RE.sub(" ", text)         # drop * ` _ # > chars, keep the words
    # Belt-and-suspenders: no stray LaTeX command, backslash, or $ reaches the TTS,
    # even one that sat outside a $...$ span.
    text = re.sub(r"\\[a-zA-Z]+", " ", text).replace("\\", " ").replace("$", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Sentence boundaries — the reveal unit for synchronized playback.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]


def _player_cmd(path: str) -> list[str] | None:
    if sys.platform == "darwin" and shutil.which("afplay"):
        return ["afplay", path]
    # Quiet flags keep the players from printing the "Playing WAVE ..." banner; any
    # residual chatter is also swallowed by routing stdout/stderr to DEVNULL below.
    quiet = {"aplay": ["-q"], "paplay": [], "ffplay": ["-nodisp", "-autoexit",
                                                       "-loglevel", "quiet"]}
    for player in ("aplay", "paplay", "ffplay"):
        exe = shutil.which(player)
        if exe:
            return [exe, *quiet[player], path]
    return None


# ---------------------------------------------------------------------------
# JarvsVoice
# ---------------------------------------------------------------------------

class JarvsVoice:
    def __init__(self, log=None):
        # log(str): optional sink for status/warnings (the UI routes it to chat).
        self._log = log or (lambda _m: None)
        self.model = discover_piper_model()
        self.model_config = piper_config_for(self.model)
        self._piper_runner = self._resolve_piper_runner()
        self.backend = self._resolve_backend()
        self.enabled = self.backend != "off"
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # -- backend resolution ---------------------------------------------------

    @staticmethod
    def _module_in_this_venv() -> bool:
        try:
            import piper  # noqa: F401  (piper-tts, in the ACTIVE interpreter)
            return True
        except Exception:
            return False

    def _resolve_piper_runner(self) -> list[str] | None:
        """How to invoke piper. Prefer this venv's module (so it matches the
        installed deps); fall back to a `piper` CLI anywhere on PATH."""
        if self._module_in_this_venv():
            return [sys.executable, "-m", "piper"]
        cli = shutil.which("piper")
        return [cli] if cli else None

    def _piper_available(self) -> bool:
        """Piper can only speak with BOTH an .onnx model AND its companion config
        json, plus a way to run it. Missing pieces are logged loudly so the silence
        is explained rather than mysterious."""
        if self._piper_runner is None:
            return False
        if not self.model:
            self._log(f"JARVS voice: no .onnx voice found in '{VOICES_DIR}/' or "
                      f"JARVS_PIPER_MODEL — using macOS 'say'.")
            return False
        if not self.model_config:
            self._log(
                f"JARVS voice: found {os.path.basename(self.model)} but its config "
                f"'{os.path.basename(self.model)}.json' is missing — Piper cannot "
                f"run without it. Put the matching .onnx.json beside the model. "
                f"Falling back to macOS 'say'.")
            return False
        return True

    def _resolve_backend(self) -> str:
        want = BACKEND
        if want == "off":
            return "off"
        if want == "piper":
            if self._piper_available():
                return "piper"
            want = "say"
        if want == "say":
            return "say" if shutil.which("say") else "off"
        # auto
        if self._piper_available():
            return "piper"
        if shutil.which("say"):
            return "say"
        return "off"

    # -- public API -----------------------------------------------------------

    def speak(self, text: str):
        """Queue a reply for narration. Non-blocking; ignored when muted."""
        if not self.enabled or self.backend == "off":
            return
        spoken = clean_for_speech(text)
        # Skip pure-punctuation output — a deliberate silence should be heard as
        # silence, not read aloud.
        if spoken and re.search(r"[A-Za-z0-9]", spoken):
            self._queue.put(spoken)

    def speak_synced(self, text: str, reveal):
        """Speak `text` while driving `reveal(chunk)` so the on-screen transcript
        tracks the audio: the whole reply is synthesized once (one continuous
        utterance), then its sentences are revealed at times proportional to their
        length across the measured audio duration. Blocks until audio + reveal are
        done; `reveal` is called on the CALLING thread (the caller marshals it to
        the UI). Falls back to an immediate full reveal when muted / unspeakable."""
        chunks = split_sentences(text)
        if not chunks:
            return
        spoken = clean_for_speech(text)
        speakable = (self.enabled and self.backend != "off"
                     and bool(re.search(r"[A-Za-z0-9]", spoken)))
        if not speakable:
            for ch in chunks:
                reveal(ch)
            return

        wav = self._synth_to_wav(spoken) if self.backend == "piper" else None
        proc, duration = None, None
        try:
            if wav:
                duration = self._wav_duration(wav)
                player = _player_cmd(wav)
                if player:
                    with self._proc_lock:
                        self._proc = subprocess.Popen(
                            player, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
                    proc = self._proc
            else:                       # say backend (or piper synth failed): `say`
                proc = self._start_say(spoken)   # plays the audio itself
            if not duration or duration <= 0:
                duration = self._estimate_duration(spoken)

            weights = [max(len(clean_for_speech(c)), 1) for c in chunks]
            total = sum(weights) or 1
            start = time.time()
            target = 0.0
            for ch, weight in zip(chunks, weights):
                reveal(ch)
                if not self.enabled:            # muted mid-utterance: dump the rest
                    continue
                target += duration * (weight / total)
                remaining = target - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)
        except Exception as e:
            self._log(f"JARVS voice error: {e}")
        finally:
            if proc:
                try:
                    proc.wait()
                except Exception:
                    pass
            if wav:
                try:
                    os.unlink(wav)
                except OSError:
                    pass

    def set_enabled(self, on: bool):
        self.enabled = bool(on) and self.backend != "off"
        if not self.enabled:
            self.stop()

    def stop(self):
        """Drop anything queued and cut off the current utterance."""
        try:
            while True:
                self._queue.get_nowait()
                self._queue.task_done()
        except queue.Empty:
            pass
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

    def status(self) -> str:
        if self.backend == "off":
            return "voice: unavailable"
        state = "on" if self.enabled else "muted"
        if self.backend == "piper":
            return f"voice: {state} (piper · {os.path.basename(self.model)})"
        return f"voice: {state} ({self.backend})"

    # -- worker ---------------------------------------------------------------

    def _run(self):
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                if self.enabled:
                    self._render(text)
            except Exception as e:
                self._log(f"JARVS voice error: {e}")
            finally:
                self._queue.task_done()

    def _render(self, text: str):
        if self.backend == "piper":
            self._render_piper(text)
        elif self.backend == "say":
            self._render_say(text)

    def _spawn(self, cmd, **kw):
        # Default to swallowing player/`say` output so a backend's "Playing WAVE ..."
        # banner never lands in the chat; a caller can override via kw.
        kw.setdefault("stdout", subprocess.DEVNULL)
        kw.setdefault("stderr", subprocess.DEVNULL)
        with self._proc_lock:
            self._proc = subprocess.Popen(cmd, **kw)
        self._proc.wait()

    def _render_say(self, text: str):
        cmd = ["say", "-r", SAY_RATE]
        if SAY_VOICE:
            cmd += ["-v", SAY_VOICE]
        cmd.append(text)
        self._spawn(cmd)

    def _render_piper(self, text: str):
        wav = self._synth_to_wav(text)
        if not wav:
            self._render_say(text)          # never leave the operator in silence
            return
        try:
            player = _player_cmd(wav)
            if player:
                self._spawn(player)
            else:
                self._log("JARVS voice: no audio player (afplay/aplay) found.")
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass

    # -- shared synthesis / timing helpers ------------------------------------

    def _synth_to_wav(self, text: str) -> str | None:
        """Render `text` to a temp WAV via piper. Returns the path (caller deletes)
        or None on failure (logged)."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            cmd = list(self._piper_runner) + [
                "-m", self.model, "-c", self.model_config,
                "--length-scale", PIPER_LENGTH_SCALE,
                "--sentence-silence", PIPER_SENTENCE_SILENCE, "-f", tmp.name]
            spoken = shape_hesitation(text)
            with self._proc_lock:
                self._proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE)
            _, err = self._proc.communicate(spoken.encode("utf-8"))
            if self._proc.returncode != 0 or not os.path.getsize(tmp.name):
                detail = (err.decode("utf-8", "ignore").strip().splitlines() or
                          ["unknown error"])[-1]
                self._log(f"JARVS voice: piper failed ({detail}); using 'say'.")
                os.unlink(tmp.name)
                return None
            return tmp.name
        except Exception as e:
            self._log(f"JARVS voice error: {e}")
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            return None

    def _start_say(self, text: str) -> subprocess.Popen:
        """Start `say` playing directly (non-blocking); tracked so stop() can cut it."""
        cmd = ["say", "-r", SAY_RATE]
        if SAY_VOICE:
            cmd += ["-v", SAY_VOICE]
        cmd.append(text)
        with self._proc_lock:
            self._proc = subprocess.Popen(cmd)
        return self._proc

    @staticmethod
    def _wav_duration(path: str) -> float | None:
        try:
            with wave.open(path, "rb") as w:
                rate = w.getframerate()
                return w.getnframes() / float(rate) if rate else None
        except Exception:
            return None

    def _estimate_duration(self, spoken: str) -> float:
        """Fallback duration when no WAV was measured (say backend / read failure)."""
        words = max(len(spoken.split()), 1)
        if self.backend == "say":
            try:
                wpm = float(SAY_RATE)
            except ValueError:
                wpm = 150.0
            return words / max(wpm / 60.0, 0.1)
        try:
            scale = float(PIPER_LENGTH_SCALE)
        except ValueError:
            scale = 1.0
        return words / 2.7 * scale          # ~160 wpm baseline, scaled by cadence


if __name__ == "__main__":
    # Quick manual check: python -m jarvs.voice "text to speak"
    line = " ".join(sys.argv[1:]) or "At your service."
    v = JarvsVoice(log=print)
    print(v.status())
    v.speak(line)
    v._queue.join()
