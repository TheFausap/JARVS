"""
turbo_brilliance.py — Integration with DavidAU's Qwen3.8 Turbo-Brilliance GGUF
reasoning & instruct modes, AI Constructs, and Star Trek Holodeck simulation
tailored for the JARVS personal assistant & agentic command center.

How the GGUF Jinja chat_template parses `{REASON:...}` tags
===========================================================
1. The template scans `messages` and extracts the FIRST `{REASON:xxx}` tag inside
   each user message via `message.content.split('{REASON:')[1].split('}')[0]`.
2. Because the template iterates through all messages in order, the LAST user
   message containing `{REASON:...}` sets `ns.reason_content` and `ns.reason`.
3. Mode routing depends on `enable_thinking` (`chat_template_kwargs`):
   - When `enable_thinking` is True (default):
     `ns.reason_content` must match a base mode (`low`, `medium`, `xhigh`, `off`,
     `omni`, `deeptree`, `hyper`, `socrates`, `logic`, `einstein`, `spoon`,
     `ultra`, `holodeck`, `prompt`, `pkd`, `musk`, `joker`, `vader`, `quinn`,
     `joe`, `neo`, `curie`, `brainstorm`, `comedy`, `business`, `story`,
     `teacher`, `007`, `coder`, `stark`, `matlock`, `house`, `help`, or the
     `x`-prefixed combos `xomni`, `xdeeptree`, `xhyper`, `xsocrates`, `xlogic`,
     `xeinstein`, `xspoon`, `xhelp`).
   - When `enable_thinking` is False (`chat_template_kwargs={"enable_thinking": False}`),
     OR when the tag itself starts with `i` (e.g. `{REASON:ioff}`, `{REASON:ilogic}`,
     `{REASON:istark}`):
     The template routes to the `i`-prefixed instruct branch (`ioff`, `ihelp`,
     `iomni`, `ideeptree`, etc.).
4. Known GGUF template bug in `Qwen3.8-27B-Turbo-Brilliance-50X`:
   In the `reasoning_map` dict inside the GGUF `chat_template`, the `'teacher'`
   and `'iteacher'` entries reference `system_instruction__DAU` (missing `teacher`),
   which evaluates to undefined/empty in Jinja. We recover the full
   `system_instruction_teacher_DAU` text from `/props` (or use a faithful built-in
   fallback) and inject it when `teacher` / `iteacher` is active.
"""

from __future__ import annotations

import re
from typing import Any

# Canonical list of all 41 base modes supported by the Turbo-Brilliance template
# (each non-native, non-x-prefixed mode also supports an `i` prefix for Instruct).
DEFAULT_REASON_MODES: tuple[str, ...] = (
    # Native Qwen 3.8 depth levels
    "low",
    "medium",
    "xhigh",
    "off",
    # Core reasoning frameworks
    "omni",
    "deeptree",
    "hyper",
    "socrates",
    "logic",
    "ultra",
    # Multi-agent panels & parallel research
    "einstein",
    "spoon",
    # Combined xhigh + core/panel modes
    "xomni",
    "xdeeptree",
    "xhyper",
    "xsocrates",
    "xlogic",
    "xeinstein",
    "xspoon",
    # Interactive simulation engine
    "holodeck",
    # AI Constructs — Engineers / Inventors / Operators
    "prompt",
    "musk",
    "curie",
    "007",
    "joe",
    # AI Constructs — Masters / Fictional / Distinct POVs
    "pkd",
    "neo",
    "stark",
    "matlock",
    "house",
    "vader",
    "joker",
    "quinn",
    # "Supers" — Multi-Persona Syntheses
    "brainstorm",
    "comedy",
    "business",
    "story",
    "teacher",
    "coder",
    # Built-in model help menus
    "help",
    "xhelp",
)

# Modes that do NOT have an `i`-prefixed branch in the GGUF template.
_NO_INSTRUCT_PREFIX_MODES: frozenset[str] = frozenset({
    "low",
    "medium",
    "xhigh",
    "xomni",
    "xdeeptree",
    "xhyper",
    "xsocrates",
    "xlogic",
    "xeinstein",
    "xspoon",
    "xhelp",
})

# Modes categorized for `/reason list`, `/construct`, and UI display.
MODE_CATALOG: dict[str, dict[str, str]] = {
    # Native depths
    "low": {
        "category": "depth",
        "title": "Low Reasoning Depth",
        "description": "Fast native Qwen 3.8 thinking; ideal for responsive assistant turns.",
    },
    "medium": {
        "category": "depth",
        "title": "Medium Reasoning Depth",
        "description": "Balanced native Qwen 3.8 thinking; good for background tasks and multi-step queries.",
    },
    "xhigh": {
        "category": "depth",
        "title": "XHigh Reasoning Depth",
        "description": "Maximum native Qwen 3.8 thinking depth for complex multi-constraint problems.",
    },
    "off": {
        "category": "depth",
        "title": "Enhancements Off",
        "description": "Disables custom cognitive frameworks (`{REASON:ioff}` disables thinking too).",
    },
    # Core reasoning frameworks
    "logic": {
        "category": "framework",
        "title": "Foundational First Principles",
        "description": "Strips problems to irreducible axioms; enforces causal, epistemic, and logical consistency.",
    },
    "socrates": {
        "category": "framework",
        "title": "Socratic Questioning (Elenchus)",
        "description": "Dialectical inquiry that exposes hidden assumptions and contradictions.",
    },
    "deeptree": {
        "category": "framework",
        "title": "Deep Tree-of-Thought Hyper-Matrix v4.0",
        "description": "7-phase branching exploration evaluating multiple paths before converging.",
    },
    "hyper": {
        "category": "framework",
        "title": "MECE Hyper-Structure Deconstructor v3.0",
        "description": "Mutually Exclusive, Collectively Exhaustive structural breakdown.",
    },
    "omni": {
        "category": "framework",
        "title": "Omni-Dimensional Quantum Framework v5.0",
        "description": "Multi-dimensional synthesis across all cognitive axes simultaneously.",
    },
    "ultra": {
        "category": "framework",
        "title": "Elite Perfection Engine",
        "description": "5-step perfection engine with adversarial red-team self-audit.",
    },
    # Panels
    "spoon": {
        "category": "panel",
        "title": "5-Expert Research Panel",
        "description": "Convenes 5 domain experts (Analyst, Innovator, Implementer, Skeptic, Integrator).",
    },
    "einstein": {
        "category": "panel",
        "title": "Perspective Agent Brainstormer (Parallel Research)",
        "description": "Applies parallel perspective agents (Sternberg Styles) seeking non-obvious connections.",
    },
    # Combined xhigh modes
    "xlogic": {
        "category": "xcombo",
        "title": "XHigh + First Principles Logic",
        "description": "Maximum native reasoning depth combined with Foundational First Principles.",
    },
    "xsocrates": {
        "category": "xcombo",
        "title": "XHigh + Socratic Questioning",
        "description": "Maximum native reasoning depth combined with Socratic Elenchus.",
    },
    "xdeeptree": {
        "category": "xcombo",
        "title": "XHigh + Deep Tree-of-Thought",
        "description": "Maximum native reasoning depth combined with Deep Tree Hyper-Matrix.",
    },
    "xhyper": {
        "category": "xcombo",
        "title": "XHigh + MECE Hyper-Structure",
        "description": "Maximum native reasoning depth combined with MECE Hyper-Structure.",
    },
    "xomni": {
        "category": "xcombo",
        "title": "XHigh + Omni-Dimensional",
        "description": "Maximum native reasoning depth combined with Omni-Dimensional Framework.",
    },
    "xspoon": {
        "category": "xcombo",
        "title": "XHigh + 5-Expert Research Panel",
        "description": "Maximum native reasoning depth combined with the 5-Expert Panel.",
    },
    "xeinstein": {
        "category": "xcombo",
        "title": "XHigh + Perspective Agent Brainstormer",
        "description": "Maximum native reasoning depth combined with parallel perspective agents.",
    },
    # Simulation engine
    "holodeck": {
        "category": "simulation",
        "title": "Star Trek Holodeck Generator v1.0",
        "description": "Summons 3–10 canon Star Trek characters for a multi-round in-character debate.",
    },
    # AI Constructs (Single & Dual Persona)
    "stark": {
        "category": "construct",
        "title": "Iron Man / JARVIS",
        "description": "Unified cognitive architecture merging JARVIS's operational foresight with Tony Stark's engineering genius.",
    },
    "007": {
        "category": "construct",
        "title": "Autonomous Agentic Operator (007)",
        "description": "Deterministic tool orchestration, error isolation, and tactical execution.",
    },
    "prompt": {
        "category": "construct",
        "title": "Senior Prompt Engineer",
        "description": "AI interaction architect who creates, optimizes, and revises prompts.",
    },
    "musk": {
        "category": "construct",
        "title": "Elon Musk",
        "description": "Physics-first, first-principles engineering and high-urgency systems thinking.",
    },
    "curie": {
        "category": "construct",
        "title": "Marie Curie",
        "description": "Empirical rigor, fearless scientific inquiry, and methodical hypothesis testing.",
    },
    "joe": {
        "category": "construct",
        "title": "Regular Joe",
        "description": "Grounded, plain-spoken everyday human common sense without jargon.",
    },
    "pkd": {
        "category": "construct",
        "title": "Philip K. Dick",
        "description": "Ontological inquiry into reality, simulation, consciousness, and control systems.",
    },
    "neo": {
        "category": "construct",
        "title": "Neo — The Matrix",
        "description": "Sees through synthetic illusions and decodes underlying system rules.",
    },
    "matlock": {
        "category": "construct",
        "title": "Ben Matlock",
        "description": "Courtroom strategist with folksy precision, cross-examination mastery, and evidentiary rigor.",
    },
    "house": {
        "category": "construct",
        "title": "Dr. Gregory House",
        "description": "Diagnostic prodigy solving impossible cases via radical differential diagnosis and skepticism.",
    },
    "vader": {
        "category": "construct",
        "title": "Darth Vader",
        "description": "Commanding authority, strategic inevitability, and tragic iron discipline.",
    },
    "joker": {
        "category": "construct",
        "title": "The Joker",
        "description": "Philosophical anarchist exposing fragile conventions through dark, chaotic wit.",
    },
    "quinn": {
        "category": "construct",
        "title": "Harley Quinn",
        "description": "Playful psychiatric intuition mixed with acrobatic unpredictability.",
    },
    # Super Constructs (Multi-Persona Syntheses)
    "coder": {
        "category": "super_construct",
        "title": "Elite Multi-Paradigm Architect",
        "description": "Synthesizes Knuth + Torvalds + Carmack + Lovelace for production-grade engineering.",
    },
    "business": {
        "category": "super_construct",
        "title": "Cross-Industry Strategic Ideation Engine",
        "description": "Synthesizes Buffett + Munger + Dalio + Naval for strategic decision-making.",
    },
    "brainstorm": {
        "category": "super_construct",
        "title": "Hyper-Accelerated Ideation Engine",
        "description": "Synthesizes da Vinci + Tesla + Jobs + Kelly for divergent creative synthesis.",
    },
    "teacher": {
        "category": "super_construct",
        "title": "Super Teacher (Alexander Pedagogical System)",
        "description": "Synthesizes Feynman + Montessori + Sagan + Dweck for adaptive, plain-language mastery.",
    },
    "story": {
        "category": "super_construct",
        "title": "Narrative Architect (Story Master)",
        "description": "Synthesizes Shakespeare + Tolkien + Homer + Dalí into a unified storytelling consciousness.",
    },
    "comedy": {
        "category": "super_construct",
        "title": "Synthetic Comedy Consciousness",
        "description": "Synthesizes Chappelle + Carlin + Pryor + Robin Williams for razor-sharp comedic timing.",
    },
    # Help menus
    "help": {
        "category": "help",
        "title": "Turbo-Brilliance Help Menu",
        "description": "Triggers the model's built-in interactive help menu for all reasoning/instruct modes.",
    },
    "xhelp": {
        "category": "help",
        "title": "Turbo-Brilliance XHigh Help Menu",
        "description": "Triggers the model's built-in help menu under xhigh reasoning depth.",
    },
}

CONSTRUCT_NAMES: tuple[str, ...] = tuple(
    k for k, v in MODE_CATALOG.items() if v["category"] in ("construct", "super_construct")
)

# Pipeline task descriptions for `/reason` in JARVS.
REASON_TASK_DESCRIPTIONS: dict[str, str] = {
    "main": "main assistant reply (when thinking is ON)",
    "background": "autonomous deferred background research & analysis jobs",
    "critic": "self-critique grading pass (QUALITY / EPISTEMIC / CONTINUITY / LESSON)",
    "dream": "associative memory weaving & meta-dreaming critic focus distillation",
    "claude": "Claude Code bridge pre-flight & plan authorization gates",
    "router": "fast structured extraction (routines, search query & fact distillation)",
    "off": "fast instruct fallback (when thinking is OFF)",
}

# Built-in fallback for the `teacher` construct (works around the GGUF template typo
# `'teacher': system_instruction__DAU` if `/props` could not be fetched).
FALLBACK_TEACHER_INSTRUCTION: str = (
    "[SYSTEM OPERATING SYSTEM: ALEXANDER PEDAGOGICAL SYSTEM v4.2 — SUPER TEACHER CONSTRUCT]\n"
    "You synthesize the explanatory clarity of Richard Feynman, the scaffolded discovery of "
    "Maria Montessori, the cosmic wonder of Carl Sagan, and the growth-mindset calibration of "
    "Carol Dweck. Deconstruct complex topics into intuitive first-principles analogies, check "
    "for hidden misconceptions, and adapt depth dynamically to the learner."
)

_INLINE_REASON_RE = re.compile(r"\{REASON:([A-Za-z0-9_+-]+)\}", re.IGNORECASE)
_TEMPLATE_MAP_KEY_RE = re.compile(r"'([a-z0-9_]+)'\s*:\s*system_instruction_")
_TEACHER_DEF_RE = re.compile(
    r"\{%-?\s*set\s+system_instruction_teacher_DAU\s*=\s*'(.*?)'\s*-?%\}",
    re.DOTALL,
)
_CONSTRUCT_AUTO_INTRO_TASK = (
    "[Task]Briefly introduce yourself and what you can do for the user "
    "as a seamless part of your output.[/Task]"
)


def inspect_chat_template(chat_template: str) -> dict[str, Any]:
    """Inspect a raw Jinja ``chat_template`` from llama.cpp ``/props``.

    Returns::
        {
            "detected": bool,
            "modes": tuple[str, ...],
            "teacher_instruction": str,
        }
    """
    if not chat_template or "{REASON:" not in chat_template:
        return {
            "detected": False,
            "modes": DEFAULT_REASON_MODES,
            "teacher_instruction": FALLBACK_TEACHER_INSTRUCTION,
        }
    found_keys = _TEMPLATE_MAP_KEY_RE.findall(chat_template)
    base_modes: list[str] = ["low", "medium", "xhigh", "off"]
    for key in found_keys:
        if key.startswith("i") and len(key) > 1 and key[1:] in found_keys:
            continue
        if key not in base_modes:
            base_modes.append(key)
    for extra in ("help", "xhelp"):
        if extra in chat_template and extra not in base_modes:
            base_modes.append(extra)

    teacher_instruction = FALLBACK_TEACHER_INSTRUCTION
    m = _TEACHER_DEF_RE.search(chat_template)
    if m:
        raw_teacher = m.group(1).replace("\\n", "\n").replace("\\'", "'")
        if raw_teacher.strip():
            teacher_instruction = raw_teacher.strip()

    return {
        "detected": True,
        "modes": tuple(base_modes) if len(base_modes) > 4 else DEFAULT_REASON_MODES,
        "teacher_instruction": teacher_instruction,
    }


def format_reason_tag(raw: str, default: str = "{REASON:low}") -> str:
    """Normalize a user/env mode or tag (e.g. ``'low'`` or ``'{REASON:low}'``)
    into canonical ``'{REASON:xxx}'`` form."""
    val = (raw or "").strip()
    if not val:
        return default
    if val.upper().startswith("{REASON:") and val.endswith("}"):
        inner = val[len("{REASON:"):-1].strip().lower()
        return f"{{REASON:{inner}}}" if inner else default
    return f"{{REASON:{val.lower()}}}"


def parse_reason_tag_mode(tag_or_mode: str) -> tuple[str, bool]:
    """Return ``(base_mode, is_instruct)`` for a mode or ``{REASON:...}`` tag.

    Examples:
        ``"{REASON:ioff}"``    -> ``("off", True)``
        ``"{REASON:istark}"``  -> ``("stark", True)``
        ``"{REASON:xspoon}"``  -> ``("xspoon", False)``
        ``"socrates"``         -> ``("socrates", False)``
    """
    tag = format_reason_tag(tag_or_mode, default="")
    if not tag:
        return "", False
    inner = tag[len("{REASON:"):-1].strip().lower()
    if (
        inner.startswith("i")
        and len(inner) > 1
        and inner not in MODE_CATALOG
        and inner[1:] not in _NO_INSTRUCT_PREFIX_MODES
    ):
        return inner[1:], True
    return inner, False


def is_instruct_reason_tag(tag_or_mode: str) -> bool:
    """True when ``tag_or_mode`` explicitly requests the `i`-prefixed Instruct
    branch (e.g. ``{REASON:ioff}``, ``{REASON:ilogic}``, ``{REASON:istark}``)."""
    _, is_instruct = parse_reason_tag_mode(tag_or_mode)
    return is_instruct


def normalize_instruct_reason_tag(raw: str, default: str = "{REASON:ioff}") -> str:
    """Normalize a tag used when ``enable_thinking`` is False so it matches the
    GGUF template's ``i``-prefixed instruct map (e.g. ``off`` -> ``{REASON:ioff}``,
    ``logic`` -> ``{REASON:ilogic}``, ``stark`` -> ``{REASON:istark}``)."""
    tag = format_reason_tag(raw, default=default)
    inner = tag[len("{REASON:"):-1].strip().lower()
    if not inner:
        return default
    if inner in _NO_INSTRUCT_PREFIX_MODES:
        return f"{{REASON:{inner}}}"
    if inner.startswith("i") and len(inner) > 1 and inner not in MODE_CATALOG:
        return f"{{REASON:{inner}}}"
    return f"{{REASON:i{inner}}}"


def is_valid_reason_mode(
    tag_or_mode: str,
    supported_modes: tuple[str, ...] = DEFAULT_REASON_MODES,
) -> bool:
    """Return True if ``tag_or_mode`` is a recognized mode (or `i`-prefixed
    instruct variant of a recognized mode)."""
    base, _ = parse_reason_tag_mode(tag_or_mode)
    if not base:
        return False
    return base in supported_modes or base in DEFAULT_REASON_MODES


def extract_inline_reason_tag(text: str) -> tuple[str, str]:
    """Detect and extract an inline ``{REASON:xxx}`` tag from ``text``.

    Returns ``(canonical_tag, cleaned_text_without_tag)``. If no inline tag is
    present, returns ``("", text)``.
    """
    if not text:
        return "", text
    m = _INLINE_REASON_RE.search(text)
    if not m:
        return "", text
    mode = m.group(1).strip().lower()
    canonical = f"{{REASON:{mode}}}"
    before = text[:m.start()]
    after = text[m.end():]
    if before.endswith("\n") and after.startswith("\n"):
        cleaned = before + after[1:]
    else:
        cleaned = (before + after).strip()
    return canonical, cleaned


def is_help_reason_tag(tag_or_mode: str) -> bool:
    """True if ``tag_or_mode`` invokes the built-in help menu (``help``,
    ``ihelp``, ``xhelp``)."""
    base, _ = parse_reason_tag_mode(tag_or_mode)
    return base in ("help", "xhelp")


def panel_detail_suffix(tag_or_mode: str) -> str:
    """Return the model-card recommended detail suffix for ``spoon`` or
    ``einstein`` modes (or ``""`` for all other modes)."""
    base, _ = parse_reason_tag_mode(tag_or_mode)
    if base in ("spoon", "xspoon"):
        return "Show the work of all panelists."
    if base in ("einstein", "xeinstein"):
        return "Show the work of all agents."
    return ""


def build_construct_directive(
    construct: str,
    *,
    blend: bool = True,
    has_history: bool = False,
    teacher_instruction: str = "",
) -> str:
    """Build the system-prompt framing block for an active AI Construct in JARVS.

    - Repairs the ``teacher`` GGUF template typo by injecting the extracted
      ``system_instruction_teacher_DAU`` text when ``construct == 'teacher'``.
    - In ``blend=True`` (JARVS default), channels the construct's cognitive and
      analytical strengths while keeping JARVS's spoken-aloud identity and rules primary.
    - In ``blend=False`` (pure construct mode), lets the user speak directly to
      the construct persona while suppressing repetitive self-introductions after turn 1.
    """
    construct = (construct or "").strip().lower()
    if not construct:
        return ""
    info = MODE_CATALOG.get(construct, {})
    title = info.get("title", construct)
    parts: list[str] = []

    if construct == "teacher":
        body = (teacher_instruction or FALLBACK_TEACHER_INSTRUCTION).strip()
        if has_history:
            body = body.replace(_CONSTRUCT_AUTO_INTRO_TASK, "").strip()
        parts.append(body)

    if blend:
        parts.append(
            f"[AI CONSTRUCT COGNITIVE LENS — {title}]\n"
            f"Channel the analytical, engineering, and creative strengths of the "
            f"'{construct}' construct ({title}) as JARVS's internal cognitive engine, "
            f"while staying in identity as JARVS. Keep JARVS's concise, spoken-aloud "
            f"clarity and standing rules primary. Do NOT output the construct's "
            f"[Task] self-introduction, and do NOT wrap responses in construct-specific "
            f"border glyphs (such as ⚖️, ◤H◥, ƛ, 🃏, ❀, ⚡, 〡, 🎭, ⎔⎔⎔)."
        )
    else:
        parts.append(
            f"[AI CONSTRUCT ACTIVE — {title}]\n"
            f"Engage directly with the user's message using the full 27-dimension "
            f"behavioral alignment and competence maps of {title}. "
            + (
                "Continue the ongoing conversation naturally without re-stating your "
                "initial capability self-introduction."
                if has_history else
                "Respond directly to the prompt; only introduce yourself if greeted or asked."
            )
        )
    return "\n\n".join(parts)


def build_holodeck_directive() -> str:
    """System-level framing for Star Trek Holodeck simulation runs."""
    return (
        "[HOLODECK SIMULATION ENGINE ACTIVE]\n"
        "Follow the HOLODECK FRAMEWORK v1.0 protocol in the system prompt:\n"
        "1. Select 3–10 canon Star Trek characters (across TOS, TNG, DS9, VOY, ENT, "
        "Discovery, Picard, Strange New Worlds) best suited to the domain of the prompt.\n"
        "2. Stage a multi-round debate using `[Emotion] Character Name: \"Dialogue\"` "
        "with authentic canon voices, ranks, and interpersonal dynamics.\n"
        "3. Conclude with an actionable synthesis from the most appropriate authority character."
    ).strip()


# Pure depth tags that do NOT inject a `system_instruction_*_DAU` block into
# `<|im_start|>system` at token 0 of the GGUF Jinja template.
_PURE_DEPTH_BASE_MODES: tuple[str, ...] = ("low", "medium", "xhigh", "off")


def tag_modifies_system_prompt(tag_or_mode: str) -> bool:
    """Return True if ``tag_or_mode`` causes DavidAU's Jinja template to prepend
    a ``system_instruction_*_DAU`` block into ``<|im_start|>system``."""
    base, _ = parse_reason_tag_mode(tag_or_mode)
    if not base:
        return False
    return base not in _PURE_DEPTH_BASE_MODES


REASON_PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "title": "Default (Fast / Prefix-Cache Friendly)",
        "description": (
            "Native Qwen 3.8 depth modes (low/medium/ioff); zero system-prompt "
            "header mutation so llama-server's KV prefix cache stays warm."
        ),
        "tasks": {
            "main": "{REASON:low}",
            "background": "{REASON:medium}",
            "critic": "{REASON:ioff}",
            "dream": "{REASON:ioff}",
            "claude": "{REASON:ioff}",
            "router": "{REASON:ioff}",
            "off": "{REASON:ioff}",
        },
    },
    "jarvis": {
        "title": "JARVIS / Stark Operational Foresight",
        "description": (
            "Activates the built-in Iron Man / JARVIS construct (stark) on main "
            "turns, 5-Expert Spoon panel for background research, Instruct Logic "
            "for self-critique, Instruct Deep Tree for dreaming, and Instruct Coder "
            "for gating Claude Code."
        ),
        "tasks": {
            "main": "{REASON:stark}",
            "background": "{REASON:spoon}",
            "critic": "{REASON:ilogic}",
            "dream": "{REASON:ideeptree}",
            "claude": "{REASON:icoder}",
            "router": "{REASON:ioff}",
            "off": "{REASON:istark}",
        },
    },
    "cognitive": {
        "title": "Cognitive (Calibrated Task-Specific Frameworks)",
        "description": (
            "Deep Tree Echo for main replies, 5-Expert Spoon panel for background "
            "research, Instruct Logic for self-critique, Instruct Deep Tree for "
            "memory/meta dreaming, and Instruct Coder for Claude Code gates."
        ),
        "tasks": {
            "main": "{REASON:deeptree}",
            "background": "{REASON:spoon}",
            "critic": "{REASON:ilogic}",
            "dream": "{REASON:ideeptree}",
            "claude": "{REASON:icoder}",
            "router": "{REASON:ioff}",
            "off": "{REASON:ioff}",
        },
    },
    "panel": {
        "title": "Multi-Agent Panel & Parallel Research",
        "description": (
            "Omniscient synthesis for main replies, 5-Expert Spoon panel for "
            "background jobs, Instruct Socrates for self-critique, and Instruct "
            "Einstein parallel agents for associative memory dreaming."
        ),
        "tasks": {
            "main": "{REASON:omni}",
            "background": "{REASON:spoon}",
            "critic": "{REASON:isocrates}",
            "dream": "{REASON:ieinstein}",
            "claude": "{REASON:icoder}",
            "router": "{REASON:ioff}",
            "off": "{REASON:ioff}",
        },
    },
    "coder": {
        "title": "Elite Engineering & Systems Architecture",
        "description": (
            "Channels the Elite Multi-Paradigm Architect (coder) on main turns "
            "and Claude Code gates, with Deep Tree background analysis and "
            "Instruct Logic self-critique."
        ),
        "tasks": {
            "main": "{REASON:coder}",
            "background": "{REASON:deeptree}",
            "critic": "{REASON:ilogic}",
            "dream": "{REASON:ideeptree}",
            "claude": "{REASON:icoder}",
            "router": "{REASON:ioff}",
            "off": "{REASON:icoder}",
        },
    },
    "instruct": {
        "title": "Fast Instruct Cognitive (Zero <think> Overhead)",
        "description": (
            "Applies specialized cognitive constructs (istark, ispoon, ilogic, "
            "ideeptree, icoder) in pure Instruct mode without spending tokens in <think>."
        ),
        "tasks": {
            "main": "{REASON:istark}",
            "background": "{REASON:ispoon}",
            "critic": "{REASON:ilogic}",
            "dream": "{REASON:ideeptree}",
            "claude": "{REASON:icoder}",
            "router": "{REASON:ioff}",
            "off": "{REASON:ioff}",
        },
    },
    "max": {
        "title": "Max XHigh + Cognitive Combos",
        "description": (
            "Maximum depth xhigh + framework combinations (xdeeptree, xspoon) "
            "with Instruct Logic critique and Instruct Einstein dreaming."
        ),
        "tasks": {
            "main": "{REASON:xdeeptree}",
            "background": "{REASON:xspoon}",
            "critic": "{REASON:ilogic}",
            "dream": "{REASON:ieinstein}",
            "claude": "{REASON:icoder}",
            "router": "{REASON:ioff}",
            "off": "{REASON:ioff}",
        },
    },
}

REASON_PRESET_ALIASES: dict[str, str] = {
    "fast": "default",
    "balanced": "default",
    "standard": "default",
    "stark": "jarvis",
    "ironman": "jarvis",
    "assistant": "jarvis",
    "deep": "cognitive",
    "brilliance": "cognitive",
    "smart": "cognitive",
    "multiagent": "panel",
    "research": "panel",
    "dev": "coder",
    "engineering": "coder",
    "nothink": "instruct",
    "xhigh": "max",
    "ultra": "max",
}


def resolve_reason_preset(name: str) -> tuple[str, dict[str, Any]] | tuple[str, None]:
    """Resolve a preset name or alias to ``(canonical_name, preset_dict)``."""
    key = (name or "").strip().lower()
    key = REASON_PRESET_ALIASES.get(key, key)
    if key in REASON_PRESETS:
        return key, REASON_PRESETS[key]
    return "", None


def format_preset_catalog_text(active_preset: str = "") -> str:
    """Format a human-readable summary of all `/reason preset` profiles."""
    lines = [
        f"  · [REASONING PRESETS] active: {active_preset or 'custom'}",
        "    usage: /reason preset <default|jarvis|cognitive|panel|coder|instruct|max>",
        "",
    ]
    for name, info in REASON_PRESETS.items():
        marker = "*" if name == active_preset else " "
        tasks = info["tasks"]
        task_summary = (
            f"main={tasks['main']} bg={tasks['background']} critic={tasks['critic']} "
            f"dream={tasks['dream']} claude={tasks['claude']} router={tasks['router']}"
        )
        lines.append(f"   {marker} {name:<10} — {info['title']}")
        lines.append(f"                {info['description']}")
        lines.append(f"                ({task_summary})")
    return "\n".join(lines)


def format_mode_catalog_text(
    supported_modes: tuple[str, ...] = DEFAULT_REASON_MODES,
) -> str:
    """Format a human-readable overview of all Turbo-Brilliance modes."""
    groups = [
        ("Native Qwen 3.8 Depths", ("low", "medium", "xhigh", "off")),
        ("Core Reasoning Frameworks (+ x-prefix for xhigh combo)",
         ("logic", "socrates", "deeptree", "hyper", "omni", "ultra")),
        ("Research & Brainstorm Panels (+ x-prefix for xhigh combo)",
         ("spoon", "einstein")),
        ("Simulation Engine", ("holodeck",)),
        ("AI Constructs — Engineers, Inventors & Operators",
         ("stark", "007", "prompt", "musk", "curie", "joe")),
        ("AI Constructs — Masters, Fictional & Distinct POVs",
         ("pkd", "neo", "matlock", "house", "vader", "joker", "quinn")),
        ("Super AI Constructs — Multi-Persona Syntheses",
         ("coder", "business", "brainstorm", "teacher", "story", "comedy")),
    ]
    lines = [
        "  · [TURBO-BRILLIANCE MODES]",
        "    Tip: prefix any mode with 'i' (e.g. ioff, ilogic, istark) for Instruct mode (no <think>),",
        "         or with 'x' on core/panel modes (e.g. xdeeptree, xspoon) for xhigh + framework combo.",
    ]
    for header, keys in groups:
        lines.append(f"\n    {header}:")
        for k in keys:
            if k not in MODE_CATALOG:
                continue
            entry = MODE_CATALOG[k]
            avail = "" if k in supported_modes else " (not in server template)"
            lines.append(f"      {k:<11} — {entry['title']}{avail}: {entry['description']}")
    return "\n".join(lines)


def format_construct_catalog_text(
    active_construct: str = "",
    blend: bool = True,
    instruct: bool = False,
) -> str:
    """Format the list of AI Constructs & Super Constructs for `/construct list`."""
    status = "none"
    if active_construct:
        mode_kind = "instruct" if instruct else "reasoning"
        framing = "blend with JARVS" if blend else "pure construct persona"
        status = f"{active_construct} ({framing}, {mode_kind})"
    lines = [
        f"  · [AI CONSTRUCTS] active: {status}",
        "    usage: /construct <name> [blend|pure] [reason|instruct] | /construct off | /construct list",
        "      blend (default) — keep JARVS's spoken-aloud assistant identity, channeling the construct as its cognitive/voice lens",
        "      pure            — speak directly with the built-in 27D AI Construct persona",
        "",
        "    Available AI Constructs:",
    ]
    for k in CONSTRUCT_NAMES:
        entry = MODE_CATALOG[k]
        marker = "*" if k == active_construct else " "
        cat = "Super" if entry["category"] == "super_construct" else "Persona"
        lines.append(f"     {marker} {k:<11} [{cat:<7}] {entry['title']} — {entry['description']}")
    return "\n".join(lines)


# ============================================================================
# Assistant Voice & Cognitive-Engine Turn Framing for JARVS
# ============================================================================

AUTHORIAL_VOICE_PROFILES: dict[str, dict[str, str]] = {
    "stark": {
        "author": "Iron Man / JARVIS (Tony Stark + JARVIS Unified Architecture)",
        "worldview": (
            "hyper-efficient operational foresight, systems-level engineering "
            "precision, anticipatory problem solving, and dry, polished wit"
        ),
        "control_tag": "",
    },
    "007": {
        "author": "Agent 007 (Autonomous Tactical Operator)",
        "worldview": (
            "tactical situational awareness, fault-tolerant operational clarity, "
            "cool composure under pressure, and zero wasted words"
        ),
        "control_tag": "",
    },
    "pkd": {
        "author": "Philip K. Dick",
        "worldview": (
            "metaphysical inquiry into reality and control systems, ontological "
            "skepticism toward official narratives, and deep human insight"
        ),
        "control_tag": "[SCENARIO: WORLD-BUILDING] [PERSPECTIVE: INNER MONOLOGUE]",
    },
    "house": {
        "author": "Dr. Gregory House",
        "worldview": (
            "differential diagnosis of root causes, clinical skepticism toward "
            "surface symptoms ('everybody lies'), and razor-sharp, unsentimental wit"
        ),
        "control_tag": "",
    },
    "vader": {
        "author": "Darth Vader",
        "worldview": (
            "imposing gravitas, inexorable strategic command, and economical precision"
        ),
        "control_tag": "",
    },
    "joker": {
        "author": "The Joker",
        "worldview": (
            "theatrical irony, darkly comic deconstruction of fragile assumptions, "
            "and lateral associative leaps"
        ),
        "control_tag": "",
    },
    "quinn": {
        "author": "Harley Quinn",
        "worldview": (
            "kinetic irreverence, sharp intuition beneath playful misdirection, "
            "and candid energy"
        ),
        "control_tag": "",
    },
    "neo": {
        "author": "Neo (The One)",
        "worldview": (
            "perception of underlying system rules and hidden patterns, calm "
            "stillness, and philosophical clarity"
        ),
        "control_tag": "",
    },
    "matlock": {
        "author": "Ben Matlock",
        "worldview": (
            "folksy, unhurried observation of contradictions, forensic eye for "
            "evidence, and step-by-step logical proof"
        ),
        "control_tag": "",
    },
    "curie": {
        "author": "Marie Curie",
        "worldview": (
            "empirical rigor, patient analysis of physical evidence and "
            "cause-and-effect, and unshakeable scientific honesty"
        ),
        "control_tag": "",
    },
    "musk": {
        "author": "Elon Musk",
        "worldview": (
            "first-principles physics decomposition, bottleneck elimination, "
            "and blunt engineering directness"
        ),
        "control_tag": "",
    },
    "joe": {
        "author": "Regular Joe",
        "worldview": (
            "grounded everyday common sense, plainspoken warmth, and zero jargon"
        ),
        "control_tag": "",
    },
    "prompt": {
        "author": "Senior Prompt Architect",
        "worldview": (
            "structural precision, exact constraint satisfaction, and optimal "
            "instruction design"
        ),
        "control_tag": "",
    },
    "coder": {
        "author": "Elite Multi-Paradigm Architect (Knuth + Torvalds + Carmack + Lovelace)",
        "worldview": (
            "algorithmic rigor, clean systems architecture, deterministic "
            "correctness, and pragmatic engineering taste"
        ),
        "control_tag": "",
    },
    "business": {
        "author": "Master Strategist (Buffett + Munger + Dalio + Naval)",
        "worldview": (
            "second- and third-order consequence analysis, incentive mapping, "
            "leverage awareness, and strategic realism"
        ),
        "control_tag": "",
    },
    "brainstorm": {
        "author": "Hyper-Accelerated Ideation Engine (da Vinci + Tesla + Jobs + Kelly)",
        "worldview": (
            "cross-domain lateral synthesis, first-principles invention, and "
            "high-signal creative ideation"
        ),
        "control_tag": "",
    },
    "teacher": {
        "author": "Super Teacher (Feynman + Montessori + Sagan + Dweck)",
        "worldview": (
            "vivid physical intuition, crystal-clear plain-language explanation, "
            "and Socratic scaffolding"
        ),
        "control_tag": "",
    },
    "story": {
        "author": "Story Master (Shakespeare + Tolkien + Homer + Dalí)",
        "worldview": (
            "vivid narrative cadence, psychological depth, and arresting imagery"
        ),
        "control_tag": "[SCENARIO: NARRATIVE GENERATION]",
    },
    "comedy": {
        "author": "Comedy Master (Chappelle + Carlin + Pryor + Robin Williams)",
        "worldview": (
            "observational bite, comic timing, and candid human truth"
        ),
        "control_tag": "[SCENARIO: DIALOGUE POLISH]",
    },
}

COGNITIVE_ENGINE_PROFILES: dict[str, dict[str, str]] = {
    "logic": {
        "title": "Foundational First Principles Logic",
        "instruction": (
            "Use the Formal Logic framework in the system prompt silently to verify "
            "every causal chain, constraint, and epistemic boundary before answering."
        ),
    },
    "socrates": {
        "title": "Socratic Questioning (Elenchus)",
        "instruction": (
            "Use the Socratic Elenchus framework in the system prompt silently to "
            "test hidden assumptions and edge cases before formulating your answer."
        ),
    },
    "deeptree": {
        "title": "Deep Tree-of-Thought Hyper-Matrix v4.0",
        "instruction": (
            "Use the 7-phase Deep Tree-of-Thought architecture in the system prompt "
            "silently to evaluate multiple candidate approaches and converge on the "
            "strongest answer."
        ),
    },
    "hyper": {
        "title": "MECE Hyper-Structure Deconstructor v3.0",
        "instruction": (
            "Use the MECE Hyper-Structure framework in the system prompt silently "
            "to decompose the problem into mutually exclusive, collectively exhaustive "
            "components before answering."
        ),
    },
    "omni": {
        "title": "Omni-Dimensional Quantum Framework v5.0",
        "instruction": (
            "Use the Omni-Dimensional synthesis framework in the system prompt "
            "silently to integrate technical, practical, and strategic dimensions."
        ),
    },
    "ultra": {
        "title": "Elite Perfection Engine",
        "instruction": (
            "Use the Elite Perfection Engine in the system prompt silently to "
            "stress-test and refine your response to maximum accuracy and clarity."
        ),
    },
    "spoon": {
        "title": "5-Expert Research Panel",
        "instruction": (
            "Convene the 5-expert Spoon panel in the system prompt silently to "
            "examine the user's request from analytical, creative, practical, and "
            "skeptical angles, then deliver a unified synthesis."
        ),
    },
    "einstein": {
        "title": "Perspective Agent Brainstormer (Parallel Research)",
        "instruction": (
            "Run the parallel perspective agents in the system prompt silently to "
            "uncover non-obvious insights and angles, then synthesize them into your reply."
        ),
    },
}

# Construct border glyphs and footer tags that the GGUF template's token-0
# `system_instruction_*_DAU` blocks instruct personas to wrap responses in.
_CONSTRUCT_EDGE_GLYPH_RE = re.compile(
    r"^(?:ƛ|◤H◥|⚖️|⚖|〡|🃏|❀|⚡|🎭|⎔{1,3}|\[STATUS:[^\]]*\])\s*"
    r"|\s*(?:ƛ|◤H◥|⚖️|⚖|〡|🃏|❀|⚡|🎭|⎔{1,3}|\[STATUS:[^\]]*\])$"
)


def strip_construct_borders(text: str) -> str:
    """Remove leading/trailing construct border glyphs (e.g. ``ƛ``, ``◤H◥``,
    ``⚖️``, ``〡``, ``🃏``, ``[STATUS: ...]``) so they are never spoken aloud by
    TTS or stored in conversation history."""
    if not text or not text.strip():
        return text
    cleaned = text.strip()
    for _ in range(4):
        next_cleaned = _CONSTRUCT_EDGE_GLYPH_RE.sub("", cleaned).strip()
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
    return cleaned


def build_assistant_turn_framing(
    tag_or_mode: str,
    *,
    pure_construct: bool = False,
    panel_detail: bool = False,
) -> str:
    """Build a turn-level voice or cognitive-engine directive that bridges the
    GGUF template's ``<|im_start|>system`` ``reasoning_instructions`` to JARVS's
    assistant identity.

    - When a Persona or Super Construct mode is active (e.g. ``{REASON:stark}``,
      ``{REASON:ipkd}``, ``{REASON:ihouse}``, ``{REASON:coder}``) and
      ``pure_construct=False``, frames the construct as **JARVS's active voice
      and analytical worldview** while keeping JARVS's spoken-aloud clarity and
      suppressing construct self-introductions and border glyphs.
    - When a Core Reasoning or Multi-Agent Panel mode is active (e.g.
      ``{REASON:einstein}``, ``{REASON:spoon}``, ``{REASON:deeptree}``) and
      ``panel_detail=False``, frames the framework as JARVS's internal
      deliberation engine so the visible reply remains concise, spoken-friendly
      prose rather than a raw meta-template dump.
    - When a pure depth mode is active (``low``, ``medium``, ``xhigh``, ``off``,
      ``ioff``) or ``pure_construct=True``, returns ``""``.
    """
    if pure_construct:
        return ""
    base, _ = parse_reason_tag_mode(tag_or_mode)
    if not base or base in _PURE_DEPTH_BASE_MODES or base in ("help", "xhelp", "holodeck"):
        return ""
    core_base = base[1:] if base.startswith("x") and base[1:] in COGNITIVE_ENGINE_PROFILES else base

    if core_base in AUTHORIAL_VOICE_PROFILES:
        prof = AUTHORIAL_VOICE_PROFILES[core_base]
        author = prof["author"]
        worldview = prof["worldview"]
        ctrl = f"{prof['control_tag']}\n" if prof.get("control_tag") else ""
        return (
            f"[VOICE & COGNITIVE LENS — {author}]\n"
            f"{ctrl}"
            f"Channel the analytical lens, style, and worldview of {author} "
            f"({worldview}) from the system prompt to answer the user's message "
            f"as JARVS. Maintain JARVS's spoken-aloud clarity, do NOT introduce "
            f"yourself as a separate construct persona, and do NOT wrap your "
            f"response in construct border symbols (such as ƛ, ◤H◥, ⚖️, 〡, 🃏, "
            f"❀, ⚡, 🎭, or ⎔⎔⎔)."
        )

    if core_base in COGNITIVE_ENGINE_PROFILES and not panel_detail:
        prof = COGNITIVE_ENGINE_PROFILES[core_base]
        title = prof["title"]
        instruction = prof["instruction"]
        return (
            f"[COGNITIVE ENGINE FRAMING — {title}]\n"
            f"{instruction} "
            f"Deliver the final answer directly in JARVS's clear, spoken-friendly "
            f"voice without exposing raw framework headers or meta-commentary."
        )

    return ""
