"""web_access.py — live-web primitives (DuckDuckGo search + URL fetch).

Engine-agnostic: no imports from the rest of JARVS, no global toggles, no rendering.
Callers (currently the background-job worker) wrap these with their own gating.
Vendored from ENGRAM2.

`ddgs` is an optional dependency — DDGS_AVAILABLE is False when it's missing, and
`web_search` returns an explanatory string rather than raising.
"""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except Exception:  # package absent or import error — degrade gracefully
    DDGS = None
    DDGS_AVAILABLE = False

URL_CONTEXT_LIMIT = 8000


class _PageTextExtractor(HTMLParser):
    """Collect readable page text, dropping script/style/nav noise."""

    _SKIP = {"script", "style", "noscript", "template", "svg"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._title_parts: list[str] = []
        self._in_title = False
        self._chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self._title_parts.append(text)
        else:
            self._chunks.append(text)

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self._chunks)).strip()


def web_search(query: str, max_results: int = 5) -> str:
    """Real-time DuckDuckGo text search. Returns a formatted block or a message
    (never raises) so callers can treat unavailability as data, not an exception."""
    if DDGS is None:
        return "Search unavailable: the 'ddgs' package is not installed."
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Source: {r.get('href', '')}\n"
                               f"Content: {r.get('body', '')}")
        return "\n\n".join(results) or "Search returned no results."
    except Exception as e:
        return f"Search failed: {e}"


def search_results(query: str, max_results: int = 5) -> list[dict]:
    """Structured variant: a list of {title, href, body} dicts (empty on failure)."""
    if DDGS is None:
        return []
    try:
        with DDGS() as ddgs:
            return [{"title": r.get("title", ""), "href": r.get("href", ""),
                     "body": r.get("body", "")}
                    for r in ddgs.text(query, max_results=max_results)]
    except Exception:
        return []


def extract_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s<>\]\)\"']+", text or "")
    return [u.rstrip(".,;:!?") for u in urls]


def fetch_url_content(url: str, timeout: int = 30,
                      max_chars: int = URL_CONTEXT_LIMIT) -> dict:
    """Fetch a URL and return {url, title, kind, content_type, text}. Raises on
    network errors (callers wrap in try/except); PDFs are not ingested."""
    headers = {"User-Agent": "JARVS/0.1 (+local research assistant)"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "application/pdf" in content_type.lower() or \
            urlparse(url).path.lower().endswith(".pdf"):
        return {"url": url, "title": os.path.basename(urlparse(url).path) or url,
                "kind": "pdf", "content_type": content_type,
                "text": "PDF content is not ingested."}
    response.encoding = response.encoding or "utf-8"
    parser = _PageTextExtractor()
    parser.feed(response.text)
    text = parser.text() or "No extractable text was found."
    return {"url": url, "title": parser.title or url, "kind": "html",
            "content_type": content_type, "text": text[:max_chars]}


_WEB_MARKERS = ("search", "look up", "internet", "web", "verify", "fact-check",
                "fact check", "sources", "current", "latest", "recent", "today",
                "yesterday", "this week", "news", "google", "who won", "price of",
                "research")
_FACT_STARTERS = ("who is ", "what is ", "what are ", "when did ", "when is ",
                  "where is ", "where did ", "how many ", "how much ", "who won ")


def needs_web_search(user_input: str) -> bool:
    """Cheap offline heuristic: explicit web/verification markers, or a factual
    question that likely needs current public data."""
    text = (user_input or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _WEB_MARKERS):
        return True
    return text.endswith("?") and text.startswith(_FACT_STARTERS)


# Cues that a claim/question is time-sensitive — version/release/availability talk,
# dates, prices, "is there a X", a product followed by a version number (Gemma 4,
# Llama 3, GPT 5). Used to LOWER the search bar when the user has set a standing rule to
# always verify current information.
_TIME_SENSITIVE_RE = re.compile(
    r"\b(?:latest|newest|current|recent|release[ds]?|releasing|announce[ds]?|"
    r"unveil\w*|available|out yet|come out|coming|exists?|is there|version|"
    r"successor|next (?:version|gen\w*)|new(?:est)? (?:version|model|release)|"
    r"v?\d+(?:\.\d+)+|today|right now|as of|20\d\d|price|stock|who won|"
    r"up[- ]?to[- ]?date|still (?:the )?(?:newest|latest|current)|"
    r"[a-z][a-z0-9.+-]{2,}\s+\d{1,2}(?:\.\d+)?)\b", re.I)

# A question ABOUT JARVS/the conversation ("did you find…", "are you sure?") rather than
# a factual lookup — not something to run a web search for.
_META_Q_RE = re.compile(
    r"^\s*(?:did|do|does|can|could|would|will|have|has|are|were)\s+you\b", re.I)


def looks_time_sensitive(user_input: str) -> bool:
    """True when the message turns on current facts (versions, releases, dates, prices)
    — part of the lower bar used under a 'always verify online' standing rule."""
    return bool(_TIME_SENSITIVE_RE.search(user_input or ""))


# A follow-up asking about a search the user expected ("did you find something?", "any
# luck?", "and?") — not itself searchable, but a cue to run the PREVIOUS question's
# search and present the results.
_FOLLOWUP_RE = re.compile(
    r"\bdid you (?:find|search|look|get)\b|\bfound (?:anything|something|it)\b|"
    r"\bfind anything\b|\bany (?:luck|results?|news|updates?)\b|"
    r"\bwhat did you find\b|\bresults?\b\s*\??|^\s*(?:and|so|well|any(?:thing)?)\s*\?",
    re.I)


def is_search_followup(user_input: str) -> bool:
    """True when the user is asking about the outcome of a search they expected."""
    return bool(_FOLLOWUP_RE.search(user_input or ""))


def last_searchable(history: list[dict], lower_bar: bool = True) -> str:
    """The most recent prior USER message that warranted a search, scanning backwards —
    the query to re-run when the user follows up with 'did you find something?'."""
    for msg in reversed(history or []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if should_search(content, lower_bar=lower_bar):
                return content
    return ""


def should_search(user_input: str, lower_bar: bool = False) -> bool:
    """Decide whether the reply path should run a live search. Always searches on the
    explicit markers. With ``lower_bar`` (a standing 'verify online' rule is on) it is
    deliberately aggressive — it also searches on time-sensitive cues and on any genuine
    factual question (ends with '?' and is not a question about JARVS itself)."""
    if needs_web_search(user_input):
        return True
    if not lower_bar:
        return False
    if looks_time_sensitive(user_input):
        return True
    t = (user_input or "").strip()
    return t.endswith("?") and not _META_Q_RE.match(t)
