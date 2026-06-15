# SPDX-License-Identifier: MIT
"""Deterministic content filter.

Classifies incoming text into importance levels (background /
important / critical / principle) using language-agnostic structural
signals and multilingual keyword lists.
"""
import re
import time
from typing import Any

# ── Structural / language-agnostic precision patterns ────────────────────────
PRECISION_PATTERNS: dict[str, str] = {
    "link": r"https?://[^\s\)\]\"']+",
    "email": r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}",
    "phone": r"\+?\d[\d\s\-\(\)]{7,}",
    "version": r"\bv?(?:\d+\.\d+(?:\.\d+)?(?:[._-]?(?:alpha|beta|rc|dev))?)\b",
    "error": r"\b(?:error|exception|traceback|errno|assert|fail|panic|fatal)\b",
    "uuid": (
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
        r"[0-9a-f]{4}-[0-9a-f]{12}\b"
    ),
}

# ── Multilingual keyword sets ────────────────────────────────────────────────
CRITICAL = [
    "decided", "chosen", "forbidden", "must not", "blocker",
    "final", "critical", "rejected", "locked in", "resolved",
]

IMPORTANT = [
    "important", "artifact", "requirement", "dependency",
    "needed", "necessary", "mandatory", "key", "essential", "must",
    # Agent conversational patterns that indicate meaningful work
    "explore", "examine", "check", "investigate", "analyze", "review",
    "implement", "build", "create", "setup", "configure",
    "fix", "debug", "modify", "update", "refactor",
    "add", "remove", "test", "deploy", "write", "design",
]

CONFLICT = [
    "but", "however", "contradicts", "changed", "cancelled",
    "instead", "revised", "conflict", "diverges", "overrides",
]

PRINCIPLE_SIGNALS = [
    "never", "always", "remember this", "non-negotiable",
    "project rule", "architectural principle", "this is a rule", "hard rule",
]

URGENCY_SIGNALS: dict[str, list[str]] = {
    "deadline_h": ["in 1 hour", "in 2 hours", "within an hour", "asap", "right now"],
    "deadline_d": ["today", "tomorrow", "by eod", "deadline today"],
    "deadline_w": [
        "this week", "by friday", "in a week", "end of week", "by eow",
    ],
}

_DEADLINE_PATTERN = re.compile(
    r'(?<![v\d.])(\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?)'
    r'|(\d{4}-\d{2}-\d{2})'
    r'|((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2})\b',
    re.IGNORECASE,
)

_CODE_PATTERN = re.compile(
    r'(?:'
    r'def \w+\s*\('
    r'|class \w+[\s(:]'
    r'|import \w+'
    r'|```'
    r'|`[^`]+`'
    r'|\w+\.\w+\(\)'
    r')',
    re.MULTILINE,
)

# ── Agent internal monologue patterns ────────────────────────────────────────
# Matches short, generic introspective phrases that are noise, not content.
MONOLOGUE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^(?:let me|now let me|i'll|I'll|i will|i WILL|first,?\s*let me"
        r"|next,?\s*let me|looking at|examining|now exploring|exploring"
        r"|checking|analyzing|investigating|reading|reviewing"
        r"|verifying|confirming)\b",
        re.IGNORECASE,
    ),
]


def _has_structural_importance(text: str, precision_items: list) -> bool:
    """True if text contains language-agnostic signals of importance."""
    if precision_items:
        return True
    if _CODE_PATTERN.search(text):
        return True
    if len(text) > 50:  # lowered from 300 — agent responses are typically 50-200 chars
        return True
    return False


def parse_deadline(text: str, level: str) -> int | None:
    """Parse a deadline timestamp from text."""
    now = int(time.time())
    offsets = {
        "deadline_h": 3600,
        "deadline_d": 86400,
        "deadline_w": 604800,
    }
    m = _DEADLINE_PATTERN.search(text)
    if m:
        return now + 86400
    return now + offsets.get(level, 86400)


def detect_urgency(text: str) -> tuple[str, int | None]:
    """Detect urgency level and optional deadline from text."""
    t = text.lower()
    for level in ("deadline_h", "deadline_d", "deadline_w"):
        if any(sig in t for sig in URGENCY_SIGNALS[level]):
            return level, parse_deadline(text, level)
    if _DEADLINE_PATTERN.search(text):
        return "deadline_d", parse_deadline(text, "deadline_d")
    return "none", None


def detect_principle(text: str) -> bool:
    """True if text contains principle-level signals."""
    t = text.lower()
    return any(sig in t for sig in PRINCIPLE_SIGNALS)


def is_agent_monologue(text: str) -> bool:
    """Return True when text looks like agent internal monologue (noise).

    The text must satisfy ALL of:
      1. Match at least one MONOLOGUE_PATTERNS (case-insensitive).
      2. Be short (< 100 characters after stripping).
      3. Contain no precision items (no matches against PRECISION_PATTERNS).
    """
    stripped = text.strip()
    if len(stripped) >= 100:
        return False
    # Check monologue patterns against lowercased text
    if not any(pat.match(stripped.lower()) for pat in MONOLOGUE_PATTERNS):
        return False
    # Reject if any precision items or code patterns are present
    for pat in PRECISION_PATTERNS.values():
        if re.search(pat, stripped, re.IGNORECASE):
            return False
    if _CODE_PATTERN.search(stripped):
        return False
    return True


def deterministic_filter(text: str) -> dict[str, Any]:
    """Classify text. Structural signals take priority.

    Returns:
        Dict with keys: importance, conflict, precision_items,
                        needs_ner, urgency, deadline_val, discard.
    """
    t = text.lower()

    precision_items: list[dict[str, str]] = []
    for ptype, pat in PRECISION_PATTERNS.items():
        for m in re.findall(pat, text, re.IGNORECASE):
            precision_items.append({"type": ptype, "value": m})

    if detect_principle(text):
        importance = "principle"
    elif any(w in t for w in CRITICAL):
        importance = "critical"
    elif any(w in t for w in IMPORTANT) or _has_structural_importance(text, precision_items):
        importance = "important"
    else:
        importance = "background"

    conflict = any(w in t for w in CONFLICT)
    urgency, deadline_val = detect_urgency(text)

    needs_ner = importance not in ("critical", "principle") or not precision_items

    discard = is_agent_monologue(text)

    return {
        "importance": importance,
        "conflict": conflict,
        "precision_items": precision_items,
        "needs_ner": needs_ner,
        "urgency": urgency,
        "deadline_val": deadline_val,
        "discard": discard,
    }
