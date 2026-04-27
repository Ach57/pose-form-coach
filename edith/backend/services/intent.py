"""
Intent parser
=============
Direct port of edith.py's _handle() / _match_exercise() logic.

Parses a free-text command string (as typed or spoken by the user) and
returns a structured IntentResult so ws.py can act on it without
duplicating any parsing logic.
"""

from __future__ import annotations

from config.config import START_WORDS, STOP_WORDS, STATUS_WORDS, STATUS_WORDS, SHUTDOWN_WORDS

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

# ── gym-form path ─────────────────────────────────────────────────────────────
_GYM_FORM = Path(__file__).resolve().parents[3] / "gym-form"
if str(_GYM_FORM) not in sys.path:
    sys.path.insert(0, str(_GYM_FORM))

from src.control.model_registry import known_exercises, resolve  # noqa: E402

Intent = Literal["start", "stop", "switch", "status", "shutdown", "unknown"]


@dataclass
class IntentResult:
    intent:   Intent
    exercise: Optional[str] = None  # registry key, e.g. "ohp", "squat"


def parse(text: str) -> IntentResult:
    """Parse a natural-language command into an IntentResult.

    Mirrors Edith._handle() from gym-form verbatim.
    """
    words = set(re.sub(r"[^\w\s]", "", text.lower()).split())

    if words & SHUTDOWN_WORDS:
        return IntentResult(intent="shutdown")

    if words & STATUS_WORDS:
        return IntentResult(intent="status")

    if (words & STOP_WORDS) and not (words & START_WORDS):
        return IntentResult(intent="stop")

    if words & START_WORDS:
        exercise = _match_exercise(text)
        if exercise:
            return IntentResult(intent="start", exercise=exercise)
        # "switch" with no exercise name = toggle
        if {"switch", "change"} & words:
            return IntentResult(intent="switch")
        return IntentResult(intent="unknown")

    return IntentResult(intent="unknown")


def _match_exercise(text: str) -> Optional[str]:
    """Return the registry key of the first exercise mentioned in text."""
    text_lower = text.lower()
    for key in known_exercises():
        entry = resolve(key)
        if entry is None:
            continue
        candidates = [key] + [a.lower() for a in entry.aliases]
        if any(c in text_lower for c in candidates):
            return key
    return None
