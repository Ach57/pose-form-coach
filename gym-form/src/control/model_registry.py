"""Registry mapping exercise names to their predictor classes and checkpoints.

Usage
-----
    from src.control.model_registry import resolve, known_exercises

    entry = resolve("overhead press")   # matches alias
    predictor = entry.predictor_cls.from_checkpoint(entry.checkpoint, ...)

    # Register a new exercise at runtime
    from src.control.model_registry import register
    register("deadlift", DeadliftPredictor, "checkpoints/deadlift/best.pt",
             aliases=["dead lift", "dl"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from src.realtime.predictor import BasePredictor, OHPPredictor, SquatPredictor


@dataclass
class ExerciseEntry:
    predictor_cls: Type[BasePredictor]
    checkpoint: str
    display_name: str
    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default registry — checkpoint paths can be overridden via register()
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, ExerciseEntry] = {
    "ohp": ExerciseEntry(
        predictor_cls=OHPPredictor,
        checkpoint="checkpoints/ohp/best.pt", 
        display_name="Overhead Press",
        aliases=["overhead press", "overhead", "shoulder press", "ohp"],
    ),
    "squat": ExerciseEntry(
        predictor_cls=SquatPredictor,
        checkpoint="checkpoints/squat/best.pt",
        display_name="Squat",
        aliases=["squats", "back squat", "back squats"],
    ),
}


def resolve(name: str) -> ExerciseEntry | None:
    """Return the *ExerciseEntry* for *name*, checking key and all aliases.

    Returns *None* if *name* is not recognized.
    """
    key = name.lower().strip()
    if key in _REGISTRY:
        return _REGISTRY[key]
    for entry in _REGISTRY.values():
        if key in [a.lower() for a in entry.aliases]:
            return entry
    return None


def known_exercises() -> list[str]:
    """Return sorted list of registry keys."""
    return sorted(_REGISTRY.keys())


def register(
    key: str,
    predictor_cls: Type[BasePredictor],
    checkpoint: str,
    display_name: str = "",
    aliases: list[str] | None = None,
) -> None:
    """Add or overwrite an exercise entry in the registry at runtime."""
    _REGISTRY[key] = ExerciseEntry(
        predictor_cls=predictor_cls,
        checkpoint=checkpoint,
        display_name=display_name or key.title(),
        aliases=aliases or [],
    )
