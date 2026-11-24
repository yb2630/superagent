"""Harness-facing detector interface.

H3 and DISCARD depend only on ``select_questions`` returning the categories and question
text.  The trained detector is loaded lazily from ``results/detector/model.json``; a
missing artifact is an explicit error rather than a silent cue-based placeholder, so a
pilot cannot accidentally run H3 with untrained questions.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from ..detector.constants import (
    LABEL_DISPLAY,
    LABELS,
    QUESTION_TEMPLATES,
)
from ..detector.trained import (
    CategoryQuestion,
    CategoryScore,
    TrainedDetector,
    load_detector,
    model_path,
)

CATEGORIES = LABELS


class QuestionSelector(Protocol):
    """Selects clarification categories for a wish (the trained detector or a test fake)."""

    def select_questions(self, wish: str) -> Sequence[CategoryQuestion | CategoryScore]:
        """Return the categories to ask about, best first."""
        ...


def questions_for(chosen: Sequence[CategoryQuestion | CategoryScore]) -> list[dict]:
    """Expand selected categories into the records the loop and prompts consume."""
    records = []
    for item in chosen:
        if isinstance(item, CategoryQuestion):
            records.append(
                {
                    "category": item.category,
                    "label": LABEL_DISPLAY.get(item.category, item.label),
                    "score": item.score,
                    "question": item.question,
                }
            )
            continue
        category = item.category
        records.append(
            {
                "category": category,
                "label": LABEL_DISPLAY.get(category, category),
                "score": float(item.score),
                "question": QUESTION_TEMPLATES[category],
            }
        )
    return records


@lru_cache(maxsize=1)
def load_default_detector() -> TrainedDetector:
    """Load the committed trained detector once per process."""
    return load_detector()


__all__ = [
    "CATEGORIES",
    "LABELS",
    "LABEL_DISPLAY",
    "CategoryQuestion",
    "CategoryScore",
    "QuestionSelector",
    "TrainedDetector",
    "load_default_detector",
    "load_detector",
    "model_path",
    "questions_for",
]
