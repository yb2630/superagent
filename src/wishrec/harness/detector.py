"""Harness-facing detector interface.

H3 and DISCARD depend only on ``select_questions`` returning the categories and question
text.  The trained detector is loaded lazily from ``results/detector/model.json``; a
missing artifact is an explicit error rather than a silent cue-based placeholder, so a
pilot cannot accidentally run H3 with untrained questions.
"""

from __future__ import annotations

from functools import lru_cache

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


def questions_for(chosen: list) -> list[dict]:
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
    return load_detector()


__all__ = [
    "CATEGORIES",
    "LABELS",
    "LABEL_DISPLAY",
    "CategoryQuestion",
    "CategoryScore",
    "TrainedDetector",
    "load_default_detector",
    "load_detector",
    "model_path",
    "questions_for",
]
