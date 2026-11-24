"""Trained five-label silence detector loaded from ``results/detector/model.json``.

``rank`` returns calibrated category-presence probabilities; ``select_questions`` applies
the H3 policy (presence x utility at the calibrated threshold, clamped to 2-3 questions).
The model artifact is JSON: encoder identity, linear-head weights, Platt calibrators, the
utility vector and the calibrated threshold.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import RESULTS_DIR
from .calibration import apply_calibrators
from .constants import (
    ENCODER_NAME,
    LABELS,
    QUESTION_TEMPLATES,
    UTILITY,
)
from .encoder import embed_cached
from .head import LinearHead
from .policy import clamp_selection, utility_scores

#: Maps a list of wishes to an ``(n, dim)`` embedding matrix; tests inject a fake one.
EncoderFn = Callable[[list[str]], np.ndarray]


def model_path(run_id: str = "r1") -> Path:
    """Return the ``model.json`` path for ``run_id``; R1 lives at the detector root."""
    return (
        RESULTS_DIR / "detector" / "model.json"
        if run_id == "r1"
        else (RESULTS_DIR / "detector" / run_id / "model.json")
    )


def default_model_path() -> Path:
    """Return the final detector used when no ``--detector`` path is given.

    This is the detector refit on the smoothed train wishes (``make detector-smoothed``),
    which the paper reports and the R2b pilot used; ``results/detector/`` holds the
    earlier raw-label R1 artifact.
    """
    return RESULTS_DIR / "detector-smoothed" / "model.json"


@dataclass(frozen=True)
class CategoryScore:
    """A category and its (calibrated or utility-weighted) score."""

    category: str
    score: float


@dataclass(frozen=True)
class CategoryQuestion:
    """A selected category with its display label, score and clarification question."""

    category: str
    label: str
    score: float
    question: str


class TrainedDetector:
    """The deployed detector: frozen encoder, linear head, Platt calibrators, H3 policy."""

    name = "trained-silence-detector/r1"

    def __init__(self, model: dict, *, encoder_fn: EncoderFn | None = None) -> None:
        if tuple(model.get("labels") or ()) != tuple(LABELS):
            raise ValueError(f"model labels {model.get('labels')} do not match {LABELS}")
        self.model = model
        self.labels = tuple(model["labels"])
        self.head = LinearHead.from_dict(model["head"])
        self.calibrators = list(model["calibrators"])
        self.encoder = dict(model.get("encoder") or {})
        self.threshold = float(model["threshold"])
        self.utility = {
            label: float((model.get("policy") or {}).get("utility", {}).get(label, UTILITY[label]))
            for label in self.labels
        }
        self._encoder_fn = encoder_fn

    def _encode(self, wishes: list[str]) -> np.ndarray:
        if self._encoder_fn is not None:
            return np.asarray(self._encoder_fn(list(wishes)), dtype=np.float64)
        return np.asarray(embed_cached(list(wishes)), dtype=np.float64)

    def predict_proba(self, wishes: list[str]) -> np.ndarray:
        """Return calibrated category-presence probabilities, one row per wish."""
        raw = self.head.predict_proba(self._encode(list(wishes)))
        return apply_calibrators(raw, self.calibrators)

    def rank(self, wish: str) -> list[CategoryScore]:
        """Return every category by descending calibrated presence probability."""
        probabilities = self.predict_proba([wish])[0]
        scores = [
            CategoryScore(category=label, score=round(float(probabilities[index]), 6))
            for index, label in enumerate(self.labels)
        ]
        return sorted(scores, key=lambda item: (-item.score, self.labels.index(item.category)))

    def utility_scores(self, wish: str) -> list[CategoryScore]:
        """Return every category by descending ``presence * utility`` score."""
        probabilities = self.predict_proba([wish])[0]
        scores = probabilities * np.asarray(
            [self.utility[label] for label in self.labels], dtype=np.float64
        )
        items = [
            CategoryScore(category=label, score=round(float(scores[index]), 6))
            for index, label in enumerate(self.labels)
        ]
        return sorted(items, key=lambda item: (-item.score, self.labels.index(item.category)))

    def select_questions(self, wish: str, *, budget: int = 5) -> list[CategoryQuestion]:
        """Apply the clamped H3 policy and return at most ``budget`` questions, best first."""
        probabilities = self.predict_proba([wish])
        selected = clamp_selection(probabilities, self.threshold)[0]
        scores = utility_scores(probabilities, self.labels)[0]
        chosen = [
            index
            for index in sorted(range(len(self.labels)), key=lambda j: (-scores[j], j))
            if selected[index]
        ][:budget]
        return [
            CategoryQuestion(
                category=self.labels[index],
                label=self.labels[index].replace("_", " "),
                score=round(float(scores[index]), 6),
                question=QUESTION_TEMPLATES[self.labels[index]],
            )
            for index in chosen
        ]


def load_detector(
    path: Path | None = None, *, encoder_fn: EncoderFn | None = None
) -> TrainedDetector:
    """Load a trained detector artifact, checking that it used the expected encoder."""
    path = path or default_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"trained detector artifact missing at {path}; run `make detector-smoothed`"
        )
    model = json.loads(path.read_text(encoding="utf-8"))
    if str(model.get("encoder", {}).get("model", ENCODER_NAME)) != ENCODER_NAME:
        raise ValueError(
            f"detector was trained with {model.get('encoder', {}).get('model')!r}, "
            f"expected {ENCODER_NAME!r}"
        )
    return TrainedDetector(model, encoder_fn=encoder_fn)


__all__ = [
    "CategoryQuestion",
    "CategoryScore",
    "EncoderFn",
    "TrainedDetector",
    "default_model_path",
    "load_detector",
    "model_path",
]
