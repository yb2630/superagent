"""Detection, calibration and selection metrics for the five-label detector."""

from __future__ import annotations

import numpy as np

from .calibration import (
    brier_score,
    expected_calibration_error,
    reliability_table,
)
from .constants import LABELS


def _both_classes(labels: np.ndarray) -> bool:
    values = np.asarray(labels)
    return bool(values.min() < values.max())


def auroc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels, dtype=np.float64)
    if not _both_classes(y):
        return None
    return float(roc_auc_score(y, np.asarray(probabilities, dtype=np.float64)))


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    from sklearn.metrics import average_precision_score

    y = np.asarray(labels, dtype=np.float64)
    if not _both_classes(y):
        return None
    return float(average_precision_score(y, np.asarray(probabilities, dtype=np.float64)))


def log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def per_label_f1(
    labels: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5
) -> dict[str, float]:
    from sklearn.metrics import f1_score

    y = np.asarray(labels)
    predicted = (np.asarray(probabilities, dtype=np.float64) >= threshold).astype(int)
    return {
        label: float(f1_score(y[:, index], predicted[:, index], zero_division=0))
        for index, label in enumerate(LABELS)
    }


def macro_f1(labels: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5) -> float:
    scores = per_label_f1(labels, probabilities, threshold=threshold)
    return float(np.mean(list(scores.values())))


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def calibration_summary(labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10) -> dict:
    per_label = {}
    for index, label in enumerate(LABELS):
        per_label[label] = {
            "brier": round(brier_score(labels[:, index], probabilities[:, index]), 6),
            "ece": round(
                expected_calibration_error(labels[:, index], probabilities[:, index], bins=bins),
                6,
            ),
            "log_loss": round(log_loss(labels[:, index], probabilities[:, index]), 6),
            "reliability": reliability_table(labels[:, index], probabilities[:, index], bins=bins),
        }
    flat_labels = np.asarray(labels).reshape(-1)
    flat_probabilities = np.asarray(probabilities).reshape(-1)
    return {
        "bins": bins,
        "brier": round(brier_score(flat_labels, flat_probabilities), 6),
        "ece": round(expected_calibration_error(flat_labels, flat_probabilities, bins=bins), 6),
        "log_loss": round(log_loss(flat_labels, flat_probabilities), 6),
        "per_label": per_label,
        "reliability_pooled": reliability_table(flat_labels, flat_probabilities, bins=bins),
    }


def detection_summary(
    labels: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5
) -> dict:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    f1_scores = per_label_f1(labels, probabilities, threshold=threshold)
    varying = [label for index, label in enumerate(LABELS) if _both_classes(labels[:, index])]
    per_label = {}
    for index, label in enumerate(LABELS):
        per_label[label] = {
            "positives": int(labels[:, index].sum()),
            "prevalence": round(float(labels[:, index].mean()), 6),
            "auroc": _rounded(auroc(labels[:, index], probabilities[:, index])),
            "average_precision": _rounded(
                average_precision(labels[:, index], probabilities[:, index])
            ),
            "f1": round(f1_scores[label], 6),
        }
    return {
        "threshold": threshold,
        "macro_f1": round(float(np.mean(list(f1_scores.values()))), 6),
        "macro_f1_varying_labels": (
            round(float(np.mean([f1_scores[label] for label in varying])), 6) if varying else None
        ),
        "varying_labels": varying,
        "per_label": per_label,
        "calibration": calibration_summary(labels, probabilities),
    }


def rank_categories(probabilities: np.ndarray) -> list[list[int]]:
    matrix = np.asarray(probabilities, dtype=np.float64)
    return [
        sorted(
            range(matrix.shape[1]),
            key=lambda index: (-matrix[row, index], index),
        )
        for row in range(matrix.shape[0])
    ]


def selection_summary(labels: np.ndarray, selected: np.ndarray) -> dict:
    """Question-policy precision, recall and count distribution on labelled tasks."""
    y = np.asarray(labels, dtype=np.float64)
    chosen = np.asarray(selected, dtype=bool)
    counts = chosen.sum(axis=1)
    true_positive = float((chosen & (y > 0)).sum())
    selected_total = int(chosen.sum())
    positives = int((y > 0).sum())
    values, frequencies = np.unique(counts, return_counts=True)
    return {
        "mean_questions": round(float(counts.mean()), 4),
        "distribution": {
            str(int(value)): int(frequency) for value, frequency in zip(values, frequencies)
        },
        "selected_total": selected_total,
        "true_positives": true_positive,
        "precision": round(true_positive / selected_total, 6) if selected_total else None,
        "recall_of_present_categories": round(true_positive / positives, 6) if positives else None,
        "selection_rate_by_category": {
            label: round(float(chosen[:, index].mean()), 4) for index, label in enumerate(LABELS)
        },
    }


__all__ = [
    "auroc",
    "average_precision",
    "calibration_summary",
    "detection_summary",
    "log_loss",
    "macro_f1",
    "per_label_f1",
    "rank_categories",
    "selection_summary",
]
