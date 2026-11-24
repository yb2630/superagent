"""Platt calibration and question-count threshold calibration.

The detector is calibrated per label with Platt scaling fitted on inner cross-validation
predictions of each outer training fold.  A separate threshold turns calibrated
``presence * utility`` scores into the H3 question policy, targeting 2-3 questions per
wish.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from .constants import LABELS, TARGET_MEAN_QUESTIONS

EPS = 1e-6


def logit(probabilities: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)))


def _logit_scalar(probability: float) -> float:
    probability = min(max(float(probability), EPS), 1.0 - EPS)
    return math.log(probability / (1.0 - probability))


def fit_platt(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Fit ``sigmoid(a * logit(p) + b)`` with smoothed targets (Platt, 1999)."""
    y = np.asarray(labels, dtype=np.float64)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        rate = (positives + 1.0) / (len(y) + 2.0)
        return {
            "slope": 0.0,
            "intercept": _logit_scalar(rate),
            "constant": True,
            "rate": rate,
        }

    z = logit(probabilities)
    target_pos = (positives + 1.0) / (positives + 2.0)
    target_neg = 1.0 / (negatives + 2.0)
    targets = np.where(y > 0, target_pos, target_neg)

    def objective(params: np.ndarray) -> float:
        slope, intercept = float(params[0]), float(params[1])
        p = np.clip(sigmoid(slope * z + intercept), EPS, 1.0 - EPS)
        return float(
            -np.mean(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p))
            + 1e-6 * (slope * slope + intercept * intercept)
        )

    result = minimize(
        objective,
        x0=np.array([1.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0), (-20.0, 20.0)],
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return {"slope": 1.0, "intercept": 0.0, "constant": False, "rate": None}
    return {
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "constant": False,
        "rate": None,
    }


def apply_platt(probabilities: np.ndarray, parameters: dict) -> np.ndarray:
    slope = float(parameters.get("slope", 1.0))
    intercept = float(parameters.get("intercept", 0.0))
    if parameters.get("constant"):
        return np.full_like(np.asarray(probabilities, dtype=np.float64), sigmoid(intercept))
    return sigmoid(slope * logit(probabilities) + intercept)


def fit_calibrators(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    label_names: tuple[str, ...] = LABELS,
) -> list[dict]:
    return [
        fit_platt(probabilities[:, index], labels[:, index]) for index in range(len(label_names))
    ]


def apply_calibrators(probabilities: np.ndarray, calibrators: list[dict]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    calibrated = np.zeros_like(probabilities)
    for index, parameters in enumerate(calibrators):
        calibrated[:, index] = apply_platt(probabilities[:, index], parameters)
    return calibrated


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def reliability_table(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> list[dict]:
    """Equal-width reliability rows, including empty bins."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        low, high = float(edges[index]), float(edges[index + 1])
        if index == bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        count = int(mask.sum())
        row = {
            "bin": index + 1,
            "range": [round(low, 6), round(high, 6)],
            "count": count,
            "mean_predicted": round(float(p[mask].mean()), 6) if count else None,
            "observed_rate": round(float(y[mask].mean()), 6) if count else None,
        }
        if count:
            row["gap"] = round(row["observed_rate"] - row["mean_predicted"], 6)
        rows.append(row)
    return rows


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    rows = reliability_table(labels, probabilities, bins=bins)
    total = sum(row["count"] for row in rows)
    if total == 0:
        return 0.0
    return float(
        sum(
            row["count"] * abs(row["observed_rate"] - row["mean_predicted"])
            for row in rows
            if row["count"]
        )
        / total
    )


def selection_counts(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=np.float64) >= float(threshold)).sum(axis=1)


def question_count_distribution(scores: np.ndarray, threshold: float) -> dict:
    counts = selection_counts(scores, threshold)
    values, frequencies = np.unique(counts, return_counts=True)
    return {str(int(value)): int(frequency) for value, frequency in zip(values, frequencies)}


def calibrate_threshold(
    scores: np.ndarray,
    *,
    target: float = TARGET_MEAN_QUESTIONS,
    low: int = 2,
    high: int = 3,
) -> dict:
    """Pick one global threshold whose mean selection count is in ``[low, high]``."""
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("scores must be (n, n_categories)")
    unique = np.unique(matrix)
    candidates = {0.0, 1.0}
    candidates.update(float(value) for value in unique)
    if len(unique) > 1:
        candidates.update(
            float((unique[index] + unique[index + 1]) / 2.0) for index in range(len(unique) - 1)
        )
    rows = [
        {
            "threshold": threshold,
            "mean_questions": float(selection_counts(matrix, threshold).mean()),
        }
        for threshold in sorted(candidates)
    ]
    within = [row for row in rows if low <= row["mean_questions"] <= high]
    pool = within or rows
    best = min(
        pool,
        key=lambda row: (
            abs(row["mean_questions"] - target),
            row["mean_questions"],
            row["threshold"],
        ),
    )
    threshold = round(float(best["threshold"]), 6)
    mean_questions = round(float(selection_counts(matrix, threshold).mean()), 4)
    return {
        "threshold": threshold,
        "mean_questions": mean_questions,
        "target_mean_questions": target,
        "range": [low, high],
        "within_range": bool(low <= mean_questions <= high),
        "distribution": question_count_distribution(matrix, threshold),
        "grid": [
            {
                "threshold": round(row["threshold"], 6),
                "mean_questions": round(row["mean_questions"], 4),
            }
            for row in rows
        ],
    }


__all__ = [
    "EPS",
    "apply_calibrators",
    "apply_platt",
    "brier_score",
    "calibrate_threshold",
    "expected_calibration_error",
    "fit_calibrators",
    "fit_platt",
    "logit",
    "question_count_distribution",
    "reliability_table",
    "selection_counts",
    "sigmoid",
]
