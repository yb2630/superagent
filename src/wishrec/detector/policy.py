"""H3 question policy: calibrated presence probability times question utility.

Intent is already in the wish (utility 0).  Signature/types are visible only as an untyped
stub, examples are already represented by the three visible tests, and constraints plus
error behaviour are the categories the agent cannot recover.  The policy ranks
``presence * utility`` and calibrates one threshold so the expected number of questions is
2-3 per task; a per-task clamp guarantees the requested range even when the threshold lands
on a degenerate score.
"""

from __future__ import annotations

import numpy as np

from .calibration import calibrate_threshold
from .constants import (
    LABELS,
    QUESTION_COUNT_RANGE,
    TARGET_MEAN_QUESTIONS,
    UTILITY,
    UTILITY_RATIONALE,
)
from .metrics import rank_categories, selection_summary


def utility_scores(probabilities: np.ndarray, label_names: tuple[str, ...] = LABELS) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    weights = np.asarray([float(UTILITY[label]) for label in label_names], dtype=np.float64)
    return probabilities * weights


def utility_selection(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    scores = utility_scores(probabilities)
    selected = scores >= float(threshold)
    # Intent has zero utility by construction; never spend a question on it.
    intent_index = LABELS.index("intent")
    selected[:, intent_index] = False
    return selected


def clamp_selection(
    probabilities: np.ndarray,
    threshold: float,
    *,
    low: int = QUESTION_COUNT_RANGE[0],
    high: int = QUESTION_COUNT_RANGE[1],
) -> np.ndarray:
    """Apply the threshold, then force each task to ask ``low``-``high`` questions."""
    scores = utility_scores(probabilities)
    selected = utility_selection(probabilities, threshold)
    orders = rank_categories(scores)
    intent_index = LABELS.index("intent")
    for row, order in enumerate(orders):
        chosen = [index for index in order if selected[row, index]]
        if len(chosen) < low:
            for index in order:
                if index == intent_index or index in chosen:
                    continue
                chosen.append(index)
                if len(chosen) >= low:
                    break
        if len(chosen) > high:
            chosen = chosen[:high]
        selected[row, :] = False
        selected[row, chosen] = True
    return selected


def calibrate_utility_policy(
    probabilities: np.ndarray,
    *,
    target: float = TARGET_MEAN_QUESTIONS,
    low: int = QUESTION_COUNT_RANGE[0],
    high: int = QUESTION_COUNT_RANGE[1],
) -> dict:
    scores = utility_scores(probabilities)
    calibration = calibrate_threshold(scores, target=target, low=low, high=high)
    calibration["policy"] = "predicted_presence_x_utility"
    calibration["utility"] = {label: UTILITY[label] for label in LABELS}
    calibration["clamped_to"] = [low, high]
    calibration["score_max"] = round(float(scores.max()), 6)
    return calibration


def selected_questions(probabilities: np.ndarray, threshold: float):
    """Return, per row, the selected category names in descending utility-score order."""
    selected = clamp_selection(probabilities, threshold)
    scores = utility_scores(probabilities)
    orders = rank_categories(scores)
    result = []
    for row, order in enumerate(orders):
        result.append([LABELS[index] for index in order if selected[row, index]])
    return result


def utility_metrics(labels: np.ndarray, selected: np.ndarray) -> dict:
    y = np.asarray(labels, dtype=np.float64)
    chosen = np.asarray(selected, dtype=bool)
    weights = np.asarray([UTILITY[label] for label in LABELS], dtype=np.float64)
    informative = y > 0
    selected_informative = chosen & informative
    selected_total = int(chosen.sum())
    informative_total = int(informative.sum())
    utility_available = float((informative * weights).sum())
    utility_selected = float((selected_informative * weights).sum())
    per_task = (selected_informative * weights).sum(axis=1)
    return {
        "selected_total": selected_total,
        "informative_pairs": informative_total,
        "selected_informative_pairs": int(selected_informative.sum()),
        "precision_informative_pairs": round(float(selected_informative.sum()) / selected_total, 6)
        if selected_total
        else None,
        "recall_informative_pairs": round(float(selected_informative.sum()) / informative_total, 6)
        if informative_total
        else None,
        "utility_available": round(utility_available, 4),
        "utility_available_per_task": round(utility_available / len(y), 4),
        "utility_selected": round(utility_selected, 4),
        "utility_recall": round(utility_selected / utility_available, 6)
        if utility_available
        else None,
        "utility_captured_per_task": round(float(per_task.mean()), 4),
    }


def policy_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict:
    calibration = calibrate_utility_policy(probabilities) if threshold is None else None
    cutoff = float(calibration["threshold"]) if calibration else float(threshold)
    selected = clamp_selection(probabilities, cutoff)
    summary = selection_summary(labels, selected)
    summary.update(utility_metrics(labels, selected))
    summary.update(
        {
            "policy": "predicted_presence_x_utility",
            "threshold": cutoff,
            "utility": {label: UTILITY[label] for label in LABELS},
            "utility_rationale": {label: UTILITY_RATIONALE[label] for label in LABELS},
        }
    )
    if calibration is not None:
        summary["threshold_calibration"] = calibration
        summary["calibrated_mean_questions"] = calibration["mean_questions"]
    return summary


__all__ = [
    "calibrate_utility_policy",
    "clamp_selection",
    "policy_summary",
    "selected_questions",
    "utility_metrics",
    "utility_scores",
    "utility_selection",
]
