"""Five-fold cross-validation with inner-fold model selection and calibration.

Every method sees exactly the same outer folds.  For the two learned methods, the L2/C
strength and the Platt calibrators are selected/fitted only on inner-fold out-of-fold
predictions of the outer training split; the outer held-out fold is never used for
selection or calibration.  The prior-only baseline is constant.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from .baselines import FitResult, PredictFn, fit_encoder_head, fit_prior, fit_tfidf_logistic
from .calibration import apply_calibrators, fit_calibrators
from .constants import (
    CV_FOLDS,
    ENCODER_L2_CANDIDATES,
    INNER_CALIBRATION_FOLDS,
    LABELS,
    TFIDF_C_CANDIDATES,
)


def _mean_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))


def _inner_select_and_calibrate(
    fit_fn_factory: Callable[[float], Callable[[np.ndarray], FitResult]],
    candidates: Sequence[float],
    Y: np.ndarray,
    train_idx: np.ndarray,
    *,
    n_labels: int,
    seed: int,
) -> tuple[PredictFn, dict[str, Any], PredictFn, dict[str, Any]]:
    """Select a hyperparameter and fit Platt calibrators on inner folds of ``train_idx``.

    ``fit_fn_factory(candidate)`` returns a fit function over row indices.  Each candidate
    is scored by inner out-of-fold Brier; the winner's inner OOF predictions fit the
    calibrators, and the winner is refitted on all of ``train_idx``.  Returns the
    calibrated predictor, its fit info (with an ``inner_selection`` record), the raw
    uncalibrated predictor and the selection record (the winning value and the scores).
    """
    train_idx = np.asarray(train_idx)
    inner = KFold(n_splits=INNER_CALIBRATION_FOLDS, shuffle=True, random_state=seed + 1)
    inner_probabilities = {
        candidate: np.zeros((len(train_idx), n_labels)) for candidate in candidates
    }
    for inner_train, inner_test in inner.split(np.arange(len(train_idx))):
        for candidate in candidates:
            predict, _ = fit_fn_factory(candidate)(train_idx[inner_train])
            inner_probabilities[candidate][inner_test] = predict(train_idx[inner_test])
    scores = {
        candidate: _mean_brier(inner_probabilities[candidate], Y[train_idx])
        for candidate in candidates
    }
    selected = min(candidates, key=lambda candidate: scores[candidate])
    calibrators = fit_calibrators(inner_probabilities[selected], Y[train_idx])
    predict, info = fit_fn_factory(selected)(train_idx)

    def calibrated(test_idx: np.ndarray) -> np.ndarray:
        return apply_calibrators(predict(np.asarray(test_idx)), calibrators)

    calibrated_inner = apply_calibrators(inner_probabilities[selected], calibrators)
    info = dict(info or {})
    info["inner_selection"] = {
        "selected": selected,
        "candidates": list(candidates),
        "inner_brier_raw": {str(candidate): round(score, 6) for candidate, score in scores.items()},
        "inner_brier_calibrated": round(_mean_brier(calibrated_inner, Y[train_idx]), 6),
        "calibrators": calibrators,
    }
    return calibrated, info, predict, {"selected": selected, "scores": scores}


def cross_validate(
    texts: list[str],
    embeddings: np.ndarray,
    Y: np.ndarray,
    *,
    seed: int = 20261121,
    folds: int = CV_FOLDS,
    encoder_l2_candidates: tuple[float, ...] = ENCODER_L2_CANDIDATES,
    tfidf_c_candidates: tuple[float, ...] = TFIDF_C_CANDIDATES,
) -> dict:
    """Run outer K-fold CV for the prior, TF-IDF and encoder-head methods.

    Returns calibrated (``oof``) and raw (``raw_oof``) out-of-fold probability matrices
    per method, plus per-fold selection and calibration records.
    """
    texts = list(texts)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, n_labels = Y.shape
    if n < folds:
        raise ValueError(f"{n} examples cannot fill {folds} folds")

    outer = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = {
        method: np.zeros((n, n_labels), dtype=np.float64)
        for method in ("prior", "tfidf", "encoder")
    }
    raw_oof = {method: np.zeros((n, n_labels), dtype=np.float64) for method in ("tfidf", "encoder")}
    fold_records = []
    for fold_index, (train_idx, test_idx) in enumerate(outer.split(np.arange(n))):
        prior_predict, prior_info = fit_prior(Y, train_idx)
        oof["prior"][test_idx] = prior_predict(test_idx)

        tfidf_predict, tfidf_info, tfidf_raw, tfidf_selection = _inner_select_and_calibrate(
            lambda candidate: lambda idx: fit_tfidf_logistic(texts, Y, idx, C=candidate, seed=seed),
            tfidf_c_candidates,
            Y,
            train_idx,
            n_labels=n_labels,
            seed=seed + fold_index,
        )
        oof["tfidf"][test_idx] = tfidf_predict(test_idx)
        raw_oof["tfidf"][test_idx] = tfidf_raw(test_idx)

        encoder_predict, encoder_info, encoder_raw, encoder_selection = _inner_select_and_calibrate(
            lambda candidate: (
                lambda idx: fit_encoder_head(embeddings, Y, idx, seed=seed, l2=candidate)
            ),
            encoder_l2_candidates,
            Y,
            train_idx,
            n_labels=n_labels,
            seed=seed + fold_index,
        )
        oof["encoder"][test_idx] = encoder_predict(test_idx)
        raw_oof["encoder"][test_idx] = encoder_raw(test_idx)

        fold_records.append(
            {
                "fold": fold_index,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "prior": prior_info,
                "tfidf": {
                    "selected_C": tfidf_selection["selected"],
                    "inner_brier_raw": {
                        str(candidate): round(score, 6)
                        for candidate, score in tfidf_selection["scores"].items()
                    },
                    "vocabulary_size": tfidf_info.get("vocabulary_size"),
                    "train_rows": tfidf_info.get("train_rows"),
                },
                "encoder": {
                    "selected_l2": encoder_selection["selected"],
                    "inner_brier_raw": {
                        str(candidate): round(score, 6)
                        for candidate, score in encoder_selection["scores"].items()
                    },
                    "epochs": encoder_info.get("epochs"),
                    "learning_rate": encoder_info.get("learning_rate"),
                },
                "inner_calibration": {
                    "tfidf": tfidf_info["inner_selection"],
                    "encoder": encoder_info["inner_selection"],
                },
            }
        )
    return {
        "seed": seed,
        "folds": fold_records,
        "oof": oof,
        "raw_oof": raw_oof,
        "labels": list(LABELS),
        "tfidf_c_candidates": list(tfidf_c_candidates),
        "encoder_l2_candidates": list(encoder_l2_candidates),
    }


def select_deployed_l2(
    embeddings: np.ndarray,
    Y: np.ndarray,
    *,
    seed: int = 20261121,
    candidates: tuple[float, ...] = ENCODER_L2_CANDIDATES,
    folds: int = CV_FOLDS,
) -> dict:
    """Select the deployed head's L2 by cross-validated Brier on the train split only."""
    X = np.asarray(embeddings, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed + 100)
    probabilities = {candidate: np.zeros_like(Y, dtype=np.float64) for candidate in candidates}
    for train_idx, test_idx in kfold.split(np.arange(len(X))):
        for candidate in candidates:
            predict, _ = fit_encoder_head(X, Y, train_idx, seed=seed, l2=candidate)
            probabilities[candidate][test_idx] = predict(test_idx)
    scores = {candidate: _mean_brier(probabilities[candidate], Y) for candidate in candidates}
    selected = min(candidates, key=lambda candidate: scores[candidate])
    return {
        "selected": selected,
        "candidates": list(candidates),
        "brier": {str(candidate): round(score, 6) for candidate, score in scores.items()},
    }


__all__ = ["cross_validate", "select_deployed_l2"]
