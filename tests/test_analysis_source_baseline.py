from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wishrec.analysis import source_baseline as sb
from wishrec.detector.constants import LABELS

COMMITTED = Path("results/analysis/source_baseline.json")


def _rows(sources, varying, encoder_scores, *, seed=7, folds=5):
    """Build prediction rows whose prior column matches the recomputed folds."""
    n = len(sources)
    Y = np.zeros((n, len(LABELS)))
    Y[:, LABELS.index("intent")] = 1
    Y[:, LABELS.index("signature_types")] = 1
    for label, column in varying.items():
        Y[:, LABELS.index(label)] = column
    prior = np.zeros_like(Y)
    for train, test in sb.outer_folds(n, seed=seed, folds=folds):
        prior[test] = Y[train].mean(axis=0)
    rows = []
    for index in range(n):
        scores = [0.99, 0.99, *[encoder_scores[label][index] for label in sb.VARYING_LABELS]]
        rows.append(
            {
                "task_key": f"t/{index}",
                "source": sources[index],
                "labels": {label: int(Y[index, j]) for j, label in enumerate(LABELS)},
                "encoder": scores,
                "encoder_raw": scores,
                "prior": prior[index].round(6).tolist(),
            }
        )
    return rows


def _synthetic(n_per_source=10):
    sources = ["a"] * n_per_source + ["b"] * n_per_source
    rng = np.random.default_rng(0)
    # constraints is fully determined by source; error varies inside source "a" only.
    constraints = np.array([1] * n_per_source + [0] * n_per_source)
    error = np.array([1, 0] * (n_per_source // 2) + [0] * n_per_source)
    examples = np.ones(2 * n_per_source, dtype=int)
    examples[-1] = 0
    varying = {"examples": examples, "constraints": constraints, "error_behaviour": error}
    encoder = {
        "examples": rng.random(2 * n_per_source).tolist(),
        "constraints": (constraints * 0.5 + rng.random(2 * n_per_source) * 0.1).tolist(),
        "error_behaviour": (error * 0.6 + rng.random(2 * n_per_source) * 0.1).tolist(),
    }
    return sources, varying, encoder


def test_source_only_baseline_is_perfect_when_a_label_is_a_function_of_source():
    sources, varying, encoder = _synthetic()
    report = sb.analyse(_rows(sources, varying, encoder), seed=7)
    assert report["folds_verified"] is True
    assert report["labels"]["constraints"]["source_only"]["auroc"] == 1.0
    # Within a source the label is constant, so within-source AUROC is undefined.
    within = report["labels"]["constraints"]["within_source"]
    assert within["a"]["encoder_auroc"] is None
    assert within["b"]["encoder_auroc"] is None
    assert within["a"]["positives"] == 10


def test_within_source_auroc_measures_signal_beyond_source():
    sources, varying, encoder = _synthetic()
    report = sb.analyse(_rows(sources, varying, encoder), seed=7)
    error = report["labels"]["error_behaviour"]
    assert error["within_source"]["a"]["encoder_auroc"] == 1.0
    assert error["within_source"]["b"]["encoder_auroc"] is None  # single class
    assert error["encoder_raw_centred_by_source_auroc"] > 0.9


def test_source_missing_from_a_training_fold_falls_back_to_the_fold_rate():
    sources = ["a"] * 9 + ["rare"]
    Y = np.zeros((10, len(LABELS)))
    Y[:5, 0] = 1
    folds = [(np.arange(9), np.array([9]))]
    oof = sb.source_only_oof(sources, Y, folds)
    assert oof[9, 0] == pytest.approx((5 + 1) / (9 + 2))


def test_centring_by_source_uses_scores_only():
    centred = sb.centred_by_source(np.array([1.0, 3.0, 10.0, 14.0]), ["a", "a", "b", "b"])
    assert centred.tolist() == [-1.0, 1.0, -2.0, 2.0]


def test_mismatched_folds_are_refused():
    sources, varying, encoder = _synthetic()
    rows = _rows(sources, varying, encoder, seed=7)
    with pytest.raises(ValueError, match="do not reproduce the committed prior"):
        sb.analyse(rows, seed=8)


def test_empty_input_is_an_error():
    with pytest.raises(ValueError, match="no prediction rows"):
        sb.analyse([], seed=1)


def test_run_writes_deterministic_outputs(tmp_path):
    sources, varying, encoder = _synthetic()
    predictions = tmp_path / "cv_predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in _rows(sources, varying, encoder, seed=7)),
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"training": {"seed": 7}}), encoding="utf-8")
    sb.run(predictions, metrics, tmp_path / "one")
    sb.run(predictions, metrics, tmp_path / "two")
    for name in ("source_baseline.json", "source_baseline.md"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_committed_analysis_reproduces_from_committed_predictions(tmp_path):
    sb.run(sb.DEFAULT_PREDICTIONS, sb.DEFAULT_METRICS, tmp_path)
    assert (tmp_path / "source_baseline.json").read_bytes() == COMMITTED.read_bytes()
    assert (tmp_path / "source_baseline.md").read_bytes() == COMMITTED.with_suffix(
        ".md"
    ).read_bytes()
