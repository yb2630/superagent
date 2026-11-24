from __future__ import annotations

import json

import numpy as np
import pytest

from wishrec.detector.calibration import (
    apply_calibrators,
    brier_score,
    calibrate_threshold,
    expected_calibration_error,
    fit_platt,
    reliability_table,
)
from wishrec.detector.constants import LABELS, QUESTION_COUNT_RANGE, UTILITY
from wishrec.detector.data import kept_label_counts, split_counts, train_dataset
from wishrec.detector.head import fit_linear_head
from wishrec.detector.metrics import auroc, detection_summary, macro_f1
from wishrec.detector.policy import (
    calibrate_utility_policy,
    clamp_selection,
    policy_summary,
    selected_questions,
    utility_scores,
    utility_selection,
)
from wishrec.detector.trained import TrainedDetector, load_detector


def synthetic_record(index: int, split: str = "train") -> dict:
    present = {
        "intent": True,
        "signature_types": True,
        "examples": index % 2 == 0,
        "constraints": index % 3 == 0,
        "error_behaviour": index % 4 == 0,
    }
    fields = {label: f"{label} content {index}" if present[label] else "" for label in LABELS}
    return {
        "key": f"synthetic/{split}/{index}",
        "source": "humanevalplus",
        "source_task_id": str(index),
        "entry_point": "f",
        "split": split,
        "fields": fields,
        "field_presence": present,
        "levels": {
            "L0": "full spec",
            "L1": "l1",
            "L2": "l2",
            "L3": f"Wish number {index}.",
        },
        "stub": "def f():\n    pass\n",
        "visible_asserts": "assert True",
        "visible_asserts_count": 3,
        "visible_test_sha256": "0" * 64,
        "hidden_test_sha256": "0" * 64,
        "canonical_sha256": "0" * 64,
        "requirements": [],
        "meta": {},
    }


def write_synthetic_artifact(path, train=500, heldout=300, extra_train=None):
    records = [synthetic_record(index, "train") for index in range(train)]
    records += [synthetic_record(index, "heldout") for index in range(heldout)]
    if extra_train:
        records.extend(extra_train)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return records


def test_train_split_has_the_expected_counts_and_no_heldout():
    dataset = train_dataset()
    assert dataset["split"] == "train"
    assert len(dataset["records"]) == 500
    assert dataset["label_counts"] == {
        "intent": 500,
        "signature_types": 500,
        "examples": 491,
        "constraints": 293,
        "error_behaviour": 85,
    }
    assert kept_label_counts()["error_behaviour"] == 158
    assert split_counts() == {"train": 500, "evolution": 200, "heldout": 300, "pilot": 100}
    assert all(not key.startswith("synthetic/heldout") for key in dataset["keys"])


def test_train_dataset_refuses_heldout_leakage(tmp_path):
    path = tmp_path / "tasks.jsonl"
    records = [synthetic_record(index, "train") for index in range(500)]
    leaked = synthetic_record(0, "heldout")
    leaked["key"] = records[0]["key"]
    records.append(leaked)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="held-out"):
        train_dataset(path)


def test_head_is_reproducible_and_handles_constant_labels():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 16))
    Y = np.zeros((80, 5))
    Y[:, 0] = 1
    Y[:, 1] = 1
    Y[:, 2] = rng.random(80) < 0.8
    Y[:, 3] = rng.random(80) < 0.5
    Y[:, 4] = rng.random(80) < 0.2
    X[:, 3] += 2.0 * Y[:, 3]
    X[:, 4] += 2.0 * Y[:, 4]
    first = fit_linear_head(X, Y, seed=3, epochs=80, l2=0.1)
    second = fit_linear_head(X, Y, seed=3, epochs=80, l2=0.1)
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.bias, second.bias)
    probabilities = first.predict_proba(X)
    assert probabilities.shape == (80, 5)
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert np.all(probabilities[:, 0] > 0.98)
    assert np.all(probabilities[:, 1] > 0.98)
    assert "intent" in first.hyperparameters["constant_labels"]
    assert "signature_types" in first.hyperparameters["constant_labels"]


def test_platt_calibration_reduces_brier_on_miscalibrated_scores():
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 2, size=400)
    logits = labels * 2.0 - 1.0 + rng.normal(scale=0.8, size=400)
    probabilities = 1.0 / (1.0 + np.exp(-logits * 4.0))  # overconfident
    calibrator = fit_platt(probabilities, labels)
    calibrated = apply_calibrators(probabilities[:, None], [calibrator])[:, 0]
    assert brier_score(labels, calibrated) <= brier_score(labels, probabilities) + 1e-9
    assert 0.0 <= expected_calibration_error(labels, calibrated) <= 1.0
    rows = reliability_table(labels, calibrated, bins=5)
    assert len(rows) == 5
    assert sum(row["count"] for row in rows) == 400


def test_platt_handles_a_constant_label():
    probabilities = np.linspace(0.1, 0.9, 20)
    labels = np.ones(20)
    calibrator = fit_platt(probabilities, labels)
    assert calibrator["constant"] is True
    calibrated = apply_calibrators(probabilities[:, None], [calibrator])[:, 0]
    assert np.allclose(calibrated, calibrated[0])


def test_detection_metrics_on_perfect_scores():
    labels = np.asarray([[1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 1, 0]])
    probabilities = labels.astype(float)
    assert auroc(labels[:, 3], probabilities[:, 3]) == 1.0
    assert macro_f1(labels, probabilities) == 1.0
    summary = detection_summary(labels, probabilities)
    assert summary["macro_f1"] == 1.0
    assert summary["per_label"]["constraints"]["auroc"] == 1.0
    assert summary["calibration"]["brier"] == 0.0
    assert summary["calibration"]["ece"] == 0.0


def test_metrics_report_undefined_auroc_for_a_constant_label():
    labels = np.ones((10, 5))
    probabilities = np.full((10, 5), 0.9)
    summary = detection_summary(labels, probabilities)
    assert summary["per_label"]["intent"]["auroc"] is None
    assert summary["per_label"]["intent"]["f1"] == 1.0
    assert summary["varying_labels"] == []


def test_question_policy_always_asks_two_or_three_and_never_intent():
    rng = np.random.default_rng(5)
    probabilities = rng.uniform(0.01, 0.99, size=(200, 5))
    probabilities[:, 0] = 1.0  # intent present but utility zero
    calibration = calibrate_utility_policy(probabilities)
    selected = clamp_selection(probabilities, calibration["threshold"])
    counts = selected.sum(axis=1)
    assert set(np.unique(counts)) <= set(
        range(QUESTION_COUNT_RANGE[0], QUESTION_COUNT_RANGE[1] + 1)
    )
    assert not selected[:, 0].any()
    assert calibration["within_range"] is True
    assert calibration["utility"]["intent"] == 0.0
    assert calibration["utility"]["constraints"] == 1.0
    assert calibration["utility"]["error_behaviour"] == 1.0


def test_utility_scores_multiply_presence_by_utility_and_threshold_decisions():
    probabilities = np.asarray([[1.0, 0.8, 0.9, 0.6, 0.4]])
    scores = utility_scores(probabilities)[0]
    assert scores[0] == 0.0
    assert scores[1] == pytest.approx(0.4)
    assert scores[2] == pytest.approx(0.18)
    assert scores[3] == pytest.approx(0.6)
    assert scores[4] == pytest.approx(0.4)
    chosen = utility_selection(probabilities, 0.35)[0]
    assert list(chosen) == [False, True, False, True, True]


def test_policy_summary_reports_distribution_and_utility_recall():
    labels = np.asarray([[1, 1, 1, 1, 0], [1, 1, 1, 0, 1], [1, 1, 0, 1, 0], [1, 1, 1, 0, 1]])
    probabilities = labels.astype(float) * 0.8 + 0.1
    summary = policy_summary(labels, probabilities)
    assert summary["policy"] == "predicted_presence_x_utility"
    assert 2 <= summary["mean_questions"] <= 3
    assert 0.0 <= summary["utility_recall"] <= 1.0
    assert summary["selection_rate_by_category"]["intent"] == 0.0
    assert set(summary["distribution"]) <= {"2", "3"}
    calibration = summary["threshold_calibration"]
    assert calibration["range"] == [2, 3]
    assert len(selected_questions(probabilities, summary["threshold"])) == len(labels)


def test_threshold_calibration_finds_a_threshold_in_range():
    scores = np.linspace(0.0, 1.0, 500).reshape(100, 5)
    calibration = calibrate_threshold(scores)
    assert 2 <= calibration["mean_questions"] <= 3
    assert calibration["within_range"] is True


def _trained_model(dimension: int = 8) -> dict:
    head = fit_linear_head(
        np.random.default_rng(0).normal(size=(40, dimension)),
        np.tile([1.0, 1.0, 1.0, 0.0, 0.0], (40, 1)),
        epochs=20,
    )
    return {
        "format_version": 1,
        "name": "test",
        "created_at": "now",
        "labels": list(LABELS),
        "encoder": {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "revision": "x",
            "dimension": dimension,
            "frozen": True,
        },
        "head": head.to_dict(),
        "calibrators": [
            {"slope": 1.0, "intercept": 0.0, "constant": False, "rate": None} for _ in LABELS
        ],
        "threshold": 0.3,
        "policy": {"utility": dict(UTILITY)},
        "training": {},
    }


def test_trained_detector_ranks_and_selects_questions():
    model = _trained_model()
    detector = TrainedDetector(model, encoder_fn=lambda texts: np.ones((len(texts), 8)))
    ranked = detector.rank("do something")
    assert [item.category for item in ranked] == list(LABELS)
    selected = detector.select_questions("do something")
    assert 2 <= len(selected) <= 3
    assert all(item.category != "intent" for item in selected)
    assert all(item.question for item in selected)


def test_trained_detector_artifact_loading(tmp_path):
    model = _trained_model()
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    detector = load_detector(path, encoder_fn=lambda texts: np.ones((len(texts), 8)))
    assert detector.predict_proba(["wish"]).shape == (1, 5)


def test_missing_detector_artifact_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_detector(tmp_path / "missing.json")


def test_trained_detector_rejects_the_wrong_label_set():
    model = _trained_model()
    model["labels"] = list(LABELS[:-1])
    with pytest.raises(ValueError, match="labels"):
        TrainedDetector(model, encoder_fn=lambda texts: np.ones((len(texts), 8)))


def test_synthetic_train_artifact_roundtrip_is_not_leaking_heldout(tmp_path):
    records = write_synthetic_artifact(tmp_path / "tasks.jsonl")
    assert len(records) == 800
    dataset = train_dataset(tmp_path / "tasks.jsonl")
    assert len(dataset["keys"]) == 500
    assert all(record["split"] == "train" for record in dataset["records"])


def test_encoder_loads_the_pinned_revision(monkeypatch):
    """The manifest and cache key claim ENCODER_REVISION, so the load must pin it."""
    import sentence_transformers

    from wishrec.detector import encoder
    from wishrec.detector.constants import ENCODER_NAME, ENCODER_REVISION

    calls: list[tuple[tuple, dict]] = []

    class RecordingModel:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def eval(self):
            return self

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", RecordingModel)
    encoder._build_model.cache_clear()
    try:
        encoder._build_model()
    finally:
        encoder._build_model.cache_clear()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert ENCODER_NAME in args or kwargs.get("model_name_or_path") == ENCODER_NAME
    assert kwargs.get("revision") == ENCODER_REVISION


def test_default_detector_is_the_final_smoothed_detector():
    """Without --detector, H3 must use results/detector-smoothed, not the R1 artifact."""
    from pathlib import Path

    from wishrec.detector.trained import default_model_path
    from wishrec.harness.runner import detector_identity

    final = Path("results/detector-smoothed/model.json").resolve()
    assert default_model_path().resolve() == final
    assert load_detector().model == json.loads(final.read_text(encoding="utf-8"))
    assert Path(detector_identity()["path"]).resolve() == final


def test_missing_detector_message_names_the_reproducing_target(tmp_path):
    with pytest.raises(FileNotFoundError, match="make detector-smoothed"):
        load_detector(tmp_path / "model.json")


def test_report_limitations_are_derived_from_the_metrics():
    """Prose numbers must match the table, not a stale hard-coded run."""
    from pathlib import Path

    from wishrec.detector.train import render_report

    metrics = json.loads(Path("results/detector-smoothed/metrics.json").read_text(encoding="utf-8"))
    report = render_report(metrics)
    error = metrics["detection"]["encoder"]["per_label"]["error_behaviour"]
    prevalence = metrics["data"]["label_prevalence"]["error_behaviour"]
    kept = sum(metrics["data"]["split_counts"].values())

    assert "12.6%" not in report
    assert f"F1 at the default 0.5 threshold is {error['f1']:.3f}" in report
    assert f"{prevalence:.1%} base rate" in report
    assert f"across all {kept:,} kept tasks" in report
    assert f"in the {metrics['data']['tasks']} train tasks" in report
