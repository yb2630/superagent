from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wishrec.detector.constants import LABELS
from wishrec.detector.trained import load_detector

DETECTOR = "results/detector"


def _metrics():
    return json.loads(open(f"{DETECTOR}/metrics.json", encoding="utf-8").read())


def test_detector_artifacts_exist():
    for name in (
        "model.json",
        "metrics.json",
        "manifest.json",
        "cv_predictions.jsonl",
        "report.md",
    ):
        assert Path(DETECTOR, name).is_file(), name


def test_detector_was_trained_on_exactly_the_500_train_tasks():
    metrics = _metrics()
    assert metrics["data"]["tasks"] == 500
    assert metrics["data"]["split"] == "train"
    assert metrics["data"]["heldout_tasks_used"] == 0
    assert metrics["data"]["split_counts"]["heldout"] == 300
    assert metrics["data"]["label_counts"]["error_behaviour"] == 63
    assert metrics["data"]["label_counts_all_kept"]["error_behaviour"] == 147
    assert (
        metrics["training"]["head"]["l2"]
        in (metrics["training"]["deployed_l2_selection"]["candidates"])
    )


def test_detector_metrics_cover_all_five_labels_and_report_error_behaviour():
    metrics = _metrics()
    for method in ("prior", "tfidf", "encoder"):
        detection = metrics["detection"][method]
        assert set(detection["per_label"]) == set(LABELS)
        assert detection["macro_f1"] is not None
        assert detection["calibration"]["brier"] >= 0.0
        assert detection["calibration"]["ece"] >= 0.0
    encoder = metrics["detection"]["encoder"]
    assert encoder["per_label"]["error_behaviour"]["positives"] == 63
    assert encoder["per_label"]["error_behaviour"]["auroc"] is not None
    assert encoder["per_label"]["constraints"]["auroc"] is not None
    assert encoder["per_label"]["examples"]["auroc"] is not None
    assert metrics["comparison_to_baselines"]["encoder_minus_tfidf"]["macro_f1"] is not None


def test_h3_policy_is_calibrated_to_two_or_three_questions():
    metrics = _metrics()
    policy = metrics["h3"]["policy"]
    assert 2 <= policy["mean_questions"] <= 3
    assert metrics["h3"]["threshold_calibration"]["range"] == [2, 3]
    assert policy["selection_rate_by_category"]["intent"] == 0.0
    assert policy["utility"]["intent"] == 0.0
    assert policy["utility"]["constraints"] == 1.0
    assert policy["utility"]["error_behaviour"] == 1.0
    assert set(policy["distribution"]) <= {"2", "3"}


def test_committed_model_loads_and_selects_two_or_three_questions():
    detector = load_detector(encoder_fn=lambda texts: np.ones((len(texts), 1024)))
    assert detector.labels == LABELS
    assert 0.0 <= detector.threshold <= 1.0
    probabilities = detector.predict_proba(["Add two integers."])
    assert probabilities.shape == (1, 5)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    questions = detector.select_questions("Add two integers.")
    assert 2 <= len(questions) <= 3
    assert all(item.category != "intent" for item in questions)


def test_cv_predictions_include_oof_scores_for_the_500_train_tasks():
    rows = [
        json.loads(line)
        for line in Path(f"{DETECTOR}/cv_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 500
    assert len({row["task_key"] for row in rows}) == 500
    assert all(len(row["encoder"]) == 5 for row in rows)
    assert all(2 <= len(row["selected_categories"]) <= 3 for row in rows)
    assert all(category != "intent" for row in rows for category in row["selected_categories"])


def test_manifest_states_the_heldout_split_is_untouched():
    manifest = json.loads(Path(f"{DETECTOR}/manifest.json").read_text(encoding="utf-8"))
    assert manifest["heldout_untouched"] is True
    assert manifest["data"]["heldout_tasks_used"] == 0
    assert manifest["encoder"]["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert manifest["encoder"]["dimension"] == 1024
    assert manifest["encoder"]["frozen"] is True
    assert manifest["threshold"] >= 0.0


def test_report_mentions_error_behaviour_and_the_raw_l3_limitation():
    report = Path(f"{DETECTOR}/report.md").read_text(encoding="utf-8")
    assert "error_behaviour" in report
    assert "raw first-sentence L3" in report
    assert "H3 question policy" in report


def test_vllm_example_documents_the_transport_and_endpoint():
    example = Path("configs/vllm.env.example").read_text(encoding="utf-8")
    for key in (
        "VLLM_BASE_URL",
        "VLLM_API_KEY",
        "VLLM_MODEL=Qwen/Qwen3-8B-AWQ",
        "VLLM_TIMEOUT",
        "VLLM_TOOL_MODE",
        "VLLM_FRONTIER",
    ):
        assert key in example
    assert "Qwen/Qwen3-8B-AWQ" in example
