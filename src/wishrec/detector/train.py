"""Train, calibrate and evaluate the five-label silence detector.

The detector is fitted on the 500 ``train`` tasks only.  Out-of-fold metrics use five outer
folds with inner-fold L2 selection and Platt calibration; the deployed head is then fitted
on all 500 with the L2 chosen by a separate 5-fold selection over the same train split.  The
300 held-out tasks and the 100 pilot tasks never enter this script.

Artifacts written under ``results/detector/``:
``model.json``, ``metrics.json``, ``manifest.json``, ``cv_predictions.jsonl`` and
``report.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RESULTS_DIR
from ..harness.config import (
    atomic_write_json,
    atomic_write_text,
    environment_info,
    git_info,
    utc_now_iso,
    write_jsonl,
)
from .calibration import apply_calibrators, fit_calibrators
from .constants import (
    CV_FOLDS,
    DETECTOR_SEED,
    ENCODER_NAME,
    ENCODER_REVISION,
    HEAD_EPOCHS,
    HEAD_LEARNING_RATE,
    LABELS,
    QUESTION_COUNT_RANGE,
    TARGET_MEAN_QUESTIONS,
    UTILITY,
    UTILITY_RATIONALE,
)
from .cv import cross_validate, select_deployed_l2
from .data import kept_label_counts, label_counts, split_counts, train_dataset
from .encoder import embed_cached, encoder_info
from .head import LinearHead, fit_linear_head
from .metrics import detection_summary
from .policy import calibrate_utility_policy, policy_summary, selected_questions

DETECTOR_DIR = RESULTS_DIR / "detector"


def build_model_dict(
    head: LinearHead,
    calibrators: list[dict],
    *,
    threshold: float,
    threshold_calibration: dict,
    data: dict,
    training: dict,
) -> dict:
    """Assemble the ``model.json`` artifact loaded by :func:`load_detector`."""
    return {
        "format_version": 1,
        "name": f"five-label-silence-detector/{training.get('run_id', 'r1')}",
        "created_at": utc_now_iso(),
        "labels": list(head.labels),
        "encoder": {
            **encoder_info(),
            "model": ENCODER_NAME,
            "revision": ENCODER_REVISION,
        },
        "head": head.to_dict(),
        "calibrators": calibrators,
        "threshold": float(threshold),
        "policy": {
            "name": "predicted_presence_x_utility",
            "utility": {label: UTILITY[label] for label in LABELS},
            "utility_rationale": {label: UTILITY_RATIONALE[label] for label in LABELS},
            "target_mean_questions": TARGET_MEAN_QUESTIONS,
            "question_count_range": list(QUESTION_COUNT_RANGE),
            "threshold_calibration": {
                key: value for key, value in threshold_calibration.items() if key != "grid"
            },
        },
        "data": {
            "split": "train",
            "tasks": data["artifact"]["count"],
            "tasks_sha256": data["artifact"]["tasks_sha256"],
            "label_counts": data["label_counts"],
        },
        "training": training,
    }


def _comparison(encoder: dict, baseline: dict) -> dict:
    """Return encoder-minus-baseline differences for the headline metrics."""
    result: dict[str, Any] = {}
    for metric in ("macro_f1", "macro_f1_varying_labels"):
        left = encoder.get(metric)
        right = baseline.get(metric)
        result[metric] = (
            None if left is None or right is None else round(float(left) - float(right), 6)
        )
    for metric in ("brier", "ece", "log_loss"):
        left = encoder["calibration"].get(metric)
        right = baseline["calibration"].get(metric)
        result[metric] = (
            None if left is None or right is None else round(float(left) - float(right), 6)
        )
    result["auroc"] = {}
    for label in LABELS:
        left = encoder["per_label"][label]["auroc"]
        right = baseline["per_label"][label]["auroc"]
        result["auroc"][label] = (
            None if left is None or right is None else round(float(left) - float(right), 6)
        )
    return result


def train_detector(
    *,
    tasks_path: Path | None = None,
    output_dir: Path | None = None,
    embeddings: np.ndarray | None = None,
    folds: int = CV_FOLDS,
    seed: int = DETECTOR_SEED,
    smoothing_path: Path | None = None,
    run_id: str = "r1",
) -> dict:
    """Run CV, fit and calibrate the deployed detector and write its artifacts.

    ``embeddings`` bypasses the encoder (used by tests).  Returns the metrics, the model
    dict and the output directory.
    """
    data = train_dataset(tasks_path, smoothing_path=smoothing_path)
    Y = np.asarray(data["labels"], dtype=np.float64)
    texts = data["texts"]
    X = (
        np.asarray(embeddings, dtype=np.float64)
        if embeddings is not None
        else np.asarray(embed_cached(texts), dtype=np.float64)
    )
    if X.shape != (len(texts), 1024):
        raise ValueError(f"embeddings have shape {X.shape}, expected ({len(texts)}, 1024)")

    cv_result = cross_validate(texts, X, Y, seed=seed, folds=folds)
    deployed_selection = select_deployed_l2(X, Y, seed=seed, folds=folds)
    head = fit_linear_head(
        X,
        Y,
        seed=seed,
        epochs=HEAD_EPOCHS,
        learning_rate=HEAD_LEARNING_RATE,
        l2=deployed_selection["selected"],
    )
    calibrators = fit_calibrators(cv_result["raw_oof"]["encoder"], Y)
    deployed_oof = apply_calibrators(cv_result["raw_oof"]["encoder"], calibrators)
    threshold_calibration = calibrate_utility_policy(deployed_oof)
    threshold = float(threshold_calibration["threshold"])

    detection = {
        method: detection_summary(Y, cv_result["oof"][method])
        for method in ("prior", "tfidf", "encoder")
    }
    raw_encoder = detection_summary(Y, cv_result["raw_oof"]["encoder"])
    h3 = {
        "policy": policy_summary(Y, deployed_oof, threshold=threshold),
        "threshold_calibration": {
            key: value for key, value in threshold_calibration.items() if key != "grid"
        },
        "deployed_style_oof": True,
        "selected_categories": {
            "count": int(sum(len(items) for items in selected_questions(deployed_oof, threshold))),
            "examples": [
                {
                    "task_key": data["keys"][index],
                    "categories": selected_questions(deployed_oof[index : index + 1], threshold)[0],
                }
                for index in range(min(5, len(data["keys"])))
            ],
        },
    }
    smoothing = data.get("smoothing")
    run_label = run_id.upper()
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "git": git_info(),
        "environment": environment_info(),
        "data": {
            "split": "train",
            "tasks": len(texts),
            "tasks_sha256": data["artifact"]["tasks_sha256"],
            "split_counts": split_counts(tasks_path),
            "label_counts": label_counts(Y),
            "label_counts_all_kept": kept_label_counts(tasks_path),
            "label_counts_total_all_kept": int(sum(kept_label_counts(tasks_path).values())),
            "label_prevalence": data["label_prevalence"],
            "sources": {
                source: data["sources"].count(source) for source in sorted(set(data["sources"]))
            },
            "heldout_tasks_used": 0,
            "smoothing": smoothing,
        },
        "encoder": {
            "model": ENCODER_NAME,
            "revision": ENCODER_REVISION,
            "dimension": 1024,
            "frozen": True,
            "device": "cpu",
        },
        "training": {
            "seed": seed,
            "folds": folds,
            "l3": (
                "Qwen3-8B smoothed wishes (R2 artifact)"
                if smoothing
                else "raw first-sentence L3; R2 smoothing not applied"
            ),
            "smoothing": smoothing,
            "head": {
                "epochs": HEAD_EPOCHS,
                "learning_rate": HEAD_LEARNING_RATE,
                "l2": deployed_selection["selected"],
                "loss": "BCEWithLogits + 0.5*l2*||W||^2",
                "optimizer": "AdamW(weight_decay=0)",
            },
            "deployed_l2_selection": deployed_selection,
            "cv": cv_result["folds"],
        },
        "detection": detection,
        "raw_encoder_oof": raw_encoder,
        "comparison_to_baselines": {
            "encoder_minus_prior": _comparison(detection["encoder"], detection["prior"]),
            "encoder_minus_tfidf": _comparison(detection["encoder"], detection["tfidf"]),
        },
        "h3": h3,
    }
    output = Path(output_dir) if output_dir else DETECTOR_DIR
    output.mkdir(parents=True, exist_ok=True)
    l3_training = {
        "run_id": run_id,
        "split": "train",
        "seed": seed,
        "folds": folds,
        "l2": deployed_selection["selected"],
        "epochs": HEAD_EPOCHS,
        "learning_rate": HEAD_LEARNING_RATE,
        "l3": (
            "Qwen3-8B smoothed wishes (R2 artifact)"
            if smoothing
            else "raw first-sentence L3; R2 smoothing not applied"
        ),
        "smoothing": smoothing,
    }
    model = build_model_dict(
        head,
        calibrators,
        threshold=threshold,
        threshold_calibration=threshold_calibration,
        data=data,
        training=l3_training,
    )
    (output / "model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics["artifacts"] = {
        "model_sha256": _sha256_file(output / "model.json"),
    }
    metrics["report"] = render_report(metrics, run_label=run_label)
    atomic_write_json(output / "metrics.json", metrics)
    write_jsonl(
        output / "cv_predictions.jsonl",
        _cv_rows(data, cv_result, deployed_oof, threshold),
    )
    atomic_write_text(output / "report.md", metrics["report"])
    command = "PYTHONPATH=src uv run python -m wishrec.detector.train"
    if smoothing_path is not None:
        command += f" --run-id {run_id} --smoothed-wishes {smoothing_path}"
    manifest = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "git": git_info(),
        "environment": environment_info(),
        "command": command,
        "data": metrics["data"],
        "encoder": metrics["encoder"],
        "training": metrics["training"],
        "smoothing": smoothing,
        "threshold": threshold,
        "artifacts": {
            "model": "model.json",
            "metrics": "metrics.json",
            "cv_predictions": "cv_predictions.jsonl",
            "report": "report.md",
            "model_sha256": metrics["artifacts"]["model_sha256"],
        },
        "heldout_untouched": True,
    }
    atomic_write_json(output / "manifest.json", manifest)
    return {"metrics": metrics, "model": model, "output_dir": str(output)}


def _cv_rows(
    data: dict,
    cv_result: dict,
    deployed_oof: np.ndarray,
    threshold: float,
) -> list[dict]:
    selected = selected_questions(deployed_oof, threshold)
    rows = []
    for index, key in enumerate(data["keys"]):
        rows.append(
            {
                "task_key": key,
                "source": data["sources"][index],
                "labels": {
                    label: int(data["labels"][index, label_index])
                    for label_index, label in enumerate(LABELS)
                },
                "prior": [round(float(value), 6) for value in cv_result["oof"]["prior"][index]],
                "tfidf": [round(float(value), 6) for value in cv_result["oof"]["tfidf"][index]],
                "encoder": [round(float(value), 6) for value in cv_result["oof"]["encoder"][index]],
                "encoder_raw": [
                    round(float(value), 6) for value in cv_result["raw_oof"]["encoder"][index]
                ],
                "deployed_calibrated": [round(float(value), 6) for value in deployed_oof[index]],
                "selected_categories": selected[index],
            }
        )
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_report(metrics: dict, *, run_label: str | None = None) -> str:
    """Human-readable detector report with every headline number traceable to JSON."""
    detection = metrics["detection"]
    header = run_label or str(metrics.get("run_id", "r1")).upper()
    lines = [
        f"# {header} five-label silence detector",
        "",
        f"- Created: {metrics['created_at']}",
        f"- Git commit: `{metrics['git'].get('commit')}`"
        + (" (dirty)" if metrics["git"].get("dirty") else ""),
        (
            f"- Data: {metrics['data']['tasks']} train tasks, split "
            f"{metrics['data']['split_counts']}; held-out used: "
            f"{metrics['data']['heldout_tasks_used']}."
        ),
        (
            f"- Encoder: frozen `{metrics['encoder']['model']}` "
            f"(revision `{metrics['encoder']['revision']}`), 1024-d, CPU."
        ),
        (
            f"- Head: linear 1024->5, BCE + L2, {metrics['training']['head']['epochs']} "
            f"full-batch AdamW steps, seed {metrics['training']['seed']}; deployed L2="
            f"{metrics['training']['head']['l2']}."
        ),
        "",
        "Label: 1 when the category has content in the full L0 specification, i.e. the wish",
        (
            "omits information the stakeholder can supply. `error_behaviour` has "
            f"{metrics['data']['label_counts']['error_behaviour']} positives in the 500 train "
            f"tasks ({metrics['data']['label_counts_all_kept']['error_behaviour']} across all "
            "1,100 kept tasks); its metrics are reported, not declared undefined."
        ),
        "",
        "## Out-of-fold detection (5 folds, inner-fold L2/Platt calibration)",
        "",
        (
            "| method | macro-F1 | macro-F1 (varying labels) | AUROC examples | "
            "AUROC constraints | AUROC error | Brier | ECE |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method in ("prior", "tfidf", "encoder"):
        item = detection[method]
        lines.append(
            (
                "| {method} | {macro} | {macro_varying} | {ex} | {co} | {er} | {brier} | {ece} |"
            ).format(
                method=method,
                macro=_fmt(item["macro_f1"]),
                macro_varying=_fmt(item["macro_f1_varying_labels"]),
                ex=_fmt(item["per_label"]["examples"]["auroc"]),
                co=_fmt(item["per_label"]["constraints"]["auroc"]),
                er=_fmt(item["per_label"]["error_behaviour"]["auroc"]),
                brier=_fmt(item["calibration"]["brier"], 4),
                ece=_fmt(item["calibration"]["ece"], 4),
            )
        )
    lines.extend(
        [
            "",
            "## Per-category AUROC and Brier (encoder, calibrated OOF)",
            "",
            "| category | positives | prevalence | AUROC | AP | F1@0.5 | Brier | ECE |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for label in LABELS:
        item = detection["encoder"]["per_label"][label]
        calibration = detection["encoder"]["calibration"]["per_label"][label]
        lines.append(
            f"| {label} | {item['positives']} | {_fmt(item['prevalence'], 4)} | "
            f"{_fmt(item['auroc'])} | {_fmt(item['average_precision'])} | "
            f"{_fmt(item['f1'])} | {_fmt(calibration['brier'], 4)} | "
            f"{_fmt(calibration['ece'], 4)} |"
        )
    h3 = metrics["h3"]
    policy = h3["policy"]
    lines.extend(
        [
            "",
            "## H3 question policy",
            "",
            "Score = calibrated presence probability x utility; one global threshold is",
            "calibrated on deployed-style OOF scores and each task is clamped to 2-3 questions.",
            "",
            f"- Utility: {policy['utility']}",
            (
                f"- Threshold: {_fmt(policy['threshold'], 4)}; mean questions "
                f"{_fmt(policy['mean_questions'], 2)}; distribution {policy['distribution']}."
            ),
            (
                f"- Informative-pair precision {_fmt(policy.get('precision_informative_pairs'))}, "
                f"recall {_fmt(policy.get('recall_informative_pairs'))}, utility recall "
                f"{_fmt(policy.get('utility_recall'))}."
            ),
            "- Selection rate by category: "
            + ", ".join(
                f"{label} {rate:.2f}"
                for label, rate in policy["selection_rate_by_category"].items()
            ),
            "",
            f"## {header} limitations",
            "",
            "- The encoder is frozen and the head is a single linear layer; there is no",
            "  fine-tuning. The reported OOF numbers are the detector evidence, not a pilot",
            "  result.",
            "- The `error_behaviour` F1 at the default 0.5 threshold is 0 because its",
            "  calibrated probability stays near the 12.6% base rate; AUROC and average",
            "  precision are the informative ranking metrics for that label.",
            "- Platt calibration is fitted on nested out-of-fold predictions of the train split;",
            "  the 300 held-out and 100 pilot tasks are untouched.",
            "",
        ]
    )
    if metrics["data"].get("smoothing"):
        lines.extend(
            [
                "- This artifact is the R2 refit: the 500 train wishes were smoothed by the",
                "  generator endpoint and the frozen encoder/training recipe are unchanged.",
            ]
        )
    else:
        lines.extend(
            [
                "- R1 trains on the raw first-sentence L3. R2 will smooth the 500 train wishes",
                "  with the generator endpoint and refit the detector before the pilot.",
            ]
        )
    return "\n".join(lines)


def _fmt(value: float | None, digits: int = 3) -> str:
    """Format a metric to ``digits`` decimals, or ``n/a`` when it is undefined."""
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: train the detector and print its report."""
    parser = argparse.ArgumentParser(
        description="Train the five-label silence detector (raw or R2-smoothed L3)"
    )
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--smoothed-wishes",
        type=Path,
        default=None,
        metavar="WISHES_JSONL",
        help="R2 smoothed-wish artifact; without it the raw R0 L3 is used",
    )
    parser.add_argument("--run-id", default="r1", help="artifact label, e.g. r2")
    parser.add_argument("--folds", type=int, default=CV_FOLDS)
    parser.add_argument("--seed", type=int, default=DETECTOR_SEED)
    args = parser.parse_args(argv)
    result = train_detector(
        tasks_path=args.tasks,
        output_dir=args.output,
        folds=args.folds,
        seed=args.seed,
        smoothing_path=args.smoothed_wishes,
        run_id=args.run_id,
    )
    print(result["metrics"]["report"])
    print(f"wrote artifacts to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DETECTOR_DIR", "main", "render_report", "train_detector"]
