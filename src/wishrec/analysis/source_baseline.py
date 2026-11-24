"""Source-only baseline for the silence detector.

The three benchmarks differ sharply in which fields they specify (every BigCodeBench task has
constraints; few MBPP+ tasks do), so a predictor that knows only a task's *source* can rank
the varying labels well.  This analysis measures how much of the detector's out-of-fold
ranking that baseline explains, using the committed ``cv_predictions.jsonl``:

* the outer folds are recomputed exactly as :func:`wishrec.detector.cv.cross_validate` draws
  them and verified against the committed prior-baseline column;
* the source-only predictor is the Laplace-smoothed per-source label rate on each outer
  training fold (the same ``(positives + 1) / (n + 2)`` smoothing the head uses for
  constant labels), so it is out-of-fold like the encoder;
* per label it reports source-only and encoder AUROC/AP, encoder AUROC within each source,
  and the AUROC of encoder scores centred on their per-source mean (signal beyond source).

Run with ``python -m wishrec.analysis.source_baseline``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from ..detector.constants import CV_FOLDS, LABELS
from ..detector.metrics import auroc, average_precision
from . import ANALYSIS_DIR, dumps_deterministic, fmt, round_or_none, sha256_file

#: The labels that vary in the train split; intent and signature_types are always present.
VARYING_LABELS = ("examples", "constraints", "error_behaviour")

DEFAULT_PREDICTIONS = Path("results/detector-smoothed/cv_predictions.jsonl")
DEFAULT_METRICS = Path("results/detector-smoothed/metrics.json")
DEFAULT_SEED = 20261121


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Read the per-task out-of-fold prediction rows, in their committed order."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def outer_folds(n: int, *, seed: int, folds: int = CV_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the outer (train, test) index pairs that ``cross_validate`` uses."""
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(train, test) for train, test in splitter.split(np.arange(n))]


def verify_folds(
    Y: np.ndarray, prior: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]]
) -> bool:
    """Check that each test row's committed prior equals its training fold's label rate.

    The prior baseline predicts the training-fold base rate, so matching it row by row
    confirms the recomputed folds are the ones the committed predictions came from.
    """
    for train, test in folds:
        rates = Y[train].mean(axis=0)
        if not np.allclose(prior[test], np.broadcast_to(rates, prior[test].shape), atol=1e-6):
            return False
    return True


def source_only_oof(
    sources: Sequence[str], Y: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Out-of-fold Laplace-smoothed per-source label rates.

    A source absent from a training fold falls back to that fold's overall smoothed rate.
    """
    source_array = np.asarray(sources)
    oof = np.zeros_like(Y, dtype=np.float64)
    for train, test in folds:
        overall = (Y[train].sum(axis=0) + 1.0) / (len(train) + 2.0)
        for row in test:
            in_source = train[source_array[train] == source_array[row]]
            if len(in_source) == 0:
                oof[row] = overall
            else:
                oof[row] = (Y[in_source].sum(axis=0) + 1.0) / (len(in_source) + 2.0)
    return oof


def centred_by_source(scores: np.ndarray, sources: Sequence[str]) -> np.ndarray:
    """Subtract each source's mean score; uses scores only, never labels."""
    source_array = np.asarray(sources)
    centred = np.asarray(scores, dtype=np.float64).copy()
    for source in sorted(set(source_array.tolist())):
        mask = source_array == source
        centred[mask] -= centred[mask].mean()
    return centred


def _label_report(
    label_index: int,
    Y: np.ndarray,
    sources: Sequence[str],
    source_oof: np.ndarray,
    encoder: np.ndarray,
    encoder_raw: np.ndarray,
) -> dict[str, Any]:
    y = Y[:, label_index]
    source_array = np.asarray(sources)
    within: dict[str, dict[str, Any]] = {}
    for source in sorted(set(source_array.tolist())):
        mask = source_array == source
        within[source] = {
            "n": int(mask.sum()),
            "positives": int(y[mask].sum()),
            "encoder_auroc": round_or_none(auroc(y[mask], encoder[mask, label_index])),
            "encoder_raw_auroc": round_or_none(auroc(y[mask], encoder_raw[mask, label_index])),
        }
    return {
        "positives": int(y.sum()),
        "source_only": {
            "auroc": round_or_none(auroc(y, source_oof[:, label_index])),
            "average_precision": round_or_none(average_precision(y, source_oof[:, label_index])),
        },
        "encoder_calibrated": {
            "auroc": round_or_none(auroc(y, encoder[:, label_index])),
            "average_precision": round_or_none(average_precision(y, encoder[:, label_index])),
        },
        "encoder_raw": {
            "auroc": round_or_none(auroc(y, encoder_raw[:, label_index])),
            "average_precision": round_or_none(average_precision(y, encoder_raw[:, label_index])),
        },
        "encoder_raw_centred_by_source_auroc": round_or_none(
            auroc(y, centred_by_source(encoder_raw[:, label_index], sources))
        ),
        "within_source": within,
    }


def analyse(rows: Sequence[dict[str, Any]], *, seed: int, folds: int = CV_FOLDS) -> dict[str, Any]:
    """Compute the source-only baseline report from committed prediction rows."""
    if not rows:
        raise ValueError("no prediction rows")
    Y = np.array([[row["labels"][label] for label in LABELS] for row in rows], dtype=np.float64)
    sources = [str(row["source"]) for row in rows]
    encoder = np.array([row["encoder"] for row in rows], dtype=np.float64)
    encoder_raw = np.array([row["encoder_raw"] for row in rows], dtype=np.float64)
    prior = np.array([row["prior"] for row in rows], dtype=np.float64)

    fold_pairs = outer_folds(len(rows), seed=seed, folds=folds)
    if not verify_folds(Y, prior, fold_pairs):
        raise ValueError(
            "recomputed outer folds do not reproduce the committed prior column; "
            "the seed or fold count does not match the run"
        )
    source_oof = source_only_oof(sources, Y, fold_pairs)
    source_counts = {source: sources.count(source) for source in sorted(set(sources))}
    rates = {
        source: {
            label: round(float(Y[np.asarray(sources) == source, index].mean()), 6)
            for index, label in enumerate(LABELS)
            if label in VARYING_LABELS
        }
        for source in source_counts
    }
    return {
        "description": (
            "Source-only out-of-fold baseline versus the frozen-encoder detector on the 500 "
            "train tasks; folds recomputed and verified against the committed prior column."
        ),
        "seed": seed,
        "folds": folds,
        "folds_verified": True,
        "tasks": len(rows),
        "source_counts": source_counts,
        "label_rate_by_source": rates,
        "labels": {
            label: _label_report(LABELS.index(label), Y, sources, source_oof, encoder, encoder_raw)
            for label in VARYING_LABELS
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as a short Markdown summary."""
    lines = [
        "# Source-only baseline for the silence detector",
        "",
        (
            f"{report['tasks']} train tasks, {report['folds']} outer folds "
            f"(seed {report['seed']}), out-of-fold scores. Source-only = Laplace-smoothed "
            "per-source label rate on each training fold."
        ),
        "",
        (
            "| label | positives | source-only AUROC | encoder AUROC (cal.) | "
            "encoder AUROC (raw) | raw, centred by source | source-only AP | encoder AP (cal.) |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, item in report["labels"].items():
        lines.append(
            f"| {label} | {item['positives']} | {fmt(item['source_only']['auroc'])} | "
            f"{fmt(item['encoder_calibrated']['auroc'])} | {fmt(item['encoder_raw']['auroc'])} | "
            f"{fmt(item['encoder_raw_centred_by_source_auroc'])} | "
            f"{fmt(item['source_only']['average_precision'])} | "
            f"{fmt(item['encoder_calibrated']['average_precision'])} |"
        )
    lines += [
        "",
        "## Encoder AUROC within each source (calibrated / raw; positives of n)",
        "",
        "| label | " + " | ".join(report["source_counts"]) + " |",
        "| --- |" + " --- |" * len(report["source_counts"]),
    ]
    for label, item in report["labels"].items():
        cells = []
        for source in report["source_counts"]:
            within = item["within_source"][source]
            cells.append(
                f"{fmt(within['encoder_auroc'])} / {fmt(within['encoder_raw_auroc'])} "
                f"({within['positives']}/{within['n']})"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Label rate by source",
        "",
        "| source | n | " + " | ".join(VARYING_LABELS) + " |",
        "| --- | --- |" + " --- |" * len(VARYING_LABELS),
    ]
    for source, count in report["source_counts"].items():
        rates = report["label_rate_by_source"][source]
        lines.append(
            f"| {source} | {count} | "
            + " | ".join(fmt(rates[label], 2) for label in VARYING_LABELS)
            + " |"
        )
    lines += [
        "",
        "AUROC is undefined (—) when a source has only one class for a label.",
        "",
    ]
    return "\n".join(lines)


def run(predictions: Path, metrics: Path | None, output_dir: Path) -> dict[str, Any]:
    """Run the analysis and write ``source_baseline.json`` and ``.md`` to ``output_dir``."""
    seed = DEFAULT_SEED
    if metrics is not None and Path(metrics).is_file():
        seed = int(json.loads(Path(metrics).read_text(encoding="utf-8"))["training"]["seed"])
    report = analyse(load_predictions(predictions), seed=seed)
    report["inputs"] = {
        "predictions": {"path": str(predictions), "sha256": sha256_file(predictions)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_baseline.json").write_text(dumps_deterministic(report), encoding="utf-8")
    (output_dir / "source_baseline.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help="metrics.json whose training.seed drew the outer folds",
    )
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args(argv)
    run(args.predictions, args.metrics, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
