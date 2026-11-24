"""Load the detector's training split from the committed R0 artifact.

Only ``split == "train"`` records are returned.  The 200 evolution, 300 held-out and 100
pilot tasks are never used for training, calibration, threshold selection or model
selection; :func:`train_dataset` refuses to proceed if a held-out key appears.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import LEVELS_DIR
from ..harness.config import read_jsonl
from ..smoothing import SmoothingArtifact, load_smoothing
from .constants import LABELS

TASKS_PATH = LEVELS_DIR / "tasks.jsonl"


def load_records(*, split: str = "train", tasks_path: Path | None = None) -> list[dict]:
    """Return the level records of one ``split``, sorted by task key."""
    records = read_jsonl(tasks_path or TASKS_PATH)
    selected = [record for record in records if record["split"] == split]
    selected.sort(key=lambda record: record["key"])
    return selected


def label_matrix(records: list[dict]) -> np.ndarray:
    """Return the ``(n_records, n_labels)`` 0/1 matrix of L0 field presence."""
    return np.asarray(
        [[float(bool(record["field_presence"][label])) for label in LABELS] for record in records],
        dtype=np.float64,
    )


def wishes(records: list[dict]) -> list[str]:
    """Return each record's raw L3 wish text."""
    return [str(record["levels"]["L3"]) for record in records]


def label_counts(Y: np.ndarray) -> dict[str, int]:
    """Return the number of positive rows per label."""
    matrix = np.asarray(Y)
    return {label: int(matrix[:, index].sum()) for index, label in enumerate(LABELS)}


def train_dataset(
    tasks_path: Path | None = None,
    smoothing_path: Path | SmoothingArtifact | None = None,
) -> dict:
    """Load the 500 train records, optionally with smoothed L3 wishes.

    ``smoothing_path`` points at the R2 ``wishes.jsonl`` artifact.  The committed R0
    records are not modified; the overlay is applied in memory and the artifact SHA-256
    is recorded in the detector manifest.
    """
    records = load_records(split="train", tasks_path=tasks_path)
    if len(records) != 500:
        raise ValueError(f"expected the 500 detector-train tasks, found {len(records)}")
    all_records = read_jsonl(tasks_path or TASKS_PATH)
    heldout = {record["key"] for record in all_records if record["split"] == "heldout"}
    leakage = heldout & {record["key"] for record in records}
    if leakage:
        raise RuntimeError(f"held-out tasks in the detector train split: {sorted(leakage)}")
    Y = label_matrix(records)
    texts = wishes(records)
    smoothing: dict | None = None
    if smoothing_path is not None:
        artifact = (
            smoothing_path
            if isinstance(smoothing_path, SmoothingArtifact)
            else load_smoothing(smoothing_path)
        )
        artifact.apply_to_records(records, require_all=True)
        texts = artifact.texts_for(records)
        smoothing = {
            **artifact.identity(),
            "wishes_sha256": artifact.sha256,
        }
    return {
        "records": records,
        "keys": [record["key"] for record in records],
        "texts": texts,
        "labels": Y,
        "sources": [record["source"] for record in records],
        "label_counts": label_counts(Y),
        "label_prevalence": {
            label: round(float(Y[:, index].mean()), 6) for index, label in enumerate(LABELS)
        },
        "split": "train",
        "tasks_path": str(tasks_path or TASKS_PATH),
        "smoothing": smoothing,
        "artifact": {
            "tasks_sha256": _sha256_file(tasks_path or TASKS_PATH),
            "count": len(records),
        },
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kept_label_counts(tasks_path: Path | None = None) -> dict[str, int]:
    """Return positive counts per label across every kept task in all splits."""
    return label_counts(label_matrix(read_jsonl(tasks_path or TASKS_PATH)))


def split_counts(tasks_path: Path | None = None) -> dict[str, int]:
    """Return the number of kept tasks in each split."""
    records = read_jsonl(tasks_path or TASKS_PATH)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["split"]] = counts.get(record["split"], 0) + 1
    return counts


__all__ = [
    "TASKS_PATH",
    "kept_label_counts",
    "label_counts",
    "label_matrix",
    "load_records",
    "split_counts",
    "train_dataset",
    "wishes",
]
