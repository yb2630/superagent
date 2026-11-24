"""Smoothed-L3 artifact produced by the R2 GPU run.

The R0 artifact's ``L3`` is the raw first intent sentence (``results/levels/tasks.jsonl``).
R2 smooths the detector-train and pilot wishes with the generator endpoint
(``python -m wishrec.harness.smoothing``) and stores one record per task:

    {"task_key": ..., "split": ..., "source": ..., "raw_wish": ...,
     "smoothed_wish": ..., "changed": bool, "fallback": bool,
     "model": ..., "transport": ..., "usage": {...}, "created_at": ...,
     "duration_s": ...}

This module loads and applies that artifact.  It deliberately contains no endpoint code so
the detector can import it without pulling in the OpenAI client.  The committed R0 artifact
is never modified: the overlay is applied to in-memory records/tasks, and the runner records
the artifact SHA-256 in its manifest so an interrupted run cannot silently switch wishes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .config import RESULTS_DIR

SMOOTHED_LEVEL = "L3"
ARTIFACT_NAME = "wishes.jsonl"
MANIFEST_NAME = "manifest.json"


def default_smoothing_dir(run_id: str) -> Path:
    """``results/runs/<run_id>/smoothing`` (gitignored only if results/runs is)."""
    return RESULTS_DIR / "runs" / run_id / "smoothing"


def default_smoothing_path(run_id: str) -> Path:
    return default_smoothing_dir(run_id) / ARTIFACT_NAME


def resolve_smoothing_path(value: str | Path, *, run_id: str | None = None) -> Path:
    """Resolve a file, a run-id, or a directory to the JSONL artifact."""
    if run_id is not None and str(value) == run_id:
        return default_smoothing_path(run_id)
    path = Path(value)
    if path.is_dir():
        path = path / ARTIFACT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"smoothed-wish artifact missing at {path}; run "
            "`python -m wishrec.harness.smoothing --run-id <id>` first"
        )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        records.append(value)
    return records


@dataclass(frozen=True)
class SmoothingArtifact:
    """One validated smoothed-L3 artifact plus its optional manifest."""

    path: Path
    records: Mapping[str, dict]
    manifest: Mapping[str, object]
    sha256: str

    @property
    def model(self) -> str:
        endpoint = self.manifest.get("endpoint") or {}
        return str(endpoint.get("model", "") or "")

    @property
    def splits(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records.values():
            split = str(record.get("split", ""))
            counts[split] = counts.get(split, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def fallbacks(self) -> int:
        return sum(1 for record in self.records.values() if record.get("fallback"))

    def identity(self) -> dict:
        """Stable identity for a run manifest (no timestamps, no absolute paths)."""
        return {
            "artifact": ARTIFACT_NAME,
            "sha256": self.sha256,
            "model": self.model,
            "records": len(self.records),
            "splits": self.splits,
            "fallbacks": self.fallbacks,
        }

    def wish(self, task_key: str) -> str:
        try:
            record = self.records[task_key]
        except KeyError as exc:
            raise KeyError(f"no smoothed wish for {task_key!r}") from exc
        return str(record["smoothed_wish"])

    def missing(self, keys: Iterable[str]) -> list[str]:
        return sorted(str(key) for key in keys if str(key) not in self.records)

    def apply_to_records(self, records: Sequence[dict], *, require_all: bool = True) -> int:
        """Replace ``levels[L3]`` on task records in place; return the changed count."""
        missing = self.missing(record["key"] for record in records)
        if missing and require_all:
            preview = ", ".join(missing[:5])
            raise KeyError(
                f"smoothed-wish artifact is missing {len(missing)} task(s): {preview}; "
                "re-run the smoothing step or select tasks it covers"
            )
        changed = 0
        for record in records:
            key = str(record["key"])
            if key not in self.records:
                continue
            record.setdefault("levels", {})[SMOOTHED_LEVEL] = self.wish(key)
            changed += 1
        return changed

    def apply_to_tasks(self, tasks: Iterable, *, require_all: bool = True) -> int:
        """Replace ``task.levels[L3]``; the committed record is left untouched."""
        tasks = list(tasks)
        missing = self.missing(task.key for task in tasks)
        if missing and require_all:
            preview = ", ".join(missing[:5])
            raise KeyError(
                f"smoothed-wish artifact is missing {len(missing)} task(s): {preview}; "
                "re-run the smoothing step or select tasks it covers"
            )
        changed = 0
        for task in tasks:
            if task.key in self.records:
                task.levels[SMOOTHED_LEVEL] = self.wish(task.key)
                changed += 1
        return changed

    def texts_for(self, records: Sequence[dict]) -> list[str]:
        return [self.wish(str(record["key"])) for record in records]


def load_smoothing(
    value: str | Path | None = None,
    *,
    run_id: str | None = None,
    path: Path | None = None,
) -> SmoothingArtifact:
    """Load and validate the JSONL artifact and the manifest beside it, when present."""
    if path is not None:
        target = path
    elif value is not None:
        target = resolve_smoothing_path(value, run_id=run_id)
    elif run_id is not None:
        target = resolve_smoothing_path(run_id, run_id=run_id)
    else:
        raise ValueError("load_smoothing needs a path or a run id")
    records: dict[str, dict] = {}
    for record in _read_jsonl(target):
        key = str(record.get("task_key", "")).strip()
        if not key:
            raise ValueError(f"{target}: record without task_key")
        if key in records:
            raise ValueError(f"{target}: duplicate task_key {key!r}")
        if not str(record.get("smoothed_wish", "")).strip():
            raise ValueError(f"{target}: empty smoothed_wish for {key!r}")
        records[key] = record
    manifest_path = target.with_name(MANIFEST_NAME)
    if target.name != ARTIFACT_NAME:
        manifest_path = target.parent / MANIFEST_NAME
    manifest: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return SmoothingArtifact(
        path=target,
        records=dict(sorted(records.items())),
        manifest=manifest,
        sha256=_sha256_file(target),
    )


__all__ = [
    "ARTIFACT_NAME",
    "MANIFEST_NAME",
    "SMOOTHED_LEVEL",
    "SmoothingArtifact",
    "default_smoothing_dir",
    "default_smoothing_path",
    "load_smoothing",
    "resolve_smoothing_path",
]
