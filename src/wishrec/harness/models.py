"""Development-task view shared by the ReAct loop, runner and tests.

The committed ``results/levels/tasks.jsonl`` carries the five fields, L0-L3, the stub and
the visible-test assertions, but deliberately not the hidden suite or the canonical
solution.  Those are materialized by ``make levels`` in the gitignored
``data/derived/levels/<source>/<task_id>/`` directory and their SHA-256 is checked against
the committed record before a task is used.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ..config import FIELD_KEYS, LEVELS
from .config import DERIVED_TASKS_DIR, LEVEL_MANIFEST_PATH, LEVEL_TASKS_PATH, read_jsonl


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_task_id(source_task_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(source_task_id)
    )


def derived_task_dir(source: str, source_task_id: str, root: Path | None = None) -> Path:
    return (root or DERIVED_TASKS_DIR) / source / _safe_task_id(source_task_id)


@dataclass
class Task:
    """One kept task, with the derived test files required by the harness."""

    key: str
    source: str
    source_task_id: str
    entry_point: str
    split: str
    stub: str
    fields: dict[str, str]
    field_presence: dict[str, bool]
    levels: dict[str, str]
    visible_asserts: str
    visible_asserts_count: int
    visible_test: str
    hidden_test: str
    hidden_test_sha256: str
    canonical_solution: str = ""
    canonical_sha256: str = ""
    requirements: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict, repr=False)

    @property
    def wish(self) -> str:
        return self.levels["L3"]

    @property
    def expected_visible_cases(self) -> int:
        return int(self.visible_asserts_count or 3)

    @classmethod
    def from_record(cls, record: dict, *, derived_root: Path | None = None) -> Task:
        directory = derived_task_dir(record["source"], record["source_task_id"], root=derived_root)
        visible_path = directory / "visible_test.py"
        hidden_path = directory / "hidden_test.py"
        canonical_path = directory / "canonical_solution.py"
        if not visible_path.is_file() or not hidden_path.is_file():
            raise FileNotFoundError(
                f"derived tests for {record['key']} are missing at {directory}; "
                "run `make levels` to materialize them"
            )
        visible = visible_path.read_text(encoding="utf-8")
        hidden = hidden_path.read_text(encoding="utf-8")
        canonical = canonical_path.read_text(encoding="utf-8") if canonical_path.is_file() else ""
        expected_visible = record.get("visible_test_sha256")
        expected_hidden = record.get("hidden_test_sha256")
        expected_canonical = record.get("canonical_sha256")
        if expected_visible and _digest(visible) != expected_visible:
            raise RuntimeError(
                f"visible test hash mismatch for {record['key']}; rerun `make levels`"
            )
        if expected_hidden and _digest(hidden) != expected_hidden:
            raise RuntimeError(
                f"hidden test hash mismatch for {record['key']}; rerun `make levels`"
            )
        if expected_canonical and canonical and _digest(canonical) != expected_canonical:
            raise RuntimeError(
                f"canonical solution hash mismatch for {record['key']}; rerun `make levels`"
            )
        missing_fields = set(FIELD_KEYS) - set(record["fields"])
        if missing_fields:
            raise ValueError(f"{record['key']}: missing fields {sorted(missing_fields)}")
        if set(record["levels"]) != set(LEVELS):
            raise ValueError(f"{record['key']}: unexpected level set")
        if record["levels"]["L3"] != record["fields"]["intent"].strip():
            raise ValueError(f"{record['key']}: L3 is not exactly the intent field")
        return cls(
            key=record["key"],
            source=record["source"],
            source_task_id=str(record["source_task_id"]),
            entry_point=record["entry_point"],
            split=record["split"],
            stub=record["stub"],
            fields=dict(record["fields"]),
            field_presence={field: bool(record["field_presence"][field]) for field in FIELD_KEYS},
            levels=dict(record["levels"]),
            visible_asserts=record["visible_asserts"],
            visible_asserts_count=int(record["visible_asserts_count"]),
            visible_test=visible,
            hidden_test=hidden,
            hidden_test_sha256=record["hidden_test_sha256"],
            canonical_solution=canonical,
            canonical_sha256=record["canonical_sha256"],
            requirements=list(record.get("requirements") or []),
            meta=dict(record.get("meta") or {}),
            record=record,
        )


def load_manifest(path: Path | None = None) -> dict:
    import json

    manifest_path = path or LEVEL_MANIFEST_PATH
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_tasks(
    *,
    split: str | None = None,
    keys: list[str] | None = None,
    limit: int | None = None,
    tasks_path: Path | None = None,
    derived_root: Path | None = None,
    require_derived: bool = True,
) -> list[Task]:
    """Load kept tasks, optionally filtered by split/key.

    ``require_derived=False`` is for detector-only or planning paths that need the
    committed record but not the hidden tests; the returned task then has empty test
    sources and must not be used to generate or evaluate code.
    """
    records = read_jsonl(tasks_path or LEVEL_TASKS_PATH)
    selected: list[dict] = []
    wanted = set(keys) if keys else None
    for record in records:
        if split is not None and record["split"] != split:
            continue
        if wanted is not None and record["key"] not in wanted:
            continue
        selected.append(record)
    selected.sort(key=lambda record: record["key"])
    if limit is not None:
        selected = selected[:limit]
    if not require_derived:
        tasks = []
        for record in selected:
            tasks.append(_task_without_derived(record))
        return tasks
    tasks = [Task.from_record(record, derived_root=derived_root) for record in selected]
    if wanted is not None:
        found = {task.key for task in tasks}
        missing = wanted - found
        if missing:
            raise KeyError(f"task keys not found: {sorted(missing)}")
    return tasks


def _task_without_derived(record: dict) -> Task:
    return Task(
        key=record["key"],
        source=record["source"],
        source_task_id=str(record["source_task_id"]),
        entry_point=record["entry_point"],
        split=record["split"],
        stub=record["stub"],
        fields=dict(record["fields"]),
        field_presence={field: bool(record["field_presence"][field]) for field in FIELD_KEYS},
        levels=dict(record["levels"]),
        visible_asserts=record["visible_asserts"],
        visible_asserts_count=int(record["visible_asserts_count"]),
        visible_test="",
        hidden_test="",
        hidden_test_sha256=record["hidden_test_sha256"],
        canonical_sha256=record["canonical_sha256"],
        requirements=list(record.get("requirements") or []),
        meta=dict(record.get("meta") or {}),
        record=record,
    )


__all__ = ["Task", "derived_task_dir", "load_manifest", "load_tasks"]
