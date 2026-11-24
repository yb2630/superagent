"""Task-record serialization, committed artifact writing and invariant checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import (
    ARTIFACT_SCHEMA_VERSION,
    DERIVED_DIR,
    FIELD_KEYS,
    LEVELS,
    SPLIT_ORDER,
)
from .levels import build_levels, level_char_lengths

if TYPE_CHECKING:
    from .parse import ParsedTask


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_task_id(source_task_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in source_task_id
    )


def derived_task_dir(task: ParsedTask) -> Path:
    """Return the gitignored ``data/derived/levels`` directory for one task."""
    return DERIVED_DIR / task.source / _safe_task_id(task.source_task_id)


def write_derived_task(task: ParsedTask) -> Path:
    """Write the stub, visible/hidden tests and canonical solution; return the directory."""
    directory = derived_task_dir(task)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "stub.py": task.stub,
        "visible_test.py": task.visible_test,
        "hidden_test.py": task.hidden_test,
        "canonical_solution.py": task.canonical_solution,
    }
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


def task_record(task: ParsedTask, split: str) -> dict[str, Any]:
    """Build the committed ``tasks.jsonl`` record (levels, presence, hashes) for a task."""
    signature_is_code = task.source != "bigcodebench"
    built = build_levels(task.fields, signature_is_code=signature_is_code)
    presence = built["field_presence"]
    meta = dict(task.meta)
    meta["level_chars"] = level_char_lengths(built)
    meta["signature_is_code"] = signature_is_code
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "key": task.key,
        "source": task.source,
        "source_task_id": task.source_task_id,
        "entry_point": task.entry_point,
        "split": split,
        "fields": dict(task.fields),
        "field_presence": presence,
        "levels": built["levels"],
        "deleted_fields": built["deleted_fields"],
        "L2_c": built["L2_c"],
        "stub": task.stub,
        "stub_sha256": sha256_text(task.stub),
        "visible_asserts": task.visible_asserts,
        "visible_asserts_count": task.visible_asserts_count,
        "visible_test_sha256": sha256_text(task.visible_test),
        "hidden_test_sha256": sha256_text(task.hidden_test),
        "hidden_test_bytes": len(task.hidden_test.encode("utf-8")),
        "hidden_test_shared_across_levels": True,
        "canonical_sha256": sha256_text(task.canonical_solution),
        "requirements": list(task.requirements),
        "validated": True,
        "meta": meta,
    }


def write_atomic_text(path: Path, text: str) -> None:
    """Write ``text`` via a ``.part`` file and rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write ``data`` as indented, key-sorted JSON."""
    write_atomic_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    """Atomically write ``records`` as key-sorted JSON lines; return the payload SHA-256."""
    lines = [json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records]
    payload = "\n".join(lines) + "\n"
    write_atomic_text(path, payload)
    return sha256_text(payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file, skipping blank lines."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def check_records(records: list[dict[str, Any]]) -> dict[str, int]:
    """Check the task-record schema and internal consistency; return split counts."""
    if not records:
        raise ValueError("no task records")
    keys: set[str] = set()
    split_counts = dict.fromkeys(SPLIT_ORDER, 0)
    for record in records:
        missing = {
            "schema_version",
            "key",
            "source",
            "source_task_id",
            "entry_point",
            "split",
            "fields",
            "field_presence",
            "levels",
            "deleted_fields",
            "L2_c",
            "stub",
            "stub_sha256",
            "visible_asserts",
            "visible_asserts_count",
            "visible_test_sha256",
            "hidden_test_sha256",
            "hidden_test_bytes",
            "hidden_test_shared_across_levels",
            "canonical_sha256",
            "requirements",
            "validated",
            "meta",
        } - set(record)
        if missing:
            raise ValueError(f"{record.get('key')}: missing record keys {sorted(missing)}")
        if record["key"] in keys:
            raise ValueError(f"duplicate task key {record['key']}")
        keys.add(record["key"])
        if record["split"] not in split_counts:
            raise ValueError(f"{record['key']}: unknown split {record['split']!r}")
        split_counts[record["split"]] += 1
        if record["validated"] is not True:
            raise ValueError(f"{record['key']}: kept task was not validated")
        if set(record["fields"]) != set(FIELD_KEYS) | {"signature_untyped"}:
            raise ValueError(f"{record['key']}: unexpected field set")
        for field in FIELD_KEYS:
            present = bool(record["fields"][field].strip())
            if record["field_presence"][field] != present:
                raise ValueError(f"{record['key']}: field_presence mismatch for {field}")
        if set(record["levels"]) != set(LEVELS):
            raise ValueError(f"{record['key']}: level set mismatch")
        if record["levels"]["L3"] != record["fields"]["intent"].strip():
            raise ValueError(f"{record['key']}: L3 is not exactly the intent")
        if record["deleted_fields"]["L0"] != []:
            raise ValueError(f"{record['key']}: L0 must not delete fields")
        if set(record["L2_c"]) != set(FIELD_KEYS):
            raise ValueError(f"{record['key']}: L2_c categories mismatch")
        for category, drop in record["L2_c"].items():
            if drop["field_present"] != record["field_presence"][category]:
                raise ValueError(f"{record['key']}: L2_c presence mismatch for {category}")
            if not drop["text"].strip():
                raise ValueError(f"{record['key']}: empty L2_c text for {category}")
        if sha256_text(record["stub"]) != record["stub_sha256"]:
            raise ValueError(f"{record['key']}: stub hash mismatch")
        if record["visible_asserts_count"] != 3:
            raise ValueError(f"{record['key']}: expected exactly three visible tests")
        if record["hidden_test_shared_across_levels"] is not True:
            raise ValueError(f"{record['key']}: hidden suite must be shared across levels")
        for value in (
            record["stub_sha256"],
            record["visible_test_sha256"],
            record["hidden_test_sha256"],
            record["canonical_sha256"],
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{record['key']}: invalid sha256 {value!r}")
    return split_counts
