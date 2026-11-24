"""Materialize the committed ``data/derived/levels`` tests from committed level records.

``make levels`` re-parses, re-validates and re-selects the task set, which makes the split
environment-dependent.  The pilot must instead use the committed ``results/levels/tasks.jsonl``
and only materialize its derived test files.  This module:

- checks (``--check-only``) that every committed task's four derived files exist and match
  the SHA-256 values stored in its record;
- rebuilds the derived files by parsing the pinned raw rows, without validation or selection,
  in the visible-test mode recorded in each record (``meta.visible_tests``) unless
  ``--visible-tests`` overrides it;
- verifies the rebuilt hidden/visible/canonical/stub bytes against the record before writing,
  so a snapshot from a different task set or mode is reported and never written.

Run ``python -m wishrec.materialize --tasks results/levels/tasks.jsonl --download --clean``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .artifacts import _safe_task_id, sha256_text
from .config import DEFAULT_VISIBLE_TESTS, DERIVED_DIR, LEVELS_DIR, VISIBLE_TEST_MODES
from .datasets import download_sources, load_raw_rows
from .parse import ParsedTask, parse_task

HASH_FIELDS = {
    "hidden_test_sha256": "hidden_test",
    "visible_test_sha256": "visible_test",
    "canonical_sha256": "canonical_solution",
    "stub_sha256": "stub",
}

#: Derived file name for each task attribute.
FILE_NAMES = {
    "stub": "stub.py",
    "visible_test": "visible_test.py",
    "hidden_test": "hidden_test.py",
    "canonical_solution": "canonical_solution.py",
}


def task_dir(root: Path, source: str, source_task_id: str) -> Path:
    """Return ``<root>/<source>/<safe task id>``, the derived directory for one task."""
    return root / source / _safe_task_id(source_task_id)


def write_task_files(task: ParsedTask, root: Path) -> Path:
    """Write a task's stub, visible/hidden tests and canonical solution under ``root``."""
    directory = task_dir(root, task.source, task.source_task_id)
    directory.mkdir(parents=True, exist_ok=True)
    for attribute, name in FILE_NAMES.items():
        (directory / name).write_text(getattr(task, attribute), encoding="utf-8")
    return directory


def record_mode(record: dict[str, Any]) -> str:
    """Return the visible-test mode a record was built with (default if unrecorded)."""
    return str(record.get("meta", {}).get("visible_tests", DEFAULT_VISIBLE_TESTS))


def _hash_problems(record: dict[str, Any], texts: dict[str, str]) -> list[dict[str, str]]:
    """Compare ``texts`` (attribute -> content) with the record's hashes.

    A missing hash field is a problem too: an unverifiable record must not pass silently.
    """
    problems = []
    for field, attribute in HASH_FIELDS.items():
        expected = record.get(field)
        actual = sha256_text(texts[attribute])
        if not expected or expected != actual:
            problems.append(
                {
                    "key": record["key"],
                    "field": field,
                    "expected": expected or "<missing>",
                    "actual": actual,
                }
            )
    return problems


def verify_derived(records: list[dict], derived_dir: Path | None = None) -> dict[str, Any]:
    """Check that every record's four derived files exist and match its stored hashes."""
    root = derived_dir or DERIVED_DIR
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    for record in records:
        directory = task_dir(root, record["source"], record["source_task_id"])
        paths = {attribute: directory / name for attribute, name in FILE_NAMES.items()}
        if not all(path.is_file() for path in paths.values()):
            missing.append(record["key"])
            continue
        texts = {attribute: path.read_text(encoding="utf-8") for attribute, path in paths.items()}
        mismatched.extend(_hash_problems(record, texts))
    return {
        "records": len(records),
        "missing": missing,
        "mismatched": mismatched,
        "derived_dir": str(root),
    }


def read_records(tasks_path: Path | None = None) -> list[dict[str, Any]]:
    """Load the committed level records, failing loudly on invalid or empty files."""
    path = tasks_path or (LEVELS_DIR / "tasks.jsonl")
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def derived_path(record: dict[str, Any], derived_dir: Path | None = None) -> Path:
    """Return the expected derived ``hidden_test.py`` path for a committed record."""
    root = derived_dir or DERIVED_DIR
    return task_dir(root, record["source"], record["source_task_id"]) / "hidden_test.py"


def coverage(records: list[dict], derived_dir: Path | None = None) -> list[str]:
    """Return the keys whose derived hidden test is missing."""
    return [record["key"] for record in records if not derived_path(record, derived_dir).exists()]


def materialize(
    records: list[dict],
    *,
    download: bool = False,
    clean: bool = False,
    derived_dir: Path | None = None,
    visible_tests: str | None = None,
) -> dict:
    """Write derived files from the pinned raw rows, verifying each against its record.

    Each record is parsed in the visible-test mode it was built with unless
    ``visible_tests`` overrides it.  A task whose rebuilt bytes do not match the record's
    hashes (or whose record lacks a hash) is reported under ``mismatched`` and not written.
    """
    if download:
        download_sources()
    rows = load_raw_rows()
    modes = (
        {visible_tests}
        if visible_tests is not None
        else ({record_mode(record) for record in records} or {DEFAULT_VISIBLE_TESTS})
    )
    parsed: dict[str, dict[str, ParsedTask]] = {mode: {} for mode in sorted(modes)}
    parse_failures: list[dict[str, str]] = []
    for mode, by_key in parsed.items():
        for source, source_rows in rows.items():
            for row in source_rows:
                try:
                    candidate = parse_task(source, row, visible_tests=mode)
                except Exception as exc:  # noqa: BLE001 - record, do not abort the whole build
                    parse_failures.append({"source": source, "mode": mode, "error": str(exc)})
                    continue
                by_key[candidate.key] = candidate

    root = derived_dir or DERIVED_DIR
    if clean and root.exists():
        shutil.rmtree(root)

    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    written = 0
    for record in records:
        mode = visible_tests if visible_tests is not None else record_mode(record)
        task = parsed[mode].get(record["key"])
        if task is None:
            missing.append(record["key"])
            continue
        problems = _hash_problems(
            record, {attribute: getattr(task, attribute) for attribute in HASH_FIELDS.values()}
        )
        if problems:
            mismatched.extend(problems)
            continue
        write_task_files(task, root)
        written += 1

    still_missing = coverage(records, root)
    return {
        "records": len(records),
        "written": written,
        "missing": missing,
        "mismatched": mismatched,
        "coverage_missing": still_missing,
        "parse_failures": parse_failures,
        "derived_dir": str(root),
    }


def main(argv: list[str] | None = None) -> int:
    """Check or rebuild the derived files from the CLI; return 0 when everything matches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=LEVELS_DIR / "tasks.jsonl")
    parser.add_argument("--derived", type=Path, default=None, help="override data/derived/levels")
    parser.add_argument(
        "--download", action="store_true", help="download/verify pinned raw data first"
    )
    parser.add_argument(
        "--clean", action="store_true", help="remove the derived root before writing"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify that the derived files exist and match the record hashes; write nothing",
    )
    parser.add_argument(
        "--visible-tests",
        choices=VISIBLE_TEST_MODES,
        default=None,
        help=(
            "override how derived visible tests treat expected results "
            "(default: the mode recorded in each record)"
        ),
    )
    args = parser.parse_args(argv)

    records = read_records(args.tasks)
    if args.check_only:
        report = verify_derived(records, args.derived)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if not report["missing"] and not report["mismatched"] else 1

    summary = materialize(
        records,
        download=args.download,
        clean=args.clean,
        derived_dir=args.derived,
        visible_tests=args.visible_tests,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    ok = not summary["missing"] and not summary["mismatched"] and not summary["coverage_missing"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
