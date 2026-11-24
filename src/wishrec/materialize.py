"""Materialize the gitignored ``data/derived/levels`` tests from committed level records.

``make levels`` re-parses, re-validates and re-selects the task set, which makes the split
environment-dependent.  The pilot must instead use the committed ``results/levels/tasks.jsonl``
and only materialize its derived test files.  This module:

- checks coverage: every committed task has ``data/derived/levels/<source>/<task>/hidden_test.py``;
- rebuilds the derived files by parsing the pinned raw rows, without validation or selection;
- verifies the rebuilt hidden/visible/canonical/stub bytes against the SHA-256 values stored in
  the committed record, so a snapshot from a different task set cannot silently be used.

Run ``python -m wishrec.materialize --tasks results/levels/tasks.jsonl --download --clean``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .artifacts import _safe_task_id, sha256_text, write_derived_task
from .config import DEFAULT_VISIBLE_TESTS, DERIVED_DIR, LEVELS_DIR, VISIBLE_TEST_MODES
from .datasets import download_sources, load_raw_rows
from .parse import ParsedTask, parse_task

HASH_FIELDS = {
    "hidden_test_sha256": "hidden_test",
    "visible_test_sha256": "visible_test",
    "canonical_sha256": "canonical_solution",
    "stub_sha256": "stub",
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
    return root / record["source"] / _safe_task_id(record["source_task_id"]) / "hidden_test.py"


def coverage(records: list[dict], derived_dir: Path | None = None) -> list[str]:
    """Return the keys whose derived hidden test is missing."""
    return [record["key"] for record in records if not derived_path(record, derived_dir).exists()]


def materialize(
    records: list[dict],
    *,
    download: bool = False,
    clean: bool = False,
    derived_dir: Path | None = None,
    visible_tests: str = DEFAULT_VISIBLE_TESTS,
) -> dict:
    """Write derived files from the pinned raw rows and verify them against the records."""
    if download:
        download_sources()
    rows = load_raw_rows()
    parsed: dict[str, ParsedTask] = {}
    parse_failures: list[dict[str, str]] = []
    for source, source_rows in rows.items():
        for row in source_rows:
            try:
                candidate = parse_task(source, row, visible_tests=visible_tests)
            except Exception as exc:  # noqa: BLE001 - record, do not abort the whole build
                parse_failures.append({"source": source, "error": str(exc)})
                continue
            parsed[candidate.key] = candidate

    root = derived_dir or DERIVED_DIR
    if clean and root.exists():
        shutil.rmtree(root)

    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    written = 0
    for record in records:
        task = parsed.get(record["key"])
        if task is None:
            missing.append(record["key"])
            continue
        for field, attribute in HASH_FIELDS.items():
            expected = record.get(field)
            actual = sha256_text(getattr(task, attribute))
            if expected and expected != actual:
                mismatched.append(
                    {"key": record["key"], "field": field, "expected": expected, "actual": actual}
                )
        if derived_dir is None:
            write_derived_task(task)
        else:
            directory = derived_path(record, derived_dir).parent
            directory.mkdir(parents=True, exist_ok=True)
            for name, text in {
                "stub.py": task.stub,
                "visible_test.py": task.visible_test,
                "hidden_test.py": task.hidden_test,
                "canonical_solution.py": task.canonical_solution,
            }.items():
                (directory / name).write_text(text, encoding="utf-8")
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
    parser.add_argument("--check-only", action="store_true", help="report coverage only")
    parser.add_argument(
        "--visible-tests",
        choices=VISIBLE_TEST_MODES,
        default=DEFAULT_VISIBLE_TESTS,
        help=(
            "how derived visible tests treat expected results; match the mode used "
            "to build the records (default: full)"
        ),
    )
    args = parser.parse_args(argv)

    records = read_records(args.tasks)
    if args.check_only:
        missing = coverage(records, args.derived)
        print(json.dumps({"records": len(records), "missing": missing}, indent=2))
        return 0 if not missing else 1

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
