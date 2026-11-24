"""R0 pipeline: pinned download, parse, validate, split, level construction, artifacts.

Run with ``make levels`` or ``PYTHONPATH=src python -m wishrec.build``.  The committed
artifacts are ``results/levels/tasks.jsonl``, ``manifest.json`` and ``stats.json``; hidden
tests and canonical solutions are written to gitignored ``data/derived/levels/``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .artifacts import (
    check_records,
    sha256_text,
    task_record,
    write_atomic_json,
    write_derived_task,
    write_jsonl,
)
from .config import (
    CACHE_DIR,
    DATA_DIR,
    DEFAULT_VISIBLE_TESTS,
    DERIVED_DIR,
    FIELD_KEYS,
    IMPORT_TO_DISTRIBUTION,
    LEVELS,
    LEVELS_DIR,
    PARSER_VERSION,
    PIPELINE_VERSION,
    SOURCE_ORDER,
    SOURCES,
    SPLIT_ORDER,
    SPLIT_SEED,
    SPLIT_SIZES,
    TARGET_MAIN_TASKS,
    TARGET_PILOT_TASKS,
    TARGET_TASKS,
    VALIDATOR_VERSION,
    VISIBLE_TEST_MODES,
)
from .datasets import download_sources, load_raw_rows
from .parse import parse_task
from .split import assign_splits, select_kept
from .validate import environment_info, validate_all


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_info() -> dict:
    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def portable_detail(detail: str) -> str:
    """Strip local absolute paths from exclusion evidence before committing it.

    Tracebacks captured during validation name the temporary workspace, the virtual
    environment and the interpreter.  They are useful evidence but must not leak the
    machine layout of the run that produced the cache, so the committed manifest keeps
    the traceback with stable placeholders instead.
    """
    if not detail:
        return ""
    text = re.sub(r"/[^\s\"']*/wishrec-validate-[^/\s\"']+", "<workspace>", detail)
    # Pytest reprs can truncate a path (".../folders/[90 chars]") before the random
    # workspace suffix is visible, so cover the platform temp roots directly as well.
    text = re.sub(r"/(?:private/)?var/folders/[^\s\"')]*", "<workspace>", text)
    text = re.sub(r"/tmp/[^\s\"')]*", "<workspace>", text)
    text = re.sub(
        r"/[^\s\"']*/\.venv/lib/(python[\d.]+)/site-packages",
        r"<venv>/lib/\1/site-packages",
        text,
    )
    text = re.sub(r"/[^\s\"']*/\.local/share/uv/python/[^/\s\"']+", "<python>", text)
    return text.replace(str(Path.home()), "~")


def _parse_all(
    rows: dict[str, list[dict]], *, visible_tests: str = DEFAULT_VISIBLE_TESTS
) -> tuple[list, list[dict]]:
    parsed = []
    exclusions = []
    for source in SOURCE_ORDER:
        source_rows = rows[source]
        source_parsed = 0
        for row in source_rows:
            source_task_id = str(row.get("task_id", ""))
            try:
                task = parse_task(source, row, visible_tests=visible_tests)
            except Exception as exc:  # noqa: BLE001 - exclusion is the contract
                exclusions.append(
                    {
                        "key": f"{source}/{source_task_id}",
                        "source": source,
                        "source_task_id": source_task_id,
                        "stage": "parse",
                        "exception": type(exc).__name__,
                        "missing_library": "",
                        "reason": str(exc).strip() or type(exc).__name__,
                    }
                )
                continue
            parsed.append(task)
            source_parsed += 1
        print(f"parsed {source}: {source_parsed}/{len(source_rows)}", flush=True)
    return parsed, exclusions


def _presence_block(tasks: list) -> dict:
    """Field-presence counts and rates, overall and per source."""

    def count(tasks_subset: list) -> dict:
        counts = {field: 0 for field in FIELD_KEYS}
        for task in tasks_subset:
            presence = task.fields_present
            for field in FIELD_KEYS:
                counts[field] += int(presence[field])
        n = len(tasks_subset)
        return {
            "denominator": n,
            "counts": counts,
            "rates": {field: round(counts[field] / n, 6) if n else None for field in FIELD_KEYS},
        }

    result = count(tasks)
    result["by_source"] = {
        source: count([task for task in tasks if task.source == source]) for source in SOURCE_ORDER
    }
    return result


def _level_char_stats(records: list[dict]) -> dict:
    result: dict = {"overall": {}, "by_source": {}}
    for level in LEVELS:
        lengths = [record["meta"]["level_chars"][level] for record in records]
        result["overall"][level] = {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "mean": round(statistics.fmean(lengths), 1),
            "max": max(lengths),
        }
    for source in SOURCE_ORDER:
        source_records = [record for record in records if record["source"] == source]
        result["by_source"][source] = (
            {
                level: {
                    "min": min(record["meta"]["level_chars"][level] for record in source_records),
                    "mean": round(
                        statistics.fmean(
                            record["meta"]["level_chars"][level] for record in source_records
                        ),
                        1,
                    ),
                    "max": max(record["meta"]["level_chars"][level] for record in source_records),
                }
                for level in LEVELS
            }
            if source_records
            else {}
        )
    return result


def _library_stats(records: list[dict], exclusions: list[dict]) -> dict:
    modules = Counter()
    distributions = Counter()
    by_source: dict[str, Counter] = {source: Counter() for source in SOURCE_ORDER}
    for record in records:
        for module in record["requirements"]:
            modules[module] += 1
            distributions[IMPORT_TO_DISTRIBUTION.get(module, module)] += 1
            by_source[record["source"]][module] += 1
    missing = Counter(item["missing_library"] for item in exclusions if item.get("missing_library"))
    return {
        "kept": dict(sorted(modules.items())),
        "distributions": dict(sorted(distributions.items())),
        "by_source": {
            source: dict(sorted(counter.items()))
            for source, counter in by_source.items()
            if counter
        },
        "missing_libraries": dict(sorted(missing.items())),
    }


def _bigcodebench_raises(rows: dict[str, list[dict]], valid_tasks: list) -> dict:
    nonempty = 0
    for row in rows["bigcodebench"]:
        try:
            doc = json.loads(row["doc_struct"])
        except (TypeError, json.JSONDecodeError, KeyError):
            continue
        if isinstance(doc, dict) and any(str(item).strip() for item in doc.get("raises") or []):
            nonempty += 1
    valid_positive = sum(
        1
        for task in valid_tasks
        if task.source == "bigcodebench" and task.fields_present["error_behaviour"]
    )
    return {
        "expected_rows_with_raises": 296,
        "rows_with_raises": nonempty,
        "matched_expected": nonempty == 296,
        "valid_tasks_with_error_behaviour": valid_positive,
    }


def _exclusion_summary(exclusions: list[dict]) -> dict:
    by_stage = Counter(item["stage"] for item in exclusions)
    by_source = Counter(item["source"] for item in exclusions)
    by_exception = Counter(item.get("exception") or "unknown" for item in exclusions)
    return {
        "total": len(exclusions),
        "by_stage": dict(sorted(by_stage.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_exception": dict(sorted(by_exception.items())),
        "missing_library": sum(1 for item in exclusions if item.get("missing_library")),
    }


def _write_summary(manifest: dict, stats: dict, out_dir: Path) -> None:
    counts = manifest["counts"]
    print("\n== R0 data foundation ==")
    print(f"artifact: {out_dir}")
    print(
        "rows: "
        + ", ".join(f"{source}={counts['rows'][source]}" for source in SOURCE_ORDER)
        + f" (total {counts['rows_total']})"
    )
    print(
        f"parsed {counts['parsed']}, valid {counts['valid']}, "
        f"excluded {counts['excluded']}, kept {counts['kept']}"
    )
    if counts["shortfall"]:
        print(
            f"SHORTFALL: {counts['shortfall']} tasks short of the {TARGET_TASKS}-task target",
            file=sys.stderr,
        )
    print("split:")
    for split in SPLIT_ORDER:
        by_source = stats["splits"]["split_by_source"][split]
        print(
            f"  {split}: {stats['splits']['counts'][split]} "
            f"({', '.join(f'{source}={by_source.get(source, 0)}' for source in SOURCE_ORDER)})"
        )
    print("field presence (kept):")
    for field in FIELD_KEYS:
        counts_field = stats["field_presence"]["kept"]["counts"][field]
        denominator = stats["field_presence"]["kept"]["denominator"]
        print(f"  {field}: {counts_field}/{denominator}")
    print(
        "BigCodeBench raises: "
        f"{stats['bigcodebench_raises']['rows_with_raises']} "
        f"(expected 296, matched={stats['bigcodebench_raises']['matched_expected']})"
    )
    print(f"level chars: {stats['level_chars']['overall']}")
    print(f"validation: {manifest['validation']}")
    if manifest["exclusions"]:
        print("exclusions:")
        for item in manifest["exclusions"]:
            print(
                f"  - {item['key']} [{item['stage']}] "
                f"{item.get('exception')}: {item['reason'][:160]}"
            )
    print("done")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="redownload and revalidate")
    parser.add_argument("--no-download", action="store_true", help="use cached raw files")
    parser.add_argument("--workers", type=int, default=6, help="validation workers")
    parser.add_argument("--visible-timeout", type=float, default=60.0)
    parser.add_argument("--hidden-timeout", type=float, default=180.0)
    parser.add_argument("--out", type=Path, default=LEVELS_DIR)
    parser.add_argument("--parse-only", action="store_true", help="parse and report only")
    parser.add_argument("--force-validation", action="store_true")
    parser.add_argument(
        "--visible-tests",
        choices=VISIBLE_TEST_MODES,
        default=DEFAULT_VISIBLE_TESTS,
        help=(
            "how derived visible tests treat expected results; 'full' keeps them, "
            "'inputs_only' prints the observed results (default: full)"
        ),
    )
    args = parser.parse_args(argv)

    started = time.monotonic()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.no_download:
        download_records = []
        for source in SOURCE_ORDER:
            from .datasets import sha256_file, _raw_path

            path = _raw_path(SOURCES[source])
            if not path.exists():
                raise FileNotFoundError(f"missing {path}; rerun without --no-download")
            digest = sha256_file(path)
            if digest != SOURCES[source].sha256:
                raise RuntimeError(f"{path}: sha256 {digest} != pinned {SOURCES[source].sha256}")
            download_records.append(
                {
                    "name": source,
                    "repo": SOURCES[source].repo,
                    "revision": SOURCES[source].revision,
                    "filename": SOURCES[source].filename,
                    "url": SOURCES[source].url,
                    "path": str(path.relative_to(DATA_DIR)),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                    "licence": SOURCES[source].licence,
                    "expected_rows": SOURCES[source].expected_rows,
                }
            )
    else:
        download_records = download_sources(force=args.force)

    rows = load_raw_rows()
    parsed, exclusions = _parse_all(rows, visible_tests=args.visible_tests)
    print(
        f"parsed {len(parsed)} tasks with {len(exclusions)} parse exclusions",
        flush=True,
    )
    if args.parse_only:
        for source in SOURCE_ORDER:
            source_tasks = [task for task in parsed if task.source == source]
            print(
                f"{source}: {len(source_tasks)} parsed, "
                f"error_behaviour={sum(t.fields_present['error_behaviour'] for t in source_tasks)}"
            )
        return 0

    results, validation_summary = validate_all(
        parsed,
        workers=args.workers,
        force=args.force or args.force_validation,
    )
    validation_summary["visible_timeout_seconds"] = args.visible_timeout
    validation_summary["hidden_timeout_seconds"] = args.hidden_timeout
    valid = [task for task in parsed if results[task.key].ok]
    failed = [task for task in parsed if not results[task.key].ok]
    for task in failed:
        result = results[task.key]
        exclusions.append(
            {
                "key": task.key,
                "source": task.source,
                "source_task_id": task.source_task_id,
                "stage": result.stage,
                "exception": result.exception,
                "missing_library": result.missing_library,
                "reason": (
                    f"{result.exception}"
                    + (f" (missing {result.missing_library})" if result.missing_library else "")
                ),
                "detail": portable_detail(result.detail[-1200:]),
            }
        )
    print(f"valid tasks: {len(valid)} (failed validation: {len(failed)})", flush=True)

    kept, selection_info = select_kept(valid, target=TARGET_TASKS, seed=SPLIT_SEED)
    if selection_info["shortfall"]:
        print(
            f"WARNING: only {len(kept)} valid tasks available; "
            f"{selection_info['shortfall']} short of {TARGET_TASKS}",
            file=sys.stderr,
        )
    assigned, split_info = assign_splits(kept, sizes=SPLIT_SIZES, seed=SPLIT_SEED)
    split_of = {task.key: split for split, tasks in assigned.items() for task in tasks}
    records = [task_record(task, split_of[task.key]) for task in kept]
    records.sort(key=lambda record: record["key"])
    split_counts = check_records(records)
    if split_counts != split_info["split_sizes"]:
        raise AssertionError(f"split counts {split_counts} != {split_info['split_sizes']}")

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    for task in kept:
        write_derived_task(task)
    print(f"wrote derived tasks to {DERIVED_DIR}", flush=True)

    tasks_path = args.out / "tasks.jsonl"
    tasks_sha = write_jsonl(tasks_path, records)

    pool_presence = _presence_block(valid)
    kept_presence = _presence_block(kept)
    exclusions_summary = _exclusion_summary(exclusions)
    library_stats = _library_stats(records, exclusions)
    raises_stats = _bigcodebench_raises(rows, valid)

    split_stats = {
        "counts": {split: len(assigned[split]) for split in SPLIT_ORDER},
        "split_by_source": split_info["split_by_source"],
        "split_sizes": split_info["split_sizes"],
        "target_sizes": dict(SPLIT_SIZES),
        "seed": SPLIT_SEED,
        "algorithm": split_info["algorithm"],
        "heldout_untouched": split_info["heldout_untouched"],
        "pilot_disjoint": split_info["pilot_disjoint"],
    }
    counts = {
        "rows": {source: len(rows[source]) for source in SOURCE_ORDER},
        "rows_total": sum(len(rows[source]) for source in SOURCE_ORDER),
        "parsed": len(parsed),
        "excluded": len(exclusions),
        "valid": len(valid),
        "kept": len(kept),
        "unused": len(valid) - len(kept),
        "shortfall": selection_info["shortfall"],
        "target": TARGET_TASKS,
        "main_target": TARGET_MAIN_TASKS,
        "pilot_target": TARGET_PILOT_TASKS,
    }
    stats = {
        "counts": counts,
        "splits": split_stats,
        "field_presence": {
            "kept": kept_presence,
            "pool": pool_presence,
        },
        "bigcodebench_raises": raises_stats,
        "exclusions": exclusions_summary,
        "library_requirements": library_stats,
        "level_chars": _level_char_stats(records),
        "validation": validation_summary,
        "notes": [
            "L3 is the whole prose intent block (all docstring prose before the examples). "
            "LM smoothing of the wish is deferred to R1/R2 because R0 has a $0 API budget "
            "and no GPU; levels are the mechanical field-deletion baseline.",
            "Hidden tests and canonical solutions live in data/derived/levels/ (gitignored) "
            "and are regenerated by make levels; tasks.jsonl records their SHA-256.",
            "The 300 held-out tasks are never used for detector training, calibration, "
            "threshold selection or the pilot.",
            "BigCodeBench tasks with fewer than three unittest methods are excluded because "
            "the milestone requires three visible tests.",
        ],
    }
    stats_path = args.out / "stats.json"
    write_atomic_json(stats_path, stats)
    stats_sha = sha256_text(stats_path.read_text(encoding="utf-8"))

    manifest = {
        "schema_version": 1,
        "phase": "R0",
        "title": "R0: scaffold + 1,000-task data foundation",
        "created_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "package_version": __version__,
        "parser_version": PARSER_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "parsing": {"visible_tests": args.visible_tests},
        "git": git_info(),
        "environment": environment_info(),
        "sources": download_records,
        "counts": counts,
        "selection": {
            key: value for key, value in selection_info.items() if key != "unused_task_keys"
        },
        "unused_task_keys": selection_info["unused_task_keys"],
        "split": split_stats,
        "field_presence_pool": pool_presence,
        "field_presence_kept": kept_presence,
        "exclusions": exclusions,
        "validation": validation_summary,
        "l3": {
            "smoothing": "none",
            "reason": "$0 API budget in R0; L3 is the raw intent sentence",
            "smoothed_by": None,
        },
        "artifacts": {
            "tasks_sha256": tasks_sha,
            "stats_sha256": stats_sha,
            "task_count": len(records),
            "derived_dir": str(DERIVED_DIR.relative_to(DERIVED_DIR.parents[2])),
        },
        "notes": stats["notes"],
    }
    manifest_path = args.out / "manifest.json"
    write_atomic_json(manifest_path, manifest)

    _write_summary(manifest, stats, args.out)
    print(f"elapsed: {time.monotonic() - started:.0f}s")
    if selection_info["shortfall"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
