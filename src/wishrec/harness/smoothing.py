"""L3 smoothing: rewrite the train and pilot wishes with the generator endpoint.

The mechanical L3 is the whole parsed intent prose.  The milestone smooths it into a natural
wish with a language model before the pilot.  This module executes that step against the
operator's Qwen3-8B endpoint:

    python -m wishrec.harness.smoothing --run-id r2b-pilot --splits train pilot

It writes ``results/runs/<run-id>/smoothing/wishes.jsonl`` (one record per task) plus a
manifest, and is resumable: completed task keys are skipped on a second invocation, and each
new record is appended immediately so an interrupted Colab session keeps its progress.  The
500 ``train`` tasks feed the detector refit; the 100 ``pilot`` tasks are the ones the harness
runs, so both use one smoothed-wish artifact.  The 300 ``heldout`` tasks are never sent to
the endpoint and are refused here.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..smoothing import (
    ARTIFACT_NAME,
    MANIFEST_NAME,
    SmoothingArtifact,
    default_smoothing_dir,
    load_smoothing,
)
from .config import (
    atomic_write_json,
    atomic_write_text,
    environment_info,
    generator_config,
    git_info,
    utc_now_iso,
)
from .llm import LLMClient
from .models import load_tasks
from .prompts import SMOOTH_SYSTEM
from .wish import smooth_wish

#: Splits the R2 pilot smooths.  ``heldout`` is deliberately absent: it is R3-only.
SMOOTHING_SPLITS = ("train", "pilot")
#: The detector needs exactly this many train wishes before an R2 refit is valid.
TRAIN_TARGET = 500
PILOT_TARGET = 100


def _read_existing(path: Path) -> dict[str, dict]:
    """Lenient reader for resume; a torn final line after a disconnect is ignored."""
    records: dict[str, dict] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("task_key") and value.get("smoothed_wish"):
            records[str(value["task_key"])] = value
    return records


def _smooth_one(task, llm: LLMClient) -> dict:
    started = time.monotonic()
    raw = str(task.wish).strip()
    smoothed = smooth_wish(
        llm,
        raw,
        meta={"task_key": task.key, "split": task.split, "source": task.source},
    )
    fallback = not smoothed or smoothed == raw
    return {
        "task_key": task.key,
        "split": task.split,
        "source": task.source,
        "source_task_id": task.source_task_id,
        "raw_wish": raw,
        "smoothed_wish": smoothed or raw,
        "changed": (smoothed or raw) != raw,
        "fallback": fallback,
        "model": llm.config.model,
        "created_at": utc_now_iso(),
        "duration_s": round(time.monotonic() - started, 3),
    }


def smooth_tasks(
    tasks: list,
    llm: LLMClient,
    output_dir: Path,
    *,
    workers: int = 1,
    force: bool = False,
    progress=print,
) -> SmoothingArtifact:
    """Smooth ``tasks`` into ``output_dir/wishes.jsonl``; resume from existing records."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / ARTIFACT_NAME
    if force and artifact_path.exists():
        artifact_path.unlink()
    existing = {} if force else _read_existing(artifact_path)

    pending = [task for task in tasks if task.key not in existing]
    fallback_count = sum(1 for record in existing.values() if record.get("fallback"))
    done_count = len(existing)
    lock = threading.Lock()
    failures: list[str] = []

    with artifact_path.open("a", encoding="utf-8") as handle:

        def job(task):
            record = _smooth_one(task, llm)
            line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            with lock:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            return record

        if pending:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {executor.submit(job, task): task for task in pending}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001 - keep going, report at end
                        failures.append(f"{task.key}: {type(exc).__name__}: {exc}")
                        progress(f"smoothing FAILED {task.key}: {exc}")
                        continue
                    done_count += 1
                    if record["fallback"]:
                        fallback_count += 1
                    if done_count % 10 == 0 or done_count == len(tasks):
                        progress(
                            f"smoothing [{done_count}/{len(tasks)}] {task.key} "
                            f"fallback={record['fallback']}"
                        )
        elif tasks:
            progress(f"all {len(tasks)} selected wishes already smoothed; nothing to do")

    records = _read_existing(artifact_path)
    ordered = [records[key] for key in sorted(records)]
    atomic_write_text(
        artifact_path,
        "\n".join(json.dumps(record, sort_keys=True, ensure_ascii=False) for record in ordered)
        + ("\n" if ordered else ""),
    )
    artifact = load_smoothing(artifact_path)
    selected_keys = {task.key for task in tasks}
    covered = selected_keys & set(artifact.records)
    complete = covered == selected_keys
    manifest = {
        "format_version": 1,
        "phase": "R2",
        "created_at": utc_now_iso(),
        "git": git_info(),
        "environment": environment_info(),
        "endpoint": {
            **llm.config.safe_identity,
            "temperature": llm.temperature,
            "top_p": llm.top_p,
        },
        "prompt": {
            "system": SMOOTH_SYSTEM,
        },
        "splits_requested": sorted({task.split for task in tasks}),
        "tasks_requested": len(tasks),
        "tasks_completed": len(covered),
        "tasks_missing": sorted(selected_keys - covered)[:20],
        "complete": bool(complete and not failures),
        "failures": failures[:20],
        "fallbacks": fallback_count,
        "artifact": {
            "file": ARTIFACT_NAME,
            "sha256": artifact.sha256,
            "records": len(artifact.records),
        },
    }
    atomic_write_json(output_dir / MANIFEST_NAME, manifest)
    # Reload so the returned artifact carries the manifest just written (the JSONL hash
    # cannot change when only the sibling manifest is added).
    artifact = load_smoothing(artifact_path)
    if failures:
        raise RuntimeError(
            f"{len(failures)} wish(es) failed to smooth; see "
            f"{output_dir / MANIFEST_NAME}: {failures[:3]}"
        )
    if not complete:
        missing = sorted(selected_keys - covered)
        raise RuntimeError(
            f"smoothing incomplete: {len(missing)} selected task(s) have no record "
            f"({', '.join(missing[:5])}); rerun to resume"
        )
    progress(
        f"smoothing complete: {len(covered)} records, {fallback_count} fallbacks, "
        f"artifact {artifact_path}"
    )
    progress(f"artifact sha256: {artifact.sha256}")
    return artifact


def _select_tasks(
    splits: list[str],
    *,
    tasks_path: Path | None = None,
    limit: int | None = None,
) -> list:
    if "heldout" in splits:
        raise SystemExit("refusing to smooth the held-out split; that split belongs to R3")
    unknown = sorted(set(splits) - set(SMOOTHING_SPLITS))
    if unknown:
        raise SystemExit(
            f"unknown smoothing split(s) {unknown}; choose from {list(SMOOTHING_SPLITS)}"
        )
    tasks = []
    for split in splits:
        tasks.extend(load_tasks(split=split, tasks_path=tasks_path, require_derived=False))
    tasks.sort(key=lambda task: task.key)
    if limit is not None:
        tasks = tasks[: max(0, limit)]
    if not tasks:
        raise SystemExit("no tasks selected for smoothing")
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SMOOTHING_SPLITS),
        help="splits to smooth (default: train pilot)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="smoke-test only")
    parser.add_argument("--tasks", nargs="*", default=None, help="explicit task keys")
    parser.add_argument("--force", action="store_true", help="re-smooth completed tasks")
    args = parser.parse_args(argv)

    tasks = _select_tasks(args.splits, limit=args.limit)
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [task for task in tasks if task.key in wanted]
        found = {task.key for task in tasks}
        missing = wanted - found
        if missing:
            raise SystemExit(f"task keys not found: {sorted(missing)}")
    output = Path(args.output) if args.output else default_smoothing_dir(args.run_id)

    counts = {split: sum(1 for task in tasks if task.split == split) for split in args.splits}
    print(
        f"R2 L3 smoothing: run_id={args.run_id} splits={counts} workers={args.workers}",
        flush=True,
    )
    llm = LLMClient(generator_config(), temperature=0.0, top_p=1.0)
    print(f"endpoint: {llm.config.safe_identity}", flush=True)
    print(f"sampling: temperature={llm.temperature}, top_p={llm.top_p}", flush=True)
    print(f"output: {output}", flush=True)
    if "train" in counts and counts["train"] != TRAIN_TARGET:
        print(
            f"warning: {counts['train']} train tasks selected, expected {TRAIN_TARGET}; "
            "the smoothed detector refit needs all 500",
            flush=True,
        )
    if "pilot" in counts and counts["pilot"] != PILOT_TARGET:
        print(
            f"warning: {counts['pilot']} pilot tasks selected, expected {PILOT_TARGET}",
            flush=True,
        )

    smooth_tasks(
        tasks,
        llm,
        output,
        workers=max(1, args.workers),
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PILOT_TARGET",
    "SMOOTHING_SPLITS",
    "TRAIN_TARGET",
    "main",
    "smooth_tasks",
]
