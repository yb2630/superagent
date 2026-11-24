"""Aggregate R2 pilot units into pass@1, recovery, per-category recovery and buckets.

The runner writes one JSON unit per (condition, task, sample) and a per-directory summary.
The pilot may be split across shards or even several run directories (one per condition) to
fit a 12-hour Colab session, so this module takes any number of run directories, unions
their completed units by unit key, verifies the task count/sample count and held-out
protection, and writes a combined ``summary.json``/``report.md``.

Recovery is the fraction of the B1 (L0) to H0 (L3) gap closed by a condition, using
pass@1 (for B2 this is the first of its five direct samples):

    recovery(c) = (pass@1(c) - pass@1(H0)) / (pass@1(B1) - pass@1(H0))

Per-category recovery groups tasks by the category that their L0 carries and L3 omits
(``field_presence[category]``); the 9-condition pilot does not run the separate L2_c
single-field-drop conditions, so this grouping is the attribution available without extra
GPU hours.  ``intent`` is omitted from that table because L3 always keeps it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import FIELD_LABELS, LEVELS_DIR
from .conditions import CONDITION_ORDER
from .config import atomic_write_json, atomic_write_text, read_jsonl, utc_now_iso

CATEGORY_ATTRIBUTION = (
    "signature_types",
    "examples",
    "constraints",
    "error_behaviour",
)


def load_tasks(tasks_path: Path | None = None) -> dict[str, dict]:
    records = read_jsonl(tasks_path or (LEVELS_DIR / "tasks.jsonl"))
    return {str(record["key"]): record for record in records}


def _unit_key(unit: dict) -> tuple[str, str, int]:
    return (
        str(unit.get("condition", "")),
        str(unit.get("task_key", "")),
        int(unit.get("sample_index", 0) or 0),
    )


def load_runs(run_dirs: list[Path]) -> tuple[dict[tuple[str, str, int], dict], list[dict]]:
    units: dict[tuple[str, str, int], dict] = {}
    configs: list[dict] = []
    for run_dir in run_dirs:
        manifest_path = Path(run_dir) / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{run_dir}: manifest.json is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("config") or {}
        configs.append(config)
        for path in sorted((Path(run_dir) / "units").rglob("*.json")):
            try:
                unit = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if unit.get("status") != "complete":
                continue
            units.setdefault(_unit_key(unit), unit)
    return units, configs


def _union_config(configs: list[dict]) -> dict:
    if not configs:
        raise ValueError("no run configs")
    task_lists = [list(config.get("task_keys") or []) for config in configs]
    base_tasks = task_lists[0]
    for other in task_lists[1:]:
        if set(other) != set(base_tasks):
            raise ValueError("run directories disagree on the task set")
    conditions: list[str] = []
    for name in CONDITION_ORDER:
        if any(name in (config.get("conditions") or []) for config in configs):
            conditions.append(name)
    for config in configs:
        for name in config.get("conditions") or []:
            if name not in conditions:
                conditions.append(str(name))
    samples = max(
        [
            max((int(value) for value in config.get("sample_indices") or [0]), default=0)
            for config in configs
        ],
        default=0,
    )
    return {
        "task_keys": sorted(base_tasks),
        "conditions": conditions,
        "samples": samples + 1,
    }


def _summarise_condition(unit_list: list[dict], task_keys: set[str]) -> dict:
    selected = [unit for unit in unit_list if unit.get("task_key") in task_keys]
    passes = sum(1 for unit in selected if unit.get("pass"))
    pass_at_1 = sum(1 for unit in selected if unit.get("pass_at_1", unit.get("pass")))
    buckets: dict[str, int] = {}
    for unit in selected:
        if unit.get("pass"):
            continue
        bucket = str(unit.get("bucket") or "unknown")
        buckets[bucket] = buckets.get(bucket, 0) + 1
    turns = [int(unit.get("turns", 0) or 0) for unit in selected]
    questions = [int(unit.get("questions", 0) or 0) for unit in selected]
    totals = {
        "input_tokens": sum(
            int(unit.get("totals", {}).get("input_tokens", 0) or 0) for unit in selected
        ),
        "output_tokens": sum(
            int(unit.get("totals", {}).get("output_tokens", 0) or 0) for unit in selected
        ),
    }
    return {
        "units": len(selected),
        "passes": passes,
        "pass_rate": round(passes / len(selected), 4) if selected else None,
        "pass_at_1": round(pass_at_1 / len(selected), 4) if selected else None,
        "avg_turns": round(sum(turns) / len(turns), 2) if turns else None,
        "avg_questions": round(sum(questions) / len(questions), 2) if questions else None,
        "turn_cap_units": sum(1 for unit in selected if unit.get("turn_cap_hit")),
        "buckets": dict(sorted(buckets.items())),
        "tokens": totals,
    }


def _recovery(pass_at_1: float | None, baseline: dict) -> float | None:
    if pass_at_1 is None:
        return None
    h0 = baseline.get("H0", {}).get("pass_at_1")
    b1 = baseline.get("B1", {}).get("pass_at_1")
    if h0 is None or b1 is None or b1 <= h0:
        return None
    return round((pass_at_1 - h0) / (b1 - h0), 4)


def build_report(run_dirs: list[Path], *, tasks_path: Path | None = None) -> dict:
    units, configs = load_runs([Path(path) for path in run_dirs])
    config = _union_config(configs)
    task_keys = set(config["task_keys"])
    tasks = load_tasks(tasks_path)
    missing_metadata = sorted(key for key in task_keys if key not in tasks)
    if missing_metadata:
        raise KeyError(f"tasks missing from the levels artifact: {missing_metadata[:5]}")
    heldout = {key for key, record in tasks.items() if record.get("split") == "heldout"}
    # Check the units themselves, not just the manifest task list: a stray unit written
    # outside the configured set must still be reported as held-out usage.
    heldout_used = sorted({str(unit.get("task_key")) for unit in units.values()} & heldout)
    unit_tasks = {str(unit.get("task_key")) for unit in units.values()}
    unknown_tasks = sorted(unit_tasks - task_keys - heldout)
    units = {key: unit for key, unit in units.items() if str(unit.get("task_key")) in task_keys}

    conditions = config["conditions"]
    expected = len(task_keys) * len(conditions) * config["samples"]
    per_condition: dict[str, dict] = {}
    for condition in conditions:
        per_condition[condition] = _summarise_condition(
            [unit for unit in units.values() if unit.get("condition") == condition],
            task_keys,
        )
    for condition in conditions:
        per_condition[condition]["recovery"] = _recovery(
            per_condition[condition]["pass_at_1"], per_condition
        )

    by_category: dict[str, dict] = {}
    for category in CATEGORY_ATTRIBUTION:
        subset = {
            key for key in task_keys if bool(tasks[key].get("field_presence", {}).get(category))
        }
        item = {"label": FIELD_LABELS[category], "tasks": len(subset), "conditions": {}}
        for condition in conditions:
            item["conditions"][condition] = _summarise_condition(
                [unit for unit in units.values() if unit.get("condition") == condition],
                subset,
            )
            item["conditions"][condition]["recovery"] = _recovery(
                item["conditions"][condition]["pass_at_1"], item["conditions"]
            )
        by_category[category] = item

    manifests = []
    wall_clock = 0.0
    for run_dir in run_dirs:
        path = Path(run_dir) / "manifest.json"
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        shards = manifest.get("shards") or []
        shard_seconds = sum(
            float(entry.get("duration_s") or 0.0) for entry in shards if isinstance(entry, dict)
        )
        manifests.append(
            {
                "run_dir": str(run_dir),
                "run_id": manifest.get("run_id"),
                "status": manifest.get("status"),
                "duration_s": manifest.get("duration_s"),
                "shard_seconds": round(shard_seconds, 1),
                "planned_units": manifest.get("planned_units"),
                "shard": manifest.get("shard"),
            }
        )
        wall_clock += shard_seconds or float(manifest.get("duration_s") or 0.0)

    bucket_totals: dict[str, int] = {}
    for item in per_condition.values():
        for bucket, count in item["buckets"].items():
            bucket_totals[bucket] = bucket_totals.get(bucket, 0) + count
    token_totals = {
        "input_tokens": sum(item["tokens"]["input_tokens"] for item in per_condition.values()),
        "output_tokens": sum(item["tokens"]["output_tokens"] for item in per_condition.values()),
    }
    unit_duration = sum(float(unit.get("duration_s") or 0.0) for unit in units.values())
    missing_units = {
        condition: {
            "expected": len(task_keys) * config["samples"],
            "completed": per_condition[condition]["units"],
        }
        for condition in conditions
        if per_condition[condition]["units"] < len(task_keys) * config["samples"]
    }
    summary = {
        "format_version": 1,
        "generated_at": utc_now_iso(),
        "run_dirs": [str(path) for path in run_dirs],
        "tasks": len(task_keys),
        "conditions": conditions,
        "samples": config["samples"],
        "expected_units": expected,
        "completed_units": len(units),
        "missing_units": missing_units,
        "complete": len(units) >= expected and not missing_units,
        "heldout_tasks_used": len(heldout_used),
        "heldout_examples": heldout_used[:5],
        "unknown_task_units": len(unknown_tasks),
        "unknown_task_examples": unknown_tasks[:5],
        "conditions_summary": per_condition,
        "recovery_baselines": {
            "B1": per_condition.get("B1", {}).get("pass_rate"),
            "H0": per_condition.get("H0", {}).get("pass_rate"),
            "definition": "(pass@1(condition) - pass@1(H0)) / (pass@1(B1) - pass@1(H0))",
        },
        "by_category": by_category,
        "failure_buckets": dict(sorted(bucket_totals.items())),
        "cost": {
            "input_tokens": token_totals["input_tokens"],
            "output_tokens": token_totals["output_tokens"],
            "total_tokens": token_totals["input_tokens"] + token_totals["output_tokens"],
            "estimated_cost_usd": 0.0,
            "pricing": "local vLLM endpoint; $0 paid API is a run constraint",
        },
        "time": {
            "wall_clock_s": round(wall_clock, 1),
            "wall_clock_h": round(wall_clock / 3600.0, 2),
            "unit_duration_h": round(unit_duration / 3600.0, 2),
            "gpu_hours_estimate": round(wall_clock / 3600.0, 2),
        },
        "manifests": manifests,
        "notes": [
            "Recovery is computed from measured units only; B1 and H0 are the ceiling and floor.",
            (
                "Per-category recovery groups tasks by the L0 category that the wish omits; "
                "the pilot does not run separate L2_c single-field-drop conditions."
            ),
            "The 300 held-out tasks must not appear in any pilot unit.",
        ],
    }
    summary["report"] = render_report(summary)
    return summary


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_report(summary: dict) -> str:
    lines = [
        "# R2 GPU pilot report",
        "",
        f"- Generated: {summary['generated_at']}",
        (
            f"- Tasks: {summary['tasks']}; conditions: {len(summary['conditions'])}; "
            f"samples per condition: {summary['samples']}"
        ),
        f"- Units: {summary['completed_units']}/{summary['expected_units']}"
        + (" (complete)" if summary["complete"] else " (INCOMPLETE)"),
        f"- Held-out tasks used: {summary['heldout_tasks_used']}",
        (
            f"- Tokens: {summary['cost']['total_tokens']:,} "
            f"(input {summary['cost']['input_tokens']:,}, "
            f"output {summary['cost']['output_tokens']:,}); paid API cost: $0.00"
        ),
        (
            f"- Wall clock: {summary['time']['wall_clock_h']} h "
            f"({summary['time']['gpu_hours_estimate']} GPU-h estimate); "
            f"unit time {summary['time']['unit_duration_h']} h"
        ),
        "",
        "| condition | pass@1 | pass rate | recovery | questions | turns | cap hits |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for condition in summary["conditions"]:
        item = summary["conditions_summary"][condition]
        lines.append(
            f"| {condition} | {_fmt(item['pass_at_1'])} | {_fmt(item['pass_rate'])} | "
            f"{_fmt(item['recovery'])} | {_fmt(item['avg_questions'], 2)} | "
            f"{_fmt(item['avg_turns'], 2)} | {item['turn_cap_units']} |"
        )
    lines.extend(
        [
            "",
            "## Per-category recovery",
            "",
            "Tasks grouped by a category their L0 carries (so L3 omits it). The 9-condition",
            "pilot does not run the separate L2_c single-field-drop conditions.",
            "",
            "| category | tasks | B1 pass | H0 pass | H3 pass | H3 recovery |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for category, item in summary["by_category"].items():
        b1 = item["conditions"].get("B1", {}).get("pass_rate")
        h0 = item["conditions"].get("H0", {}).get("pass_rate")
        h3 = item["conditions"].get("H3", {}).get("pass_rate")
        recovery = item["conditions"].get("H3", {}).get("recovery")
        lines.append(
            f"| {category} | {item['tasks']} | {_fmt(b1)} | {_fmt(h0)} | {_fmt(h3)} | "
            f"{_fmt(recovery)} |"
        )
    lines.extend(
        [
            "",
            "## Failure buckets",
            "",
        ]
    )
    if summary["failure_buckets"]:
        for bucket, count in summary["failure_buckets"].items():
            lines.append(f"- `{bucket}`: {count}")
    else:
        lines.append("- none")
    lines.append("")
    if summary["missing_units"]:
        lines.append("## Missing units")
        lines.append("")
        for condition, item in summary["missing_units"].items():
            lines.append(f"- {condition}: {item['completed']}/{item['expected']} completed")
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            *[f"- {note}" for note in summary["notes"]],
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directory or directories")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit non-zero unless every expected unit is present and held-out is absent",
    )
    args = parser.parse_args(argv)

    summary = build_report(args.runs, tasks_path=args.tasks)
    output = Path(args.output) if args.output else Path(args.runs[0])
    atomic_write_json(output / "summary.json", summary)
    atomic_write_text(output / "report.md", summary["report"])
    print(summary["report"])
    print(f"wrote {output / 'summary.json'} and {output / 'report.md'}")
    if args.require_complete and (not summary["complete"] or summary["heldout_tasks_used"]):
        print("ERROR: pilot is incomplete or used a held-out task")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATEGORY_ATTRIBUTION",
    "build_report",
    "load_runs",
    "load_tasks",
    "main",
    "render_report",
]
