from __future__ import annotations

import json

from wishrec.harness.report import build_report, main


def _tasks_file(tmp_path):
    records = [
        {
            "key": "pilot/a",
            "split": "pilot",
            "source": "humanevalplus",
            "field_presence": {
                "intent": True,
                "signature_types": True,
                "examples": True,
                "constraints": False,
                "error_behaviour": False,
            },
        },
        {
            "key": "pilot/b",
            "split": "pilot",
            "source": "mbppplus",
            "field_presence": {
                "intent": True,
                "signature_types": True,
                "examples": False,
                "constraints": True,
                "error_behaviour": False,
            },
        },
        {
            "key": "heldout/secret",
            "split": "heldout",
            "source": "humanevalplus",
            "field_presence": {
                "intent": True,
                "signature_types": True,
                "examples": True,
                "constraints": True,
                "error_behaviour": True,
            },
        },
    ]
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def _unit(condition, task_key, sample, passed, bucket="AssertionError"):
    return {
        "condition": condition,
        "task_key": task_key,
        "sample_index": sample,
        "pass": passed,
        "pass_at_1": passed,
        "bucket": "None" if passed else bucket,
        "turns": 3,
        "questions": 2 if condition == "H3" else 0,
        "turn_cap_hit": False,
        "totals": {"input_tokens": 100, "output_tokens": 20},
        "duration_s": 1.5,
        "status": "complete",
    }


def _write_run(
    root,
    *,
    conditions=("B1", "H0", "H3"),
    tasks=("pilot/a", "pilot/b"),
    samples=2,
    passes=None,
    omit=None,
    task_keys=None,
):
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": root.name,
        "config": {
            "run_id": root.name,
            "task_keys": list(task_keys or tasks),
            "conditions": list(conditions),
            "sample_indices": list(range(samples)),
        },
        "duration_s": 12.0,
        "status": "complete",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for condition in conditions:
        for sample in range(samples):
            for task in tasks:
                if omit and (condition, task, sample) in omit:
                    continue
                passed = (passes or {}).get((condition, task, sample), False)
                path = root / "units" / condition / str(sample) / f"{task.replace('/', '__')}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(_unit(condition, task, sample, passed)),
                    encoding="utf-8",
                )
    return root


PASSES = {
    ("B1", "pilot/a", 0): True,
    ("B1", "pilot/a", 1): True,
    ("B1", "pilot/b", 0): True,
    ("B1", "pilot/b", 1): True,
    ("H0", "pilot/a", 0): False,
    ("H0", "pilot/a", 1): False,
    ("H0", "pilot/b", 0): True,
    ("H0", "pilot/b", 1): False,
    ("H3", "pilot/a", 0): True,
    ("H3", "pilot/a", 1): False,
    ("H3", "pilot/b", 0): True,
    ("H3", "pilot/b", 1): True,
}


def test_report_computes_pass_recovery_and_per_category_recovery(tmp_path):
    tasks = _tasks_file(tmp_path)
    run = _write_run(tmp_path / "run", passes=PASSES)
    summary = build_report([run], tasks_path=tasks)
    assert summary["complete"] is True
    assert summary["expected_units"] == 12
    assert summary["completed_units"] == 12
    assert summary["heldout_tasks_used"] == 0

    by_condition = summary["conditions_summary"]
    assert by_condition["B1"]["pass_rate"] == 1.0
    assert by_condition["H0"]["pass_rate"] == 0.25
    assert by_condition["H3"]["pass_rate"] == 0.75
    assert by_condition["H3"]["recovery"] == 0.6667
    assert by_condition["B1"]["recovery"] == 1.0
    assert by_condition["H0"]["recovery"] == 0.0

    by_category = summary["by_category"]
    assert by_category["examples"]["tasks"] == 1
    assert by_category["examples"]["conditions"]["H3"]["recovery"] == 0.5
    assert by_category["constraints"]["tasks"] == 1
    assert by_category["constraints"]["conditions"]["H3"]["recovery"] == 1.0
    assert summary["failure_buckets"] == {"AssertionError": 4}
    assert summary["cost"]["total_tokens"] == 12 * 120
    assert summary["cost"]["estimated_cost_usd"] == 0.0
    assert summary["time"]["wall_clock_h"] == 0.0
    assert "R2 GPU pilot report" in summary["report"]
    assert "| H3 |" in summary["report"]


def test_report_aggregates_condition_chunks_and_reports_missing_units(tmp_path):
    tasks = _tasks_file(tmp_path)
    run_a = _write_run(
        tmp_path / "run-b1",
        conditions=("B1",),
        passes=PASSES,
        task_keys=("pilot/a", "pilot/b"),
    )
    run_b = _write_run(
        tmp_path / "run-h0",
        conditions=("H0",),
        passes=PASSES,
        task_keys=("pilot/a", "pilot/b"),
        omit={("H0", "pilot/b", 1)},
    )
    summary = build_report([run_a, run_b], tasks_path=tasks)
    assert summary["conditions"] == ["B1", "H0"]
    assert summary["expected_units"] == 8
    assert summary["completed_units"] == 7
    assert summary["complete"] is False
    assert summary["missing_units"] == {"H0": {"expected": 4, "completed": 3}}
    assert "INCOMPLETE" in summary["report"]


def test_report_flags_a_heldout_task_in_the_run(tmp_path):
    tasks = _tasks_file(tmp_path)
    run = _write_run(
        tmp_path / "run",
        conditions=("B1",),
        tasks=("pilot/a", "heldout/secret"),
        samples=1,
        passes={("B1", "pilot/a", 0): True, ("B1", "heldout/secret", 0): True},
    )
    summary = build_report([run], tasks_path=tasks)
    assert summary["heldout_tasks_used"] == 1
    assert summary["heldout_examples"] == ["heldout/secret"]


def test_report_cli_writes_summary_and_can_require_complete(tmp_path):
    tasks = _tasks_file(tmp_path)
    run = _write_run(tmp_path / "run", passes=PASSES)
    code = main([str(run), "--tasks", str(tasks), "--require-complete"])
    assert code == 0
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True
    assert (run / "report.md").is_file()

    incomplete = _write_run(
        tmp_path / "partial", conditions=("B1",), samples=1, omit={("B1", "pilot/a", 0)}
    )
    assert main([str(incomplete), "--tasks", str(tasks), "--require-complete"]) == 1


def test_report_flags_a_heldout_unit_outside_the_config_window(tmp_path):
    tasks = _tasks_file(tmp_path)
    run = _write_run(
        tmp_path / "run",
        conditions=("B1",),
        samples=1,
        passes={("B1", "pilot/a", 0): True, ("B1", "pilot/b", 0): True},
    )
    stray = run / "units" / "B1" / "0" / "heldout__secret.json"
    stray.write_text(json.dumps(_unit("B1", "heldout/secret", 0, True)), encoding="utf-8")
    summary = build_report([run], tasks_path=tasks)
    assert summary["heldout_tasks_used"] == 1
    assert summary["heldout_examples"] == ["heldout/secret"]
    # The stray unit is not counted toward the configured pilot window.
    assert summary["completed_units"] == 2
    assert summary["complete"] is True
