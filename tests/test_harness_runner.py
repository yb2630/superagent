from __future__ import annotations

import json
from pathlib import Path

import pytest

from fakes import (
    FakeDetector,
    FakeEndpoint,
    generic_response,
    make_task,
)
from wishrec.harness import runner
from wishrec.harness.conditions import resolve_conditions


def test_sibling_contexts_exclude_the_target_and_use_examples():
    tasks = []
    for index in range(6):
        task = make_task(
            key=f"test/task-{index}",
            source_task_id=f"task-{index}",
            entry_point=f"f{index}",
            canonical_solution=f"def f{index}():\n    return {index}\n",
        )
        task.fields["examples"] = f">>> f{index}()\n{index}"
        tasks.append(task)
    contexts = runner.sibling_contexts(tasks)
    assert len(contexts) == 6
    for task in tasks:
        assert len(contexts[task.key]) == 5
        keys = {item["key"] for item in contexts[task.key]}
        assert task.key not in keys
        assert all(item["code"].startswith("def f") for item in contexts[task.key])
        assert all(item["doctests"].startswith(">>>") for item in contexts[task.key])


def test_sibling_contexts_with_one_task_is_empty():
    contexts = runner.sibling_contexts([make_task()])
    assert contexts == {"test/add": []}


def test_unit_path_is_task_relative(tmp_path):
    from wishrec.harness.conditions import get_condition

    path = runner.unit_path(tmp_path, get_condition("H0"), make_task(), 2)
    assert path == tmp_path / "units" / "H0" / "2" / "test__add.json"


def test_summarise_counts_passes_and_buckets(tmp_path):
    run_dir = tmp_path / "run"
    unit_dir = run_dir / "units" / "H0" / "0"
    unit_dir.mkdir(parents=True)
    (unit_dir / "a.json").write_text(
        json.dumps(
            {
                "condition": "H0",
                "task_key": "a",
                "pass": True,
                "bucket": "None",
                "turns": 3,
                "questions": 0,
                "totals": {"input_tokens": 1, "output_tokens": 1},
            }
        ),
        encoding="utf-8",
    )
    (unit_dir / "b.json").write_text(
        json.dumps(
            {
                "condition": "H0",
                "task_key": "b",
                "pass": False,
                "bucket": "AssertionError",
                "turns": 12,
                "questions": 0,
                "turn_cap_hit": True,
                "totals": {"input_tokens": 1, "output_tokens": 1},
            }
        ),
        encoding="utf-8",
    )
    summary, report = runner.summarise(run_dir, {"run_id": "x", "conditions": ["H0"]})
    assert summary["units"] == 2
    assert summary["conditions"]["H0"]["passes"] == 1
    assert summary["conditions"]["H0"]["turn_cap_units"] == 1
    assert summary["conditions"]["H0"]["buckets"] == {"AssertionError": 1}
    assert "| H0 | 1/2 |" in report


def test_runner_runs_all_nine_conditions_end_to_end_through_a_fake_endpoint(tmp_path, monkeypatch):
    with FakeEndpoint(handler=generic_response) as endpoint:
        monkeypatch.setenv("VLLM_BASE_URL", endpoint.base_url)
        monkeypatch.setenv("VLLM_API_KEY", "EMPTY")
        monkeypatch.setenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
        monkeypatch.setenv("VLLM_TOOL_MODE", "auto")
        monkeypatch.setenv("VLLM_TIMEOUT", "10")
        monkeypatch.setenv("VLLM_MAX_RETRIES", "1")
        monkeypatch.setattr(runner, "load_default_detector", lambda: FakeDetector())
        result = runner.run_units(
            tasks=[make_task()],
            conditions=resolve_conditions(None),
            run_id="test-run",
            output_dir=tmp_path / "run",
            samples=1,
            workers=1,
            force=True,
            progress=lambda message: None,
        )
        summary = result["summary"]
        assert summary["units"] == 9
        assert set(summary["conditions"]) == set(
            ("B1", "H0", "H1", "H2", "H3", "H4", "B2", "B3", "DISCARD")
        )
        for condition in summary["conditions"].values():
            assert condition["units"] == 1
            assert condition["passes"] == 1, condition
        assert result["manifest"]["status"] == "complete"
        assert (tmp_path / "run" / "manifest.json").is_file()
        assert (tmp_path / "run" / "summary.json").is_file()
        assert (tmp_path / "run" / "report.md").is_file()
        assert len(endpoint.requests) > 9


def test_runner_resumes_and_skips_cached_units(tmp_path, monkeypatch):
    with FakeEndpoint(handler=generic_response) as endpoint:
        monkeypatch.setenv("VLLM_BASE_URL", endpoint.base_url)
        monkeypatch.setenv("VLLM_TOOL_MODE", "auto")
        monkeypatch.setattr(runner, "load_default_detector", lambda: FakeDetector())
        kwargs = dict(
            tasks=[make_task()],
            conditions=resolve_conditions(["H0"]),
            run_id="resume-test",
            output_dir=tmp_path / "run",
            samples=1,
            workers=1,
            progress=lambda message: None,
        )
        first = runner.run_units(**kwargs)
        assert first["summary"]["units"] == 1
        requests_after_first = len(endpoint.requests)
        second = runner.run_units(**kwargs)
        assert second["manifest"]["units_skipped"] == 1
        assert len(endpoint.requests) == requests_after_first
        assert second["summary"]["conditions"]["H0"]["passes"] == 1


def test_runner_refuses_a_config_change_without_force(tmp_path, monkeypatch):
    with FakeEndpoint(handler=generic_response) as endpoint:
        monkeypatch.setenv("VLLM_BASE_URL", endpoint.base_url)
        monkeypatch.setenv("VLLM_TOOL_MODE", "auto")
        monkeypatch.setattr(runner, "load_default_detector", lambda: FakeDetector())
        kwargs = dict(
            tasks=[make_task()],
            run_id="mismatch",
            output_dir=tmp_path / "run",
            samples=1,
            workers=1,
            progress=lambda message: None,
        )
        runner.run_units(conditions=resolve_conditions(["H0"]), **kwargs)
        with pytest.raises(RuntimeError, match="exists with config"):
            runner.run_units(conditions=resolve_conditions(["H1"]), **kwargs)


def test_heldout_split_is_refused_for_a_pilot():
    class Args:
        split = "heldout"
        tasks = None
        limit = None

    with pytest.raises(SystemExit, match="held-out"):
        runner._task_selection(Args())


def test_detector_identity_is_semantic_not_a_file_timestamp(tmp_path):
    model = json.loads(
        (Path(__file__).resolve().parents[1] / "results/detector/model.json").read_text(
            encoding="utf-8"
        )
    )
    first = tmp_path / "first.json"
    first.write_text(json.dumps(model), encoding="utf-8")
    identity = runner.detector_identity(first)

    model["created_at"] = "2099-01-01T00:00:00+00:00"
    second = tmp_path / "second.json"
    second.write_text(json.dumps(model), encoding="utf-8")
    rerun = runner.detector_identity(second)
    assert rerun["sha256"] == identity["sha256"]
    assert rerun["file_sha256"] != identity["file_sha256"]

    model["threshold"] = float(model["threshold"]) + 0.1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(model), encoding="utf-8")
    assert runner.detector_identity(changed)["sha256"] != identity["sha256"]
