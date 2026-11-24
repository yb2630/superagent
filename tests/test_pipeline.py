from __future__ import annotations

import json

import pytest

from fakes import FakeDetector, FakeEndpoint, generic_response, make_task, text_response
from wishrec.harness import runner
from wishrec.harness.conditions import resolve_conditions
from wishrec.harness.config import EndpointConfig
from wishrec.harness.llm import LLMClient
from wishrec.harness.smoothing import smooth_tasks

SMOOTH_SYSTEM_MARKER = "Rewrite the following wish"


def _handler(payload):
    messages = payload.get("messages") or []
    system = str(messages[0].get("content", "")) if messages else ""
    if not payload.get("tools") and SMOOTH_SYSTEM_MARKER in system:
        return text_response("A smoothed wish for the pilot.")
    return generic_response(payload)


def _tasks(count: int = 5, *, split: str = "pilot"):
    return [
        make_task(
            key=f"script/add-{index}",
            source_task_id=f"add-{index}",
            entry_point="add",
            split=split,
        )
        for index in range(count)
    ]


def _client(endpoint) -> LLMClient:
    return LLMClient(
        EndpointConfig(base_url=endpoint.base_url, model="fake", timeout=10),
        temperature=0.0,
    )


def test_notebook_pipeline_smooths_then_runs_all_conditions_through_a_fake_endpoint(
    tmp_path, monkeypatch
):
    """The corrected notebook's local path: smooth -> overlay -> 9-condition dry run -> summary."""
    tasks = _tasks()
    with FakeEndpoint(handler=_handler) as endpoint:
        monkeypatch.setenv("VLLM_BASE_URL", endpoint.base_url)
        monkeypatch.setenv("VLLM_MODEL", "fake")
        monkeypatch.setenv("VLLM_API_KEY", "EMPTY")
        monkeypatch.setenv("VLLM_TIMEOUT", "10")
        monkeypatch.setenv("VLLM_MAX_RETRIES", "1")
        smoothing_dir = tmp_path / "smoothing"
        artifact = smooth_tasks(
            tasks,
            _client(endpoint),
            smoothing_dir,
            workers=2,
            progress=lambda message: None,
        )
        assert artifact.sha256
        monkeypatch.setattr(runner, "load_default_detector", lambda: FakeDetector())
        result = runner.run_units(
            tasks=tasks,
            conditions=resolve_conditions(None),
            run_id="fake-dry-run",
            output_dir=tmp_path / "run",
            samples=1,
            workers=1,
            smoothing_path=smoothing_dir / "wishes.jsonl",
            force=True,
            progress=lambda message: None,
        )
    summary = result["summary"]
    assert summary["units"] == 45
    assert summary["complete"] is True
    assert summary["status"] == "complete"
    assert summary["smoothing"]["sha256"] == artifact.sha256
    assert result["manifest"]["config"]["smoothing"]["sha256"] == artifact.sha256
    assert result["manifest"]["units_planned"] == 45
    assert set(summary["conditions"]) == {
        "B1",
        "H0",
        "H1",
        "H2",
        "H3",
        "H4",
        "B2",
        "B3",
        "DISCARD",
    }
    for condition in summary["conditions"].values():
        assert condition["units"] == 5
        assert condition["passes"] == 5

    h0 = json.loads(
        (tmp_path / "run" / "units" / "H0" / "0" / "script__add-0.json").read_text(encoding="utf-8")
    )
    assert "A smoothed wish for the pilot." in json.dumps(h0["messages"])
    b1 = json.loads(
        (tmp_path / "run" / "units" / "B1" / "0" / "script__add-0.json").read_text(encoding="utf-8")
    )
    assert "A smoothed wish for the pilot." not in json.dumps(b1["messages"])
    assert "Intent:" in json.dumps(b1["messages"])


def test_shards_are_config_stable_and_resume_per_unit(tmp_path, monkeypatch):
    tasks = _tasks()
    with FakeEndpoint(handler=_handler) as endpoint:
        monkeypatch.setenv("VLLM_BASE_URL", endpoint.base_url)
        monkeypatch.setenv("VLLM_MODEL", "fake")
        monkeypatch.setenv("VLLM_API_KEY", "EMPTY")
        monkeypatch.setenv("VLLM_TIMEOUT", "10")
        monkeypatch.setenv("VLLM_MAX_RETRIES", "1")
        smoothing_dir = tmp_path / "smoothing"
        smooth_tasks(
            tasks,
            _client(endpoint),
            smoothing_dir,
            progress=lambda message: None,
        )
        monkeypatch.setattr(runner, "load_default_detector", lambda: FakeDetector())
        kwargs = dict(
            tasks=tasks,
            conditions=resolve_conditions(None),
            run_id="sharded",
            output_dir=tmp_path / "run",
            samples=1,
            workers=1,
            smoothing_path=smoothing_dir / "wishes.jsonl",
            progress=lambda message: None,
        )
        first = runner.run_units(shard=0, shards=2, **kwargs)
        assert first["summary"]["units"] == 23
        assert first["summary"]["complete"] is False
        assert first["manifest"]["status"] == "partial"
        assert len((tmp_path / "run" / "results.jsonl").read_text().splitlines()) == 23
        requests_after_first = len(endpoint.requests)

        second = runner.run_units(shard=1, shards=2, **kwargs)
        assert second["summary"]["units"] == 45
        assert second["summary"]["complete"] is True
        assert second["manifest"]["status"] == "complete"
        assert len(second["manifest"]["shards"]) == 2
        assert len((tmp_path / "run" / "results.jsonl").read_text().splitlines()) == 45

        requests_after_second = len(endpoint.requests)
        resumed = runner.run_units(shard=0, shards=2, **kwargs)
        assert resumed["summary"]["units"] == 45
        assert resumed["manifest"]["units_skipped"] == 23
        assert len(endpoint.requests) == requests_after_second
        assert len((tmp_path / "run" / "results.jsonl").read_text().splitlines()) == 45

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["units_planned"] == 45
    assert manifest["units_completed"] == 45


def test_smoothing_overlay_refuses_a_task_without_a_smoothed_wish(tmp_path, monkeypatch):
    tasks = _tasks()
    with FakeEndpoint(handler=_handler) as endpoint:
        monkeypatch.setenv("VLLM_BASE_URL", endpoint.base_url)
        monkeypatch.setenv("VLLM_MODEL", "fake")
        monkeypatch.setenv("VLLM_API_KEY", "EMPTY")
        monkeypatch.setenv("VLLM_TIMEOUT", "10")
        monkeypatch.setenv("VLLM_MAX_RETRIES", "1")
        smoothing_dir = tmp_path / "smoothing"
        smooth_tasks(
            tasks,
            _client(endpoint),
            smoothing_dir,
            progress=lambda message: None,
        )
        monkeypatch.setattr(runner, "load_default_detector", lambda: FakeDetector())
        with pytest.raises(KeyError, match="missing 1 task"):
            runner.run_units(
                tasks=tasks + [make_task(key="script/extra", source_task_id="extra")],
                conditions=resolve_conditions(["H0"]),
                run_id="missing-smoothing",
                output_dir=tmp_path / "run",
                samples=1,
                smoothing_path=smoothing_dir / "wishes.jsonl",
                progress=lambda message: None,
            )
