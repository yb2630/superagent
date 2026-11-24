from __future__ import annotations

import json

import pytest

from fakes import FakeEndpoint, make_task, text_response
from wishrec.detector.data import train_dataset
from wishrec.harness.config import EndpointConfig
from wishrec.harness.llm import LLMClient
from wishrec.harness.smoothing import smooth_tasks
from wishrec.smoothing import load_smoothing


def _client(endpoint, model="Qwen/Qwen3-8B-AWQ") -> LLMClient:
    return LLMClient(
        EndpointConfig(base_url=endpoint.base_url, model=model, timeout=10),
        temperature=0.0,
        top_p=1.0,
    )


def _handler(payload):
    messages = payload.get("messages") or []
    user = str(messages[-1].get("content", "")) if messages else ""
    return text_response(f"Smoothed: {user.strip()}")


def test_smoothing_is_resumable_and_records_provenance(tmp_path):
    tasks = [
        make_task(key="test/add-0", source_task_id="0", split="train"),
        make_task(key="test/add-1", source_task_id="1", split="pilot"),
    ]
    with FakeEndpoint(handler=_handler, model="Qwen/Qwen3-8B-AWQ") as endpoint:
        output = tmp_path / "smoothing"
        first = smooth_tasks(
            tasks, _client(endpoint), output, workers=2, progress=lambda message: None
        )
        assert len(endpoint.requests) == 2
        assert len(first.records) == 2
        assert first.splits == {"pilot": 1, "train": 1}
        requests_after_first = len(endpoint.requests)

        second = smooth_tasks(
            tasks, _client(endpoint), output, workers=1, progress=lambda message: None
        )
        assert len(endpoint.requests) == requests_after_first  # no repeat calls
        assert len(second.records) == 2

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["tasks_completed"] == 2
    assert manifest["endpoint"]["model"] == "Qwen/Qwen3-8B-AWQ"
    assert manifest["endpoint"]["temperature"] == 0.0
    assert manifest["endpoint"]["top_p"] == 1.0
    assert "api_key" not in json.dumps(manifest).lower()
    record = next(iter(second.records.values()))
    assert record["smoothed_wish"].startswith("Smoothed:")


def test_smoothing_falls_back_to_the_raw_wish_on_empty_output(tmp_path):
    tasks = [make_task(key="test/add", split="train")]
    with FakeEndpoint(handler=lambda payload: text_response(""), model="fake") as endpoint:
        artifact = smooth_tasks(
            tasks,
            _client(endpoint, model="fake"),
            tmp_path / "smoothing",
            progress=lambda message: None,
        )
    record = artifact.records["test/add"]
    assert record["fallback"] is True
    assert record["smoothed_wish"] == "Add two integers."


def test_loaded_smoothing_overlays_tasks_and_reports_missing_keys(tmp_path):
    tasks = [make_task(key="test/add", split="train")]
    with FakeEndpoint(handler=_handler, model="fake") as endpoint:
        artifact = smooth_tasks(
            tasks,
            _client(endpoint, model="fake"),
            tmp_path / "smoothing",
            progress=lambda message: None,
        )
    target = make_task(key="test/add", split="pilot")
    assert artifact.apply_to_tasks([target]) == 1
    assert target.levels["L3"].startswith("Smoothed:")
    assert target.levels["L0"].startswith("Intent:")
    other = make_task(key="test/other", source_task_id="other")
    with pytest.raises(KeyError, match="missing 1 task"):
        artifact.apply_to_tasks([other])


def _synthetic_record(index: int) -> dict:
    presence = {
        "intent": True,
        "signature_types": True,
        "examples": index % 2 == 0,
        "constraints": index % 3 == 0,
        "error_behaviour": index % 4 == 0,
    }
    return {
        "key": f"synthetic/train/{index}",
        "source": "humanevalplus",
        "source_task_id": str(index),
        "entry_point": "f",
        "split": "train",
        "fields": {label: f"{label} {index}" if presence[label] else "" for label in presence},
        "field_presence": presence,
        "levels": {"L0": "full", "L1": "l1", "L2": "l2", "L3": f"Wish {index}."},
        "stub": "def f():\n    pass\n",
        "visible_asserts": "assert True",
        "visible_asserts_count": 3,
        "visible_test_sha256": "0" * 64,
        "hidden_test_sha256": "0" * 64,
        "canonical_sha256": "0" * 64,
        "requirements": [],
        "meta": {},
    }


def test_detector_dataset_uses_smoothed_wishes_without_touching_the_artifact(tmp_path):
    records = [_synthetic_record(index) for index in range(500)]
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    tasks = [
        make_task(key=record["key"], source_task_id=str(index), split="train")
        for index, record in enumerate(records)
    ]
    smoothing_dir = tmp_path / "smoothing"
    with FakeEndpoint(handler=_handler, model="fake") as endpoint:
        artifact = smooth_tasks(
            tasks,
            _client(endpoint, model="fake"),
            smoothing_dir,
            workers=8,
            progress=lambda message: None,
        )

    dataset = train_dataset(tasks_path, smoothing_path=smoothing_dir / "wishes.jsonl")
    assert dataset["texts"][0] == artifact.wish(dataset["keys"][0])
    assert all(text.startswith("Smoothed:") for text in dataset["texts"])
    assert dataset["smoothing"]["sha256"] == artifact.sha256
    assert dataset["smoothing"]["records"] == 500
    # The committed-style raw record on disk is unchanged.
    raw = json.loads(tasks_path.read_text(encoding="utf-8").splitlines()[0])
    assert raw["levels"]["L3"].startswith("Wish ")


def test_detector_dataset_refuses_an_incomplete_smoothing_artifact(tmp_path):
    records = [_synthetic_record(index) for index in range(500)]
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    tasks = [
        make_task(key=record["key"], source_task_id=str(index), split="train")
        for index, record in enumerate(records)
    ]
    with FakeEndpoint(handler=_handler, model="fake") as endpoint:
        smooth_tasks(
            tasks[:2],
            _client(endpoint, model="fake"),
            tmp_path / "smoothing",
            progress=lambda message: None,
        )
    with pytest.raises(KeyError, match="missing 498"):
        train_dataset(tasks_path, smoothing_path=tmp_path / "smoothing" / "wishes.jsonl")
