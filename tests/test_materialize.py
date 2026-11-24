from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from wishrec import materialize as mat


def _record(**overrides):
    record = {
        "key": "bigcodebench/BigCodeBench/103",
        "source": "bigcodebench",
        "source_task_id": "BigCodeBench/103",
    }
    record.update(overrides)
    return record


def _task():
    return SimpleNamespace(
        key="bigcodebench/BigCodeBench/103",
        source="bigcodebench",
        source_task_id="BigCodeBench/103",
        stub="def task_func():\n    pass\n",
        visible_test="def test_visible():\n    pass\n",
        hidden_test="def test_hidden():\n    pass\n",
        canonical_solution="def task_func():\n    return 1\n",
    )


def test_coverage_reports_missing_and_present(tmp_path: Path):
    record = _record()
    assert mat.coverage([record], tmp_path) == [record["key"]]
    directory = tmp_path / "bigcodebench" / "BigCodeBench_103"
    directory.mkdir(parents=True)
    (directory / "hidden_test.py").write_text("x", encoding="utf-8")
    assert mat.coverage([record], tmp_path) == []


def test_materialize_threads_the_visible_tests_mode(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(mat, "load_raw_rows", lambda: {"humanevalplus": [{}]})
    seen: list[dict] = []

    def fake_parse(source, row, **kwargs):
        seen.append(kwargs)
        return _task()

    monkeypatch.setattr(mat, "parse_task", fake_parse)
    mat.materialize([], derived_dir=tmp_path, visible_tests="full")
    mat.materialize([], derived_dir=tmp_path, visible_tests="inputs_only")

    assert seen == [{"visible_tests": "full"}, {"visible_tests": "inputs_only"}]


def test_materialize_defaults_to_the_mode_recorded_in_each_record(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(mat, "load_raw_rows", lambda: {"bigcodebench": [{}]})
    seen: list[str] = []

    def fake_parse(source, row, **kwargs):
        seen.append(kwargs["visible_tests"])
        return _task()

    monkeypatch.setattr(mat, "parse_task", fake_parse)
    mat.materialize([_record(meta={"visible_tests": "inputs_only"})], derived_dir=tmp_path)
    assert seen == ["inputs_only"]

    seen.clear()
    mat.materialize([], derived_dir=tmp_path)
    assert seen == ["inputs_only"], "the default must be the committed protocol"


def test_materialize_writes_derived_files_and_verifies_hashes(monkeypatch, tmp_path: Path):
    task = _task()
    record = _record(
        hidden_test_sha256=mat.sha256_text(task.hidden_test),
        visible_test_sha256=mat.sha256_text(task.visible_test),
        canonical_sha256=mat.sha256_text(task.canonical_solution),
        stub_sha256=mat.sha256_text(task.stub),
    )
    monkeypatch.setattr(mat, "load_raw_rows", lambda: {"bigcodebench": [{}]})
    monkeypatch.setattr(mat, "parse_task", lambda source, row, **kwargs: task)

    summary = mat.materialize([record], derived_dir=tmp_path)

    assert summary["written"] == 1
    assert summary["missing"] == []
    assert summary["mismatched"] == []
    assert summary["coverage_missing"] == []
    directory = tmp_path / "bigcodebench" / "BigCodeBench_103"
    assert (directory / "hidden_test.py").read_text(encoding="utf-8") == task.hidden_test
    assert (directory / "canonical_solution.py").exists()


def test_materialize_reports_a_hash_mismatch(monkeypatch, tmp_path: Path):
    task = _task()
    record = _record(hidden_test_sha256="0" * 64)
    monkeypatch.setattr(mat, "load_raw_rows", lambda: {"bigcodebench": [{}]})
    monkeypatch.setattr(mat, "parse_task", lambda source, row, **kwargs: task)

    summary = mat.materialize([record], derived_dir=tmp_path)

    assert summary["mismatched"]
    assert summary["mismatched"][0]["field"] == "hidden_test_sha256"

    assert not (tmp_path / "bigcodebench" / "BigCodeBench_103").exists(), (
        "a mismatching task must not be written"
    )


def _hashed_record(task):
    return _record(
        hidden_test_sha256=mat.sha256_text(task.hidden_test),
        visible_test_sha256=mat.sha256_text(task.visible_test),
        canonical_sha256=mat.sha256_text(task.canonical_solution),
        stub_sha256=mat.sha256_text(task.stub),
    )


def test_materialize_rejects_a_record_without_hashes(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(mat, "load_raw_rows", lambda: {"bigcodebench": [{}]})
    monkeypatch.setattr(mat, "parse_task", lambda source, row, **kwargs: _task())
    record = _hashed_record(_task())
    del record["stub_sha256"]

    summary = mat.materialize([record], derived_dir=tmp_path)

    assert summary["written"] == 0
    assert summary["mismatched"][0]["field"] == "stub_sha256"


def test_check_only_verifies_file_hashes(tmp_path: Path):
    task = _task()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(_hashed_record(task)) + "\n", encoding="utf-8")
    derived = tmp_path / "derived"
    directory = derived / "bigcodebench" / "BigCodeBench_103"
    directory.mkdir(parents=True)
    files = {
        "stub.py": task.stub,
        "visible_test.py": task.visible_test,
        "hidden_test.py": task.hidden_test,
        "canonical_solution.py": task.canonical_solution,
    }
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    argv = ["--tasks", str(tasks), "--derived", str(derived), "--check-only"]
    assert mat.main(argv) == 0

    (directory / "visible_test.py").write_text("tampered\n", encoding="utf-8")
    assert mat.main(argv) == 1


def test_committed_derived_files_match_the_committed_records():
    report = mat.verify_derived(mat.read_records())
    assert report["missing"] == []
    assert report["mismatched"] == []
