from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wishrec import build
from wishrec.config import DERIVED_DIR, LEVELS_DIR


class _StopBuildError(Exception):
    pass


def test_build_passes_the_cli_timeouts_to_validation(monkeypatch, tmp_path: Path):
    task = SimpleNamespace(key="humanevalplus/HumanEval/0", source="humanevalplus")
    seen: dict = {}

    def fake_validate_all(tasks, **kwargs):
        seen.update(kwargs)
        raise _StopBuildError

    monkeypatch.setattr(build, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(build, "download_sources", lambda force=False: [])
    monkeypatch.setattr(build, "load_raw_rows", dict)
    monkeypatch.setattr(build, "_parse_all", lambda rows, visible_tests: ([task], []))
    monkeypatch.setattr(build, "validate_all", fake_validate_all)

    with pytest.raises(_StopBuildError):
        build.main(
            ["--out", str(tmp_path / "out"), "--visible-timeout", "7", "--hidden-timeout", "9"]
        )
    assert seen["visible_timeout"] == 7.0
    assert seen["hidden_timeout"] == 9.0


def test_derived_dir_follows_out_unless_given():
    assert build.resolve_derived_dir(LEVELS_DIR, None) == DERIVED_DIR
    scratch = Path("/scratch/levels")
    assert build.resolve_derived_dir(scratch, None) == scratch / "derived"
    explicit = Path("/elsewhere/derived")
    assert build.resolve_derived_dir(scratch, explicit) == explicit
    assert build.resolve_derived_dir(LEVELS_DIR, explicit) == explicit


def _write_tasks(path: Path, splits: dict[str, str]) -> None:
    path.write_text(
        "".join(json.dumps({"key": key, "split": split}) + "\n" for key, split in splits.items()),
        encoding="utf-8",
    )


def test_split_guard_refuses_to_change_a_committed_split(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks, {"a": "train", "b": "heldout"})
    moved = [{"key": "a", "split": "heldout"}, {"key": "b", "split": "train"}]

    with pytest.raises(RuntimeError, match="held-out"):
        build.check_split_unchanged(tasks, moved, allow_resplit=False)
    build.check_split_unchanged(tasks, moved, allow_resplit=True)


def test_split_guard_rejects_a_changed_task_set(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    _write_tasks(tasks, {"a": "train", "b": "heldout"})
    with pytest.raises(RuntimeError, match="--allow-resplit"):
        build.check_split_unchanged(
            tasks,
            [{"key": "a", "split": "train"}, {"key": "c", "split": "heldout"}],
            allow_resplit=False,
        )


def test_split_guard_accepts_an_identical_or_new_split(tmp_path: Path):
    tasks = tmp_path / "tasks.jsonl"
    same = [{"key": "a", "split": "train"}, {"key": "b", "split": "heldout"}]
    build.check_split_unchanged(tasks, same, allow_resplit=False)  # no file yet
    _write_tasks(tasks, {"a": "train", "b": "heldout"})
    build.check_split_unchanged(tasks, same, allow_resplit=False)


def test_committed_levels_pass_the_split_guard_against_themselves():
    tasks = LEVELS_DIR / "tasks.jsonl"
    records = [json.loads(line) for line in tasks.read_text(encoding="utf-8").splitlines()]
    build.check_split_unchanged(tasks, records, allow_resplit=False)
