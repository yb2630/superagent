from __future__ import annotations

import contextlib
from dataclasses import replace
from pathlib import Path

import pytest

from wishrec import datasets


def test_hf_download_is_for_a_dataset_repo(monkeypatch, tmp_path: Path):
    """hf_hub_download must name the dataset repo type, or a fresh make levels 404s."""
    source = datasets.SOURCES["humanevalplus"]
    calls: dict = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        blob = tmp_path / "blob.jsonl"
        blob.write_bytes(b"not the pinned file")
        return str(blob)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(datasets, "RAW_DIR", tmp_path / "raw")
    # The fake blob cannot match the pinned SHA-256; the call itself is what we check.
    with contextlib.suppress(RuntimeError):
        datasets._download(source, force=True)
    assert calls["repo_id"] == source.repo
    assert calls["revision"] == source.revision
    assert calls["repo_type"] == "dataset"


def test_a_download_failing_its_checksum_is_not_left_in_place(monkeypatch, tmp_path: Path):
    source = datasets.SOURCES["humanevalplus"]
    blob = tmp_path / "blob.jsonl"
    blob.write_bytes(b"not the pinned file")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(blob))
    monkeypatch.setattr(datasets, "RAW_DIR", tmp_path / "raw")

    with pytest.raises(RuntimeError, match="sha256"):
        datasets._download(source, force=True)

    destination = datasets._raw_path(source)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))


def test_loading_raw_rows_verifies_the_pinned_hash(monkeypatch, tmp_path: Path):
    pinned = datasets.SOURCES["humanevalplus"]
    source = replace(pinned, expected_rows=1)
    monkeypatch.setattr(datasets, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(datasets, "SOURCE_ORDER", ("humanevalplus",))
    monkeypatch.setattr(datasets, "SOURCES", {"humanevalplus": source})
    path = datasets._raw_path(source)
    path.parent.mkdir(parents=True)
    path.write_text('{"task_id": "tampered"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="sha256"):
        datasets.load_raw_rows()
