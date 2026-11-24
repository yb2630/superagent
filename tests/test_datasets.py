from __future__ import annotations

import contextlib
from pathlib import Path

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
