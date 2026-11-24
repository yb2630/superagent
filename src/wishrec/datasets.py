"""Pinned, checksum-verified dataset downloads and row loading."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .config import DATA_DIR, RAW_DIR, SOURCE_ORDER, SOURCES, Source


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_path(source: Source) -> Path:
    return RAW_DIR / source.name / source.revision / Path(source.filename).name


def _download(source: Source, force: bool = False) -> dict:
    """Fetch one pinned file into data/raw/ and verify its SHA-256.

    ``huggingface_hub`` resolves the pinned commit and gives us the repository's
    caching and retry behaviour; we copy the blob to a stable, revision-keyed path so
    later stages never depend on the cache layout.
    """
    dest = _raw_path(source)
    if dest.exists() and not force:
        digest = sha256_file(dest)
        if digest != source.sha256:
            raise RuntimeError(
                f"{dest} has sha256 {digest}, expected {source.sha256}; "
                "delete it or rerun with --force"
            )
    else:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            repo_id=source.repo,
            filename=source.filename,
            revision=source.revision,
            repo_type="dataset",
            cache_dir=str(RAW_DIR / "_hf-cache"),
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(dest.suffix + ".part")
        shutil.copyfile(cached, temporary)
        # Verify before moving into place, so a bad download never sits at the final path
        # where later stages would read it.
        digest = sha256_file(temporary)
        if digest != source.sha256:
            temporary.unlink()
            raise RuntimeError(
                f"downloaded {source.name} sha256 {digest} != pinned {source.sha256}"
            )
        temporary.replace(dest)
    return {
        "name": source.name,
        "repo": source.repo,
        "revision": source.revision,
        "filename": source.filename,
        "url": source.url,
        "path": str(dest.relative_to(DATA_DIR)),
        "sha256": digest,
        "bytes": dest.stat().st_size,
        "licence": source.licence,
        "expected_rows": source.expected_rows,
    }


def download_sources(force: bool = False) -> list[dict]:
    """Download (or verify) all pinned sources, in a fixed order."""
    records = []
    for name in SOURCE_ORDER:
        source = SOURCES[name]
        print(f"dataset {source.name} @ {source.revision[:12]} ...", flush=True)
        records.append(_download(source, force=force))
    return records


def _load_file(source: Source) -> list[dict]:
    path = _raw_path(source)
    suffix = path.suffix
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    raise ValueError(f"unsupported dataset format: {path}")


def load_raw_rows() -> dict[str, list[dict]]:
    """Read the pinned files, checking each file's SHA-256 and row count."""
    rows: dict[str, list[dict]] = {}
    for name in SOURCE_ORDER:
        source = SOURCES[name]
        path = _raw_path(source)
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run make levels or download first")
        digest = sha256_file(path)
        if digest != source.sha256:
            raise RuntimeError(
                f"{path} has sha256 {digest}, expected the pinned {source.sha256}; "
                "delete it and rerun make levels to download the pinned file"
            )
        source_rows = _load_file(source)
        if len(source_rows) != source.expected_rows:
            raise RuntimeError(
                f"{source.name}: {len(source_rows)} rows, expected {source.expected_rows}"
            )
        rows[source.name] = source_rows
    return rows
