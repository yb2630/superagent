"""Offline analyses over committed artifacts.

Every analysis here reads only files already committed under ``results/`` (no model
download, no endpoint, no network) and writes a deterministic JSON plus a Markdown table
under ``results/analysis/``.  Rerunning an analysis on the same inputs reproduces its output
byte-for-byte, which the tests check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: Default output directory for analysis artifacts.
ANALYSIS_DIR = Path("results/analysis")


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dumps_deterministic(payload: dict[str, Any]) -> str:
    """Serialise ``payload`` as sorted, indented JSON with a trailing newline."""
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    """Round ``value`` to ``digits`` places, passing ``None`` through."""
    return None if value is None else round(float(value), digits)


def fmt(value: float | int | None, digits: int = 3) -> str:
    """Format a number for a Markdown table; ``None`` renders as an em dash."""
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


__all__ = ["ANALYSIS_DIR", "dumps_deterministic", "fmt", "round_or_none", "sha256_file"]
