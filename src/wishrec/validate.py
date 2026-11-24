"""Canonical-solution validation against the derived visible and hidden tests.

Each task is validated in an isolated temporary directory by a subprocess with a hard
timeout.  Results are cached in ``data/cache/validation/`` keyed by the task content,
the validator version and the installed package versions, so a rerun only revalidates
tasks whose inputs or environment changed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CACHE_DIR, VALIDATOR_VERSION

_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "faker",
    "requests",
    "beautifulsoup4",
    "lxml",
    "python-dateutil",
    "pytz",
    "regex",
    "sympy",
    "nltk",
    "pillow",
    "cryptography",
    "rsa",
    "pycryptodome",
    "flask",
    "django",
    "folium",
    "python-docx",
    "xmltodict",
    "prettytable",
    "texttable",
    "geopy",
    "holidays",
    "natsort",
    "chardet",
    "openpyxl",
    "xlwt",
    "psutil",
    "pyfakefs",
    "opencv-python-headless",
    "python-Levenshtein",
    "blake3",
    "wordninja",
    "pyquery",
    "sendgrid",
    "mechanize",
    "wikipedia",
    "wordcloud",
    "scikit-image",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = ""
    return versions


_FINGERPRINT: str | None = None


def environment_fingerprint() -> str:
    global _FINGERPRINT
    if _FINGERPRINT is None:
        payload = {
            "validator_version": VALIDATOR_VERSION,
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(),
        }
        _FINGERPRINT = _sha256(json.dumps(payload, sort_keys=True))
    return _FINGERPRINT


def environment_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": _package_versions(),
        "validator_version": VALIDATOR_VERSION,
        "fingerprint": environment_fingerprint(),
    }


@dataclass
class ValidationResult:
    ok: bool
    stage: str = "ok"
    exception: str = ""
    missing_library: str = ""
    detail: str = ""
    visible_seconds: float = 0.0
    hidden_seconds: float = 0.0
    cached: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("cached", None)
        return data


def _classify(text: str) -> tuple[str, str]:
    missing = re.search(r"ModuleNotFoundError: No module named '([^']+)'", text)
    if not missing:
        missing = re.search(r"ImportError: No module named '?([\w.]+)'?", text)
    if missing:
        return "ModuleNotFoundError", missing.group(1).split(".")[0]
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.match(
            r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Failure|Interrupt|Exit))(:|\b)",
            line,
        )
        if match:
            return match.group(1), ""
    return "non-zero exit", ""


def _run(args: list[str], cwd: Path, timeout: float) -> tuple[bool, str, str, float]:
    env = dict(os.environ)
    env.update(
        {
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "WISHREC_VALIDATION": "1",
        }
    )
    started = time.monotonic()
    process = subprocess.Popen(
        args,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    duration = time.monotonic() - started
    combined = (stdout or "") + "\n" + (stderr or "")
    ok = (not timed_out) and process.returncode == 0
    return ok, combined, ("timeout" if timed_out else ""), duration


def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / "validation" / f"{cache_key}.json"


def _cache_key(task) -> str:
    payload = {
        "key": task.key,
        "validator": VALIDATOR_VERSION,
        "environment": environment_fingerprint(),
        "canonical": _sha256(task.canonical_solution),
        "visible": _sha256(task.visible_test),
        "hidden": _sha256(task.hidden_test),
    }
    return _sha256(json.dumps(payload, sort_keys=True))


def _write_task(task, workspace: Path) -> None:
    (workspace / "solution.py").write_text(task.canonical_solution, encoding="utf-8")
    (workspace / "visible_test.py").write_text(task.visible_test, encoding="utf-8")
    (workspace / "hidden_test.py").write_text(task.hidden_test, encoding="utf-8")


def _run_visible(task, workspace: Path, timeout: float):
    if task.source == "bigcodebench":
        args = [sys.executable, "-m", "unittest", "visible_test"]
    else:
        args = [sys.executable, "visible_test.py"]
    return _run(args, workspace, timeout)


def _run_hidden(task, workspace: Path, timeout: float):
    if task.source == "bigcodebench":
        args = [sys.executable, "-m", "unittest", "hidden_test"]
    else:
        args = [sys.executable, "hidden_test.py"]
    return _run(args, workspace, timeout)


def validate_task(
    task,
    *,
    visible_timeout: float = 60.0,
    hidden_timeout: float = 180.0,
    use_cache: bool = True,
    force: bool = False,
) -> ValidationResult:
    cache_key = _cache_key(task)
    cache_file = _cache_path(cache_key)
    if use_cache and not force and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return ValidationResult(cached=True, **data)
        except (json.JSONDecodeError, TypeError):
            pass

    with tempfile.TemporaryDirectory(prefix="wishrec-validate-") as temporary:
        workspace = Path(temporary)
        _write_task(task, workspace)
        visible_ok, visible_output, visible_timeout_flag, visible_seconds = _run_visible(
            task, workspace, visible_timeout
        )
        if not visible_ok:
            exception, missing = (
                ("Timeout", "") if visible_timeout_flag else _classify(visible_output)
            )
            result = ValidationResult(
                ok=False,
                stage="visible",
                exception=exception,
                missing_library=missing,
                detail=visible_output[-3000:],
                visible_seconds=round(visible_seconds, 3),
            )
            _store(cache_file, result)
            return result
        hidden_ok, hidden_output, hidden_timeout_flag, hidden_seconds = _run_hidden(
            task, workspace, hidden_timeout
        )
        if not hidden_ok:
            exception, missing = (
                ("Timeout", "") if hidden_timeout_flag else _classify(hidden_output)
            )
            result = ValidationResult(
                ok=False,
                stage="hidden",
                exception=exception,
                missing_library=missing,
                detail=hidden_output[-3000:],
                visible_seconds=round(visible_seconds, 3),
                hidden_seconds=round(hidden_seconds, 3),
            )
            _store(cache_file, result)
            return result
    result = ValidationResult(
        ok=True,
        visible_seconds=round(visible_seconds, 3),
        hidden_seconds=round(hidden_seconds, 3),
    )
    _store(cache_file, result)
    return result


def _store(cache_file: Path, result: ValidationResult) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(".json.part")
    temporary.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    temporary.replace(cache_file)


def validate_all(
    tasks: list,
    *,
    workers: int = 6,
    force: bool = False,
    use_cache: bool = True,
    progress_every: int = 25,
) -> tuple[dict[str, ValidationResult], dict]:
    """Validate every task, returning results keyed by task key plus summary counts."""
    results: dict[str, ValidationResult] = {}
    ordered = sorted(tasks, key=lambda task: task.key)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(validate_task, task, use_cache=use_cache, force=force): task
            for task in ordered
        }
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                results[task.key] = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad task must not kill the run
                results[task.key] = ValidationResult(
                    ok=False,
                    stage="error",
                    exception=type(exc).__name__,
                    detail=str(exc)[-1000:],
                )
            if index % progress_every == 0 or index == len(ordered):
                passed = sum(1 for result in results.values() if result.ok)
                cached = sum(1 for result in results.values() if result.cached)
                print(
                    f"validated {index}/{len(ordered)} "
                    f"({passed} pass, {cached} cached, "
                    f"{time.monotonic() - started:.0f}s)",
                    flush=True,
                )
    summary = {
        "attempted": len(ordered),
        "passed": sum(1 for result in results.values() if result.ok),
        "failed": sum(1 for result in results.values() if not result.ok),
        "cached": sum(1 for result in results.values() if result.cached),
        "visible_failures": sum(1 for result in results.values() if result.stage == "visible"),
        "hidden_failures": sum(1 for result in results.values() if result.stage == "hidden"),
        "timeouts": sum(1 for result in results.values() if result.exception == "Timeout"),
        "workers": workers,
        "seconds": round(time.monotonic() - started, 1),
    }
    return results, summary
