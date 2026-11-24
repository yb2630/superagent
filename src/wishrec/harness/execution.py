"""Subprocess execution of generated code with a scrubbed environment and timeouts.

The agent never runs inside the harness process: every visible-test run and the hidden
evaluation execute in a fresh subprocess whose working directory is the per-unit
workspace.  The environment drops API keys, the repository path and ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

EXCEPTION_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt|Failure))\b")

VISIBLE_TIMEOUT = 30.0
HIDDEN_TIMEOUT = 120.0


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    script: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def combined(self) -> str:
        return (self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr).strip()


def scrubbed_env(workspace: Path) -> dict[str, str]:
    """Only what Python needs to import the standard library and installed packages."""
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(workspace),
        "TMPDIR": str(workspace),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "MPLBACKEND": "Agg",
    }


def _run(args: list[str], cwd: Path, timeout: float) -> RunResult:
    started = time.monotonic()
    process = subprocess.Popen(
        args,
        cwd=str(cwd),
        env=scrubbed_env(cwd),
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
        except (ProcessLookupError, PermissionError):
            process.kill()
        stdout, stderr = process.communicate()
    return RunResult(
        returncode=124 if timed_out else int(process.returncode),
        stdout=stdout or "",
        stderr=stderr or "",
        duration_s=time.monotonic() - started,
        timed_out=timed_out,
        script=" ".join(args[1:]),
    )


def run_python_file(script: Path, cwd: Path, timeout: float = VISIBLE_TIMEOUT) -> RunResult:
    return _run([sys.executable, str(script)], cwd, timeout)


def run_python_module(
    module: str, cwd: Path, timeout: float = VISIBLE_TIMEOUT, *args: str
) -> RunResult:
    return _run([sys.executable, "-m", module, *args], cwd, timeout)


def run_visible_suite(source: str, workspace: Path, timeout: float = VISIBLE_TIMEOUT) -> RunResult:
    """Run the derived three-method visible test file for the task's source."""
    if source == "bigcodebench":
        return run_python_module("unittest", workspace, timeout, "visible_test")
    return run_python_file(workspace / "visible_test.py", workspace, timeout)


def run_hidden_suite(source: str, workspace: Path, timeout: float = HIDDEN_TIMEOUT) -> RunResult:
    """Run the full hidden suite against the current ``solution.py``."""
    if source == "bigcodebench":
        return run_python_module("unittest", workspace, timeout, "hidden_test")
    return run_python_file(workspace / "hidden_test.py", workspace, timeout)


def classify_exception(result: RunResult) -> str:
    if result.timed_out:
        return "Timeout"
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    for line in reversed(lines):
        match = EXCEPTION_RE.match(line)
        if match:
            return match.group(1).split(".")[-1]
    if result.returncode != 0:
        return "NonZeroExit"
    return "None"


__all__ = [
    "HIDDEN_TIMEOUT",
    "VISIBLE_TIMEOUT",
    "RunResult",
    "classify_exception",
    "run_hidden_suite",
    "run_python_file",
    "run_python_module",
    "run_visible_suite",
    "scrubbed_env",
]
