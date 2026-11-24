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
    """Outcome of one subprocess run; a timeout is reported as return code 124."""

    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    script: str = ""

    @property
    def ok(self) -> bool:
        """Whether the run exited cleanly within its timeout."""
        return self.returncode == 0 and not self.timed_out

    def combined(self) -> str:
        """Return stdout and stderr joined by a newline, stripped."""
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


# After SIGKILL, how long to keep reading output.  A grandchild that called setsid()
# escapes killpg and can hold the pipes open indefinitely, so the read must be bounded.
KILL_GRACE_S = 2.0


def _as_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _drain_after_kill(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Collect a killed process's output without waiting on escaped descendants."""
    try:
        stdout, stderr = process.communicate(timeout=KILL_GRACE_S)
        return stdout or "", stderr or ""
    except subprocess.TimeoutExpired as exc:
        # Keep what arrived, close our pipe ends and reap the (already killed) child.
        stdout, stderr = _as_text(exc.stdout), _as_text(exc.stderr)
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()
    process.wait(timeout=KILL_GRACE_S)
    return stdout, stderr


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
        stdout, stderr = _drain_after_kill(process)
    return RunResult(
        returncode=124 if timed_out else int(process.returncode),
        stdout=stdout or "",
        stderr=stderr or "",
        duration_s=time.monotonic() - started,
        timed_out=timed_out,
        script=" ".join(args[1:]),
    )


def run_python_file(script: Path, cwd: Path, timeout: float = VISIBLE_TIMEOUT) -> RunResult:
    """Run a Python script in ``cwd`` under the scrubbed environment."""
    return _run([sys.executable, str(script)], cwd, timeout)


def run_python_module(
    module: str, cwd: Path, timeout: float = VISIBLE_TIMEOUT, *args: str
) -> RunResult:
    """Run ``python -m module *args`` in ``cwd`` under the scrubbed environment."""
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
    """Return ``Timeout`` for a timed-out run, else the last exception named in stderr.

    Without a named exception, a non-zero exit is ``NonZeroExit`` and success is ``None``.
    """
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
