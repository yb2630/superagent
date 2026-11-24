from __future__ import annotations

import sys
import textwrap
import time

from wishrec.harness.execution import RunResult, classify_exception, run_python_file


def test_timeout_returns_even_when_a_detached_grandchild_holds_the_pipes(tmp_path):
    """killpg cannot reach a setsid'd grandchild; the harness must not wait for it."""
    script = tmp_path / "spawn.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import subprocess, time
            print("started", flush=True)
            subprocess.Popen(
                [{sys.executable!r}, "-c", "import time; time.sleep(30)"],
                start_new_session=True,
            )
            time.sleep(30)
            """
        ),
        encoding="utf-8",
    )
    started = time.monotonic()
    result = run_python_file(script, tmp_path, timeout=1.0)
    elapsed = time.monotonic() - started
    assert result.timed_out is True
    assert result.returncode == 124
    assert classify_exception(result) == "Timeout"
    assert elapsed < 10, f"run blocked for {elapsed:.1f}s on a detached grandchild"


def test_classify_exception_prefers_timeout_then_stderr_then_exit_code():
    def result(**overrides):
        values = {
            "returncode": 1,
            "stdout": "",
            "stderr": "",
            "duration_s": 0.1,
            "timed_out": False,
            "script": "x",
        }
        values.update(overrides)
        return RunResult(**values)

    assert classify_exception(result(timed_out=True, stderr="ValueError: x")) == "Timeout"
    assert classify_exception(result(stderr="Traceback\nValueError: bad")) == "ValueError"
    assert classify_exception(result()) == "NonZeroExit"
    assert classify_exception(result(returncode=0)) == "None"
