from types import SimpleNamespace

import pytest

import wishrec.validate as validate_module
from wishrec.validate import (
    ValidationResult,
    environment_fingerprint,
    validate_task,
)

PASSING_SOLUTION = "def double(x):\n    return x * 2\n"
PASSING_VISIBLE = "from solution import double\nassert double(2) == 4\nassert double(3) == 6\nassert double(4) == 8\n"
PASSING_HIDDEN = "from solution import double\nassert double(5) == 10\nassert double(6) == 12\nassert double(7) == 14\n"


def make_task(**overrides):
    values = {
        "key": "test/task/1",
        "source": "humanevalplus",
        "canonical_solution": PASSING_SOLUTION,
        "visible_test": PASSING_VISIBLE,
        "hidden_test": PASSING_HIDDEN,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_module, "CACHE_DIR", tmp_path / "cache")


def test_passing_task_is_validated_and_cached():
    task = make_task()
    first = validate_task(task, use_cache=True)
    assert first.ok
    assert first.stage == "ok"
    assert first.cached is False
    second = validate_task(task, use_cache=True)
    assert second.ok
    assert second.cached is True


def test_visible_failure_is_reported_before_hidden():
    task = make_task(visible_test="from solution import double\nassert double(1) == 3\n")
    result = validate_task(task, use_cache=False)
    assert not result.ok
    assert result.stage == "visible"
    assert result.exception == "AssertionError"
    assert result.hidden_seconds == 0.0


def test_hidden_failure_is_reported():
    task = make_task(hidden_test="from solution import double\nassert double(1) == 3\n")
    result = validate_task(task, use_cache=False)
    assert not result.ok
    assert result.stage == "hidden"
    assert result.exception == "AssertionError"


def test_missing_library_is_classified():
    task = make_task(
        hidden_test="import definitely_missing_module_xyz\nfrom solution import double\n"
    )
    result = validate_task(task, use_cache=False)
    assert not result.ok
    assert result.stage == "hidden"
    assert result.exception == "ModuleNotFoundError"
    assert result.missing_library == "definitely_missing_module_xyz"


def test_timeout_is_reported():
    task = make_task(
        visible_test="import time\ntime.sleep(5)\n",
        hidden_test=PASSING_HIDDEN,
    )
    result = validate_task(task, visible_timeout=0.2, use_cache=False)
    assert not result.ok
    assert result.stage == "visible"
    assert result.exception == "Timeout"


def test_bigcodebench_runs_as_unittest_class():
    task = make_task(
        key="bigcodebench/test",
        source="bigcodebench",
        visible_test=(
            "from solution import double\n"
            "import unittest\n"
            "class TestCases(unittest.TestCase):\n"
            "    def test_a(self):\n"
            "        self.assertEqual(double(1), 2)\n"
            "    def test_b(self):\n"
            "        self.assertEqual(double(2), 4)\n"
            "    def test_c(self):\n"
            "        self.assertEqual(double(3), 6)\n"
        ),
        hidden_test=(
            "from solution import double\n"
            "import unittest\n"
            "class TestCases(unittest.TestCase):\n"
            "    def test_a(self):\n"
            "        self.assertEqual(double(4), 8)\n"
        ),
    )
    result = validate_task(task, use_cache=False)
    assert result.ok


def test_environment_fingerprint_is_stable():
    assert environment_fingerprint() == environment_fingerprint()
    assert len(environment_fingerprint()) == 64


def test_result_dict_is_json_serialisable():
    result = ValidationResult(ok=True, visible_seconds=0.1, hidden_seconds=0.2)
    data = result.to_dict()
    assert data["ok"] is True
    assert "cached" not in data
