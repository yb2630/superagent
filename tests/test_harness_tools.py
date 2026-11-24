from __future__ import annotations

import json

import pytest

from fakes import HIDDEN_TEST, SOLUTION, VISIBLE_TEST, make_task
from wishrec.harness.tools import MAX_CONTENT_BYTES, ToolSession, tool_schemas


def session(tmp_path, **kwargs):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return None\n", encoding="utf-8")
    (tmp_path / "visible_test.py").write_text(VISIBLE_TEST, encoding="utf-8")
    return ToolSession(workspace=tmp_path, **kwargs)


def test_tool_schemas_have_the_four_milestone_tools():
    base = tool_schemas()
    assert [schema["function"]["name"] for schema in base] == [
        "write_file",
        "run_visible_tests",
        "read_error",
    ]
    with_ask = tool_schemas(allow_ask=True)
    assert [schema["function"]["name"] for schema in with_ask][-1] == "ask_stakeholder"
    h1 = tool_schemas(allow_acceptance=True, allow_ask=True)
    assert "acceptance_tests.py" in h1[0]["function"]["parameters"]["properties"]["path"]["enum"]
    assert (
        "acceptance_tests.py" not in base[0]["function"]["parameters"]["properties"]["path"]["enum"]
    )


def test_write_file_passes_visible_tests_and_read_error_returns_traceback(tmp_path):
    tool = session(tmp_path)
    assert tool.execute("write_file", {"path": "solution.py", "content": SOLUTION}).startswith(
        "Wrote"
    )
    assert tool.execute("run_visible_tests", {}) == "Visible tests passed."
    assert tool.execute("read_error", {}) == "No error from the last visible-test run."

    tool.execute(
        "write_file",
        {"path": "solution.py", "content": "def add(a, b):\n    return a - b\n"},
    )
    message = tool.execute("run_visible_tests", {})
    assert "failed" in message
    error = tool.execute("read_error", {})
    assert "AssertionError" in error
    assert tool.visible_runs[-1]["ok"] is False


def test_run_visible_tests_returns_printed_results(tmp_path):
    (tmp_path / "solution.py").write_text(SOLUTION, encoding="utf-8")
    (tmp_path / "visible_test.py").write_text(
        "from solution import add\nprint(add(1, 2))\nprint(add(-1, 1))\n",
        encoding="utf-8",
    )
    tool = ToolSession(workspace=tmp_path)
    message = tool.execute("run_visible_tests", {})
    assert message.startswith("Visible tests passed. Printed output:")
    assert "\n3\n0" in message


def test_write_file_rejects_paths_outside_the_whitelist(tmp_path):
    tool = session(tmp_path)
    for path in ("../solution.py", "/tmp/solution.py", "hidden_test.py", "visible_test.py"):
        message = tool.execute("write_file", {"path": path, "content": "x = 1"})
        assert message.startswith("Error: path")
    assert not (tmp_path.parent / "solution.py").exists()
    assert tool.solution_written is False


def test_acceptance_tests_are_writable_only_in_h1(tmp_path):
    tool = session(tmp_path)
    assert tool.execute(
        "write_file", {"path": "acceptance_tests.py", "content": "assert True"}
    ).startswith("Error:")
    assert not (tmp_path / "acceptance_tests.py").exists()

    h1 = session(tmp_path, allow_acceptance=True)
    assert h1.execute(
        "write_file", {"path": "acceptance_tests.py", "content": "assert True"}
    ).startswith("Wrote")
    assert (tmp_path / "acceptance_tests.py").is_file()


def test_write_file_size_limit(tmp_path):
    tool = session(tmp_path)
    too_big = "x = 1\n" * (MAX_CONTENT_BYTES // 5 + 10)
    assert "larger than" in tool.execute("write_file", {"path": "solution.py", "content": too_big})


def test_ask_stakeholder_consumes_budget_and_disappears_from_schemas(tmp_path):
    answers = []

    def ask(question: str) -> str:
        answers.append(question)
        return "Inputs are integers."

    tool = session(tmp_path, ask_fn=ask, ask_budget=2)
    assert tool.execute("ask_stakeholder", {"question": "Types?"}) == "Inputs are integers."
    assert tool.execute("ask_stakeholder", {"question": "Errors?"}) == "Inputs are integers."
    assert tool.ask_used == 2
    assert tool.remaining_questions == 0
    assert "ask_stakeholder" not in [s["function"]["name"] for s in tool.schemas()]
    exhausted = tool.execute("ask_stakeholder", {"question": "More?"})
    assert exhausted.startswith("Error: the question budget")
    assert answers == ["Types?", "Errors?"]
    assert len(tool.questions) == 2


def test_ask_without_stakeholder_is_an_error(tmp_path):
    tool = session(tmp_path)
    message = tool.execute("ask_stakeholder", {"question": "Anything?"})
    assert message.startswith("Error: no stakeholder")
    assert tool.ask_used == 0


def test_malformed_arguments_are_returned_as_data(tmp_path):
    tool = session(tmp_path)
    assert tool.execute("write_file", "{not json").startswith("Error:")
    assert tool.execute("write_file", json.dumps(["not", "an", "object"])).startswith("Error:")
    assert tool.execute("unknown_tool", {}).startswith("Error: unknown tool")


def test_tool_session_starts_with_the_task_stub(tmp_path):
    (tmp_path / "solution.py").unlink(missing_ok=True)
    tool = ToolSession(workspace=tmp_path, source="humanevalplus")
    assert (tmp_path / "solution.py").read_text(encoding="utf-8") == ""
    assert tool.schemas()


def test_unittest_visible_suite_for_bigcodebench(tmp_path):
    from fakes import make_unittest_task

    task = make_unittest_task()
    (tmp_path / "solution.py").write_text(SOLUTION, encoding="utf-8")
    (tmp_path / "visible_test.py").write_text(task.visible_test, encoding="utf-8")
    tool = ToolSession(workspace=tmp_path, source="bigcodebench")
    assert tool.execute("run_visible_tests", {}) == "Visible tests passed."


def test_hidden_test_file_is_not_exposed_to_the_writer(tmp_path):
    tool = session(tmp_path)
    (tmp_path / "hidden_test.py").write_text(HIDDEN_TEST, encoding="utf-8")
    message = tool.execute("write_file", {"path": "hidden_test.py", "content": "pass"})
    assert message.startswith("Error:")
    assert task_stub_unchanged(tmp_path)


def task_stub_unchanged(tmp_path) -> bool:
    return "return None" in (tmp_path / "solution.py").read_text(encoding="utf-8")
