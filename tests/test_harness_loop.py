from __future__ import annotations

import json

from fakes import (
    SOLUTION,
    FakeDetector,
    FakeLLM,
    make_task,
    text_result,
    tool_result,
)
from wishrec.harness.conditions import get_condition
from wishrec.harness.loop import run_direct_unit, run_repl_unit, run_unit


def test_h0_writes_runs_visible_tests_and_passes_hidden(tmp_path):
    llm = FakeLLM(
        [
            tool_result("write_file", {"path": "solution.py", "content": SOLUTION}),
            tool_result("run_visible_tests", {}),
            text_result("done"),
        ]
    )
    result = run_repl_unit(make_task(), get_condition("H0"), llm)
    assert result["pass"] is True
    assert result["bucket"] == "None"
    assert result["turns"] == 3
    assert result["solution_written"] is True
    assert result["questions"] == 0
    assert result["final_solution"] == SOLUTION
    assert result["tools"]["visible_runs"][0]["ok"] is True


def test_h0_without_a_written_solution_is_no_solution():
    llm = FakeLLM([tool_result("read_error", {}), text_result("I could not do it")])
    result = run_repl_unit(make_task(), get_condition("H0"), llm)
    assert result["pass"] is False
    assert result["bucket"] == "NoSolution"
    assert result["solution_written"] is False


def test_turn_cap_is_twelve_and_is_recorded():
    llm = FakeLLM([tool_result("read_error", {}) for _ in range(20)])
    result = run_repl_unit(make_task(), get_condition("H0"), llm)
    assert result["turns"] == 12
    assert result["turn_cap_hit"] is True
    assert result["pass"] is False
    assert any(event["event"] == "turn_cap_hit" for event in result["trace_events"])


def test_h1_can_write_acceptance_tests():
    llm = FakeLLM(
        [
            tool_result(
                "write_file",
                {"path": "acceptance_tests.py", "content": "assert add(1, 2) == 3"},
            ),
            tool_result("write_file", {"path": "solution.py", "content": SOLUTION}),
            text_result("done"),
        ]
    )
    result = run_repl_unit(make_task(), get_condition("H1"), llm)
    assert result["pass"] is True
    assert result["tools"]["calls"][0]["name"] == "write_file"
    assert "acceptance_tests.py" in result["tools"]["calls"][0]["result"]


def test_h2_question_budget_is_five_and_extra_questions_are_rejected():
    responses = []
    for index in range(5):
        responses.append(tool_result("ask_stakeholder", {"question": f"question {index}?"}))
        responses.append(text_result(f"answer {index}"))
    responses.append(tool_result("ask_stakeholder", {"question": "question 6?"}))
    responses.append(tool_result("write_file", {"path": "solution.py", "content": SOLUTION}))
    responses.append(text_result("done"))
    llm = FakeLLM(responses)
    result = run_repl_unit(make_task(), get_condition("H2"), llm)
    assert result["questions"] == 5
    assert len(result["follow_up_questions"]) == 5
    assert result["follow_up_questions"][0]["answer"] == "answer 0"
    assert result["questions"] <= 5
    assert result["pass"] is True
    calls = [call for call in result["tools"]["calls"] if call["name"] == "ask_stakeholder"]
    assert calls[-1]["ok"] is False
    assert "budget" in calls[-1]["result"]


def test_h3_asks_detector_questions_and_uses_the_answers():
    llm = FakeLLM(
        [
            text_result("Inputs are integers."),
            text_result("Raise ValueError for non-integers."),
            tool_result("write_file", {"path": "solution.py", "content": SOLUTION}),
            text_result("done"),
        ]
    )
    result = run_repl_unit(make_task(), get_condition("H3"), llm, detector=FakeDetector())
    assert result["questions"] == 2
    assert len(result["detector_questions"]) == 2
    assert result["detector_questions"][0]["answer"] == "Inputs are integers."
    assert result["follow_up_questions"] == []
    assert result["pass"] is True
    assert result["self_check"] is not None


def test_h3_without_a_detector_is_an_explicit_error():
    try:
        run_repl_unit(make_task(), get_condition("H3"), FakeLLM([]), detector=None)
    except ValueError as exc:
        assert "requires a trained detector" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("H3 ran without a detector")


def test_discard_asks_the_same_questions_but_withholds_the_answers():
    llm = FakeLLM(
        [
            text_result("Inputs are integers."),
            text_result("Raise ValueError."),
            tool_result("write_file", {"path": "solution.py", "content": SOLUTION}),
            text_result("done"),
        ]
    )
    result = run_repl_unit(make_task(), get_condition("DISCARD"), llm, detector=FakeDetector())
    assert result["questions"] == 2
    assert all(item["answer"] is None for item in result["detector_questions"])
    prompt = json.dumps(result["messages"])
    assert "Inputs are integers." not in prompt
    assert "Raise ValueError." not in prompt
    assert "Question about constraints?" in prompt


def test_h3_self_check_can_repair_the_solution():
    bad = "def add(a, b):\n    return None\n"
    llm = FakeLLM(
        [
            text_result("Inputs are integers."),
            text_result("Return the sum."),
            tool_result("write_file", {"path": "solution.py", "content": bad}),
            text_result("done"),
            tool_result("write_file", {"path": "solution.py", "content": SOLUTION}),
        ]
    )
    result = run_repl_unit(make_task(), get_condition("H3"), llm, detector=FakeDetector())
    assert result["self_check"]["wrote"] is True
    assert result["pass"] is True
    assert result["final_solution"] == SOLUTION


def test_b2_uses_five_direct_samples_and_reports_pass_at_1():
    llm = FakeLLM([text_result("```python\n" + SOLUTION + "```") for _ in range(5)])
    result = run_direct_unit(make_task(), get_condition("B2"), llm)
    assert result["turns"] == 5
    assert len(result["samples"]) == 5
    assert result["pass_at_1"] is True
    assert result["pass"] is True
    assert result["questions"] == 0
    assert all(sample["pass"] for sample in result["samples"])


def test_b3_is_one_call_on_the_wish_only():
    llm = FakeLLM([text_result("```python\n" + SOLUTION + "```")])
    result = run_direct_unit(make_task(), get_condition("B3"), llm)
    assert result["turns"] == 1
    assert result["pass"] is True
    prompt = json.dumps(llm.calls[0]["messages"])
    assert "Visible tests" not in prompt
    assert "Inputs are integers." not in prompt
    assert "Add two integers." in prompt
    assert llm.calls[0]["temperature"] == 0.7


def test_api_error_is_recorded_and_does_not_crash_the_unit():
    class Failing:
        def chat(self, *args, **kwargs):
            raise RuntimeError("connection refused")

    result = run_repl_unit(make_task(), get_condition("H0"), Failing())
    assert result["status"] == "complete"
    assert result["pass"] is False
    assert "connection refused" in result["api_error"]
    assert any(event["event"] == "api_error" for event in result["trace_events"])


def test_run_unit_dispatches_to_loop_and_direct_conditions():
    loop = run_unit(
        make_task(),
        get_condition("H0"),
        FakeLLM([text_result("done")]),
    )
    direct = run_unit(
        make_task(),
        get_condition("B3"),
        FakeLLM([text_result("```python\n" + SOLUTION + "```")]),
    )
    assert loop["condition"] == "H0"
    assert direct["condition"] == "B3"
    assert "samples" not in loop
    assert "samples" in direct


def test_hidden_test_is_evaluated_in_a_fresh_workspace():
    """The loop's workspace never receives hidden_test.py; evaluation writes it separately."""
    llm = FakeLLM(
        [
            tool_result("write_file", {"path": "solution.py", "content": SOLUTION}),
            text_result("done"),
        ]
    )
    result = run_repl_unit(make_task(), get_condition("H0"), llm)
    tool_calls = [call["name"] for call in result["tools"]["calls"]]
    assert "hidden_test.py" not in json.dumps(result["tools"])
    assert result["pass"] is True
    assert tool_calls == ["write_file"]


def test_bucket_classifies_assertion_failures():
    llm = FakeLLM(
        [
            tool_result(
                "write_file",
                {"path": "solution.py", "content": "def add(a, b):\n    return 0\n"},
            ),
            text_result("done"),
        ]
    )
    result = run_repl_unit(make_task(), get_condition("H0"), llm)
    assert result["pass"] is False
    assert result["bucket"] == "AssertionError"
    assert "AssertionError" in result["hidden_output_tail"]
