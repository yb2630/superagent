from __future__ import annotations

import json

import pytest

from fakes import HIDDEN_TEST, FakeLLM, make_task, text_result
from wishrec.harness.conditions import CONDITION_ORDER, CONDITIONS, get_condition
from wishrec.harness.prompts import (
    initial_messages,
    smoothing_messages,
    stakeholder_messages,
    system_prompt,
)
from wishrec.harness.stakeholder import NOT_SPECIFIED, Stakeholder


def test_all_nine_conditions_and_no_h5():
    assert CONDITION_ORDER == ("B1", "H0", "H1", "H2", "H3", "H4", "B2", "B3", "DISCARD")
    assert "H5" not in CONDITIONS
    assert get_condition("H0").level == "L3"
    assert get_condition("B1").level == "L0"
    assert get_condition("H1").acceptance_first is True
    assert get_condition("H4").siblings is True
    assert get_condition("B2").b2_samples == 5
    assert get_condition("B3").b3 is True
    assert get_condition("H3").targeted is True
    assert get_condition("DISCARD").targeted is True
    assert get_condition("DISCARD").discard is True
    assert get_condition("H2").question_budget == 5
    assert get_condition("B2").uses_loop is False
    assert get_condition("B3").uses_loop is False
    assert get_condition("H0").uses_loop is True


def test_b1_gets_the_complete_specification_and_h0_gets_the_wish_only():
    task = make_task()
    b1 = initial_messages(task, get_condition("B1"))
    h0 = initial_messages(task, get_condition("H0"))
    b1_text = json.dumps(b1)
    h0_text = json.dumps(h0)
    assert "Inputs are integers." in b1_text
    assert "Inputs are integers." not in h0_text
    assert "Add two integers." in h0_text
    assert HIDDEN_TEST.strip() not in b1_text
    assert HIDDEN_TEST.strip() not in h0_text
    assert "Parameters" not in h0_text


def test_visible_tests_are_shown_but_hidden_tests_never_are():
    task = make_task()
    messages = initial_messages(task, get_condition("H0"))
    text = "\n".join(message["content"] for message in messages)
    assert task.visible_test.strip() in text
    assert task.hidden_test.strip() not in text
    assert "hidden" not in text.lower()


def test_h1_instruction_asks_for_acceptance_tests_first():
    task = make_task()
    text = json.dumps(initial_messages(task, get_condition("H1")))
    assert "acceptance_tests.py" in text
    assert "Before implementing" in text


def test_h2_prompt_exposes_the_question_tool_and_budget():
    system = system_prompt(get_condition("H2"))
    assert "ask_stakeholder" in system
    assert "5" in system
    assert "No further" not in system


def test_h3_answers_appear_and_discard_answers_are_withheld():
    task = make_task()
    answers = [
        {
            "category": "constraints",
            "label": "constraints",
            "score": 0.9,
            "question": "What constraints?",
            "answer": "Inputs are integers.",
        }
    ]
    h3_text = json.dumps(
        initial_messages(task, get_condition("H3"), answers=answers, unanswered=[])
    )
    assert "Inputs are integers." in h3_text
    assert "What constraints?" not in h3_text

    withheld = [dict(answers[0], answer=None)]
    discard_text = json.dumps(initial_messages(task, get_condition("DISCARD"), withheld=withheld))
    assert "What constraints?" in discard_text
    assert "Inputs are integers." not in discard_text
    assert "Code defensively" in discard_text


def test_h4_siblings_are_included_only_for_h4():
    task = make_task()
    siblings = [
        {
            "key": "test/sibling",
            "entry_point": "double",
            "code": "def double(x):\n    return 2 * x",
            "doctests": ">>> double(2)\n4",
        }
    ]
    h4_text = json.dumps(initial_messages(task, get_condition("H4"), siblings=siblings))
    h0_text = json.dumps(initial_messages(task, get_condition("H0"), siblings=siblings))
    assert "double" in h4_text
    assert "double" not in h0_text


def test_stakeholder_sees_only_the_l0_specification_at_temperature_zero():
    task = make_task()
    llm = FakeLLM([text_result("Inputs are integers.")])
    stakeholder = Stakeholder(llm, task.levels["L0"])
    answer = stakeholder.answer("What are the input types?")
    assert answer == "Inputs are integers."
    call = llm.calls[0]
    prompt = "\n".join(message["content"] for message in call["messages"])
    assert task.levels["L0"] in prompt
    assert task.hidden_test.strip() not in prompt
    assert HIDDEN_TEST.strip() not in prompt
    assert task.canonical_solution not in prompt
    assert call["temperature"] == 0.0
    assert call["seed"] == 0
    assert "That is not specified." in prompt


def test_empty_stakeholder_answer_is_normalized_to_refusal():
    llm = FakeLLM([text_result("")])
    stakeholder = Stakeholder(llm, "Intent: do it")
    assert stakeholder.answer("What about errors?") == NOT_SPECIFIED
    assert stakeholder.answer("") == NOT_SPECIFIED
    assert len(llm.calls) == 1


def test_stakeholder_requires_a_specification():
    with pytest.raises(ValueError, match="non-empty"):
        Stakeholder(FakeLLM([]), "   ")


def test_smoothing_prompt_is_a_separate_temperature_zero_call():
    llm = FakeLLM([text_result("Return the sum of two integers.")])
    from wishrec.harness.wish import smooth_wish

    result = smooth_wish(llm, "Add two integers.")
    assert result == "Return the sum of two integers."
    assert "without adding requirements" in llm.calls[0]["messages"][0]["content"]
    assert llm.calls[0]["temperature"] == 0.0
    assert smoothing_messages("Add two integers.") == llm.calls[0]["messages"]


def test_stakeholder_messages_include_the_complete_specification_verbatim():
    messages = stakeholder_messages("Intent: x\n\nConstraints:\ny", "What?")
    assert "Intent: x" in messages[0]["content"]
    assert "Constraints:" in messages[0]["content"]
