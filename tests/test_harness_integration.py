from __future__ import annotations

import json

from fakes import (
    SOLUTION,
    FakeEndpoint,
    make_task,
    native_response,
    text_response,
)
from wishrec.harness.conditions import get_condition
from wishrec.harness.config import EndpointConfig
from wishrec.harness.llm import LLMClient
from wishrec.harness.loop import run_repl_unit


def _json_call(name: str, arguments: dict) -> str:
    return json.dumps({"tool": name, "arguments": arguments})


def test_text_protocol_loop_writes_then_finishes():
    with FakeEndpoint(
        [
            text_response(_json_call("write_file", {"path": "solution.py", "content": SOLUTION})),
            text_response(_json_call("run_visible_tests", {})),
            text_response(_json_call("finish", {})),
        ]
    ) as endpoint:
        client = LLMClient(
            EndpointConfig(base_url=endpoint.base_url, tool_mode="text"),
            max_tokens=512,
        )
        result = run_repl_unit(make_task(), get_condition("H0"), client)
        assert result["pass"] is True
        assert result["turns"] == 3
        assert [call["name"] for call in result["tools"]["calls"]] == [
            "write_file",
            "run_visible_tests",
        ]


def test_text_protocol_h2_can_ask_the_stakeholder():
    with FakeEndpoint(
        [
            text_response(_json_call("ask_stakeholder", {"question": "What are the types?"})),
            text_response("Both inputs and the result are integers."),
            text_response(_json_call("write_file", {"path": "solution.py", "content": SOLUTION})),
            text_response(_json_call("finish", {})),
        ]
    ) as endpoint:
        client = LLMClient(
            EndpointConfig(base_url=endpoint.base_url, tool_mode="text"),
            max_tokens=512,
        )
        result = run_repl_unit(make_task(), get_condition("H2"), client)
        assert result["pass"] is True
        assert result["questions"] == 1
        assert result["follow_up_questions"][0]["question"] == "What are the types?"
        assert "integers" in result["follow_up_questions"][0]["answer"]


def test_auto_fallback_loop_works_with_an_endpoint_that_ignores_tools():
    with FakeEndpoint(
        [
            text_response("I will write the implementation."),
            text_response(_json_call("write_file", {"path": "solution.py", "content": SOLUTION})),
            native_response(tool_calls=[("run_visible_tests", {})]),
            # Under the text protocol only the explicit finish ends the loop.
            text_response('{"tool": "finish"}'),
        ]
    ) as endpoint:
        client = LLMClient(EndpointConfig(base_url=endpoint.base_url, tool_mode="auto"))
        result = run_repl_unit(make_task(), get_condition("H0"), client)
        assert result["pass"] is True
        assert result["turns"] == 3
        assert result["llm_calls"][0]["transport"] == "text"
