from __future__ import annotations

import json

import pytest

from fakes import FakeEndpoint, native_response, text_response
from wishrec.harness.config import EndpointConfig, frontier_config, generator_config
from wishrec.harness.llm import (
    TEXT_PROTOCOL,
    LLMClient,
    extract_code_block,
    parse_text_tool_calls,
)


def schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        }
    ]


def test_endpoint_config_reads_environment_and_never_exposes_key():
    config = generator_config(
        {
            "VLLM_BASE_URL": "http://example.test:9000/v1",
            "VLLM_API_KEY": "secret-key",
            "VLLM_MODEL": "Qwen/Qwen3-8B-AWQ",
            "VLLM_TIMEOUT": "12",
            "VLLM_TOOL_MODE": "text",
            "VLLM_TEMPERATURE": "0.3",
        }
    )
    assert config.base_url == "http://example.test:9000/v1"
    assert config.api_key == "secret-key"
    assert config.timeout == 12.0
    assert config.tool_mode == "text"
    assert "secret-key" not in json.dumps(config.safe_identity)


def test_invalid_tool_mode_is_rejected():
    with pytest.raises(ValueError):
        EndpointConfig(tool_mode="magic")


def test_frontier_falls_back_to_generator_endpoint_and_model():
    generator = generator_config({"VLLM_BASE_URL": "http://local/v1", "VLLM_MODEL": "m"})
    frontier = frontier_config({"VLLM_BASE_URL": "http://local/v1", "VLLM_MODEL": "m"})
    assert frontier.base_url == generator.base_url
    assert frontier.model == generator.model
    assert frontier.temperature == 0.7


def test_frontier_can_be_overridden():
    frontier = frontier_config(
        {
            "VLLM_BASE_URL": "http://local/v1",
            "VLLM_FRONTIER_BASE_URL": "https://frontier.test/v1",
            "VLLM_FRONTIER_API_KEY": "frontier-key",
            "VLLM_FRONTIER_MODEL": "frontier-model",
        }
    )
    assert frontier.base_url == "https://frontier.test/v1"
    assert frontier.model == "frontier-model"
    assert "frontier-key" not in json.dumps(frontier.safe_identity)


def test_parse_text_tool_calls_accepts_fenced_json_and_flattened_arguments():
    calls, final = parse_text_tool_calls(
        'I will write it.\n```json\n{"tool": "write_file", "arguments": '
        '{"path": "solution.py", "content": "x = 1"}}\n```'
    )
    assert final == ""
    assert calls[0].name == "write_file"
    assert json.loads(calls[0].arguments)["path"] == "solution.py"

    flattened, _ = parse_text_tool_calls('{"tool": "run_visible_tests", "arguments": {}}')
    assert flattened[0].name == "run_visible_tests"

    named, _ = parse_text_tool_calls('{"name": "read_error", "arguments": {}}')
    assert named[0].name == "read_error"


def test_parse_text_tool_calls_recognises_finish_and_plain_text():
    assert parse_text_tool_calls('{"tool": "finish"}') == ([], "")
    calls, final = parse_text_tool_calls("All done; the solution is written.")
    assert calls == []
    assert final == "All done; the solution is written."


def test_extract_code_block():
    assert extract_code_block("```python\nx = 1\n```") == "x = 1\n"
    assert extract_code_block("no fence") == "no fence\n"
    assert extract_code_block("```\ny = 2\n```") == "y = 2\n"


def test_native_tool_call_over_http_and_request_shape():
    with FakeEndpoint(
        [
            native_response(
                tool_calls=[("write_file", {"path": "solution.py", "content": "x = 1"})]
            ),
            text_response("done"),
        ]
    ) as endpoint:
        client = LLMClient(EndpointConfig(base_url=endpoint.base_url, api_key="EMPTY"))
        result = client.chat(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            tools=schemas(),
            temperature=0.7,
            seed=3,
        )
        assert result.transport == "native"
        assert result.tool_calls[0].name == "write_file"
        assert result.usage["input_tokens"] == 10
        request = endpoint.requests[0]
        assert request["model"] == "Qwen/Qwen3-8B-AWQ"
        assert request["temperature"] == 0.7
        assert request["seed"] == 3
        assert request["tools"][0]["function"]["name"] == "write_file"
        assert request["chat_template_kwargs"] == {"enable_thinking": False}


def test_text_mode_uses_json_protocol_without_native_tools():
    payload = {"tool": "write_file", "arguments": {"path": "solution.py", "content": "x = 1"}}
    with FakeEndpoint([text_response(json.dumps(payload))]) as endpoint:
        client = LLMClient(
            EndpointConfig(base_url=endpoint.base_url, tool_mode="text"), max_tokens=32
        )
        result = client.chat(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            tools=schemas(),
        )
        assert result.transport == "text"
        assert result.tool_calls[0].name == "write_file"
        request = endpoint.requests[0]
        assert "tools" not in request
        assert TEXT_PROTOCOL.split("\n")[0] in request["messages"][0]["content"]
        assert "write_file" in request["messages"][0]["content"]


def test_auto_mode_retries_an_endpoint_that_ignores_native_tools():
    payload = {"tool": "write_file", "arguments": {"path": "solution.py", "content": "x = 1"}}
    with FakeEndpoint(
        [text_response("I will write the file now."), text_response(json.dumps(payload))]
    ) as endpoint:
        client = LLMClient(EndpointConfig(base_url=endpoint.base_url, tool_mode="auto"))
        result = client.chat(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            tools=schemas(),
        )
        assert result.tool_calls[0].name == "write_file"
        assert result.transport == "text"
        assert len(endpoint.requests) == 2
        assert "tools" in endpoint.requests[0]
        assert "tools" not in endpoint.requests[1]

        second = client.chat(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "again"}],
            tools=schemas(),
        )
        assert len(endpoint.requests) == 3
        assert "tools" not in endpoint.requests[2]
        assert second is not None


def test_auto_mode_keeps_native_when_tools_are_supported():
    with FakeEndpoint(
        [
            native_response(
                tool_calls=[("run_visible_tests", {})],
            )
        ]
    ) as endpoint:
        client = LLMClient(EndpointConfig(base_url=endpoint.base_url, tool_mode="auto"))
        result = client.chat(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            tools=schemas(),
        )
        assert result.transport == "native"
        assert len(endpoint.requests) == 1


def test_direct_text_call_has_no_protocol_injection():
    with FakeEndpoint([text_response("```python\nx = 1\n```")]) as endpoint:
        client = LLMClient(EndpointConfig(base_url=endpoint.base_url))
        result = client.chat([{"role": "user", "content": "solve"}], temperature=0.7)
        assert "Tool protocol" not in endpoint.requests[0]["messages"][0]["content"]
        assert result.content.startswith("```python")


def test_api_error_is_not_retried_into_a_hang():
    with FakeEndpoint([]) as endpoint:
        client = LLMClient(
            EndpointConfig(
                base_url="http://127.0.0.1:1/v1",
                timeout=0.2,
                max_retries=1,
            )
        )
        with pytest.raises(Exception):
            client.chat([{"role": "user", "content": "hi"}], max_tokens=8)
