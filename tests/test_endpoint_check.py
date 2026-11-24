from __future__ import annotations

import json

from fakes import FakeEndpoint, native_response, text_response
from wishrec.harness.config import EndpointConfig
from wishrec.harness.endpoint_check import check_endpoint, env_lines, main


def _json_call(name: str, arguments: dict) -> str:
    return json.dumps({"tool": name, "arguments": arguments})


def _config(endpoint, *, api_key="EMPTY", model="fake") -> EndpointConfig:
    return EndpointConfig(
        base_url=endpoint.base_url,
        api_key=api_key,
        model=model,
        timeout=10,
        max_retries=1,
    )


def test_native_tool_calls_are_preferred_when_the_endpoint_supports_them():
    with FakeEndpoint(
        [
            native_response(tool_calls=[("echo", {"value": "ok"})]),
            text_response(_json_call("echo", {"value": "ok"})),
        ],
        model="Qwen/Qwen3-8B-AWQ",
    ) as endpoint:
        artifact = check_endpoint(_config(endpoint))
        assert artifact["models"]["ok"] is True
        assert artifact["models"]["models"] == ["Qwen/Qwen3-8B-AWQ"]
        assert artifact["native"]["ok"] is True
        assert artifact["text"]["ok"] is True
        assert artifact["selected_transport"] == "native"
        assert artifact["ok"] is True
        assert "api_key" not in json.dumps(artifact)


def test_text_fallback_is_selected_when_native_tools_are_ignored():
    with FakeEndpoint(
        [
            text_response("I cannot call tools."),
            text_response(_json_call("echo", {"value": "ok"})),
        ]
    ) as endpoint:
        artifact = check_endpoint(_config(endpoint))
        assert artifact["native"]["ok"] is False
        assert artifact["text"]["ok"] is True
        assert artifact["selected_transport"] == "text"
        assert artifact["ok"] is True


def test_endpoint_without_any_tool_transport_is_flagged():
    with FakeEndpoint([text_response("plain answer"), text_response("still plain")]) as endpoint:
        artifact = check_endpoint(_config(endpoint))
        assert artifact["selected_transport"] is None
        assert artifact["ok"] is False
        assert "do not start the pilot" in artifact["note"]


def test_env_file_never_contains_a_real_key():
    config = EndpointConfig(
        base_url="http://127.0.0.1:8000/v1",
        api_key="super-secret-token",
        model="Qwen/Qwen3-8B-AWQ",
        timeout=180,
    )
    text = env_lines(config, "text")
    assert "super-secret-token" not in text
    assert "VLLM_API_KEY=EMPTY" not in text
    assert "VLLM_API_KEY omitted" in text
    assert "VLLM_TOOL_MODE=text" in text


def test_env_file_marks_the_local_empty_key():
    config = EndpointConfig(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        model="Qwen/Qwen3-8B-AWQ",
        timeout=180,
    )
    text = env_lines(config, "native")
    assert "VLLM_API_KEY=EMPTY" in text
    assert "VLLM_TOOL_MODE=native" in text


def test_endpoint_check_cli_writes_artifact_and_env(tmp_path):
    with FakeEndpoint(
        [
            native_response(tool_calls=[("echo", {"value": "ok"})]),
            text_response(_json_call("echo", {"value": "ok"})),
        ]
    ) as endpoint:
        import os

        old = {key: os.environ.get(key) for key in ("VLLM_BASE_URL", "VLLM_MODEL")}
        os.environ["VLLM_BASE_URL"] = endpoint.base_url
        os.environ["VLLM_MODEL"] = "fake"
        try:
            code = main(
                [
                    "--run-id",
                    "probe",
                    "--output",
                    str(tmp_path / "endpoint.json"),
                    "--env-out",
                    str(tmp_path / "vllm.env"),
                    "--timeout",
                    "10",
                ]
            )
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    assert code == 0
    artifact = json.loads((tmp_path / "endpoint.json").read_text(encoding="utf-8"))
    assert artifact["selected_transport"] == "native"
    assert (tmp_path / "vllm.env").read_text(encoding="utf-8").count("VLLM_TOOL_MODE=native") == 1


def test_list_models_explains_a_base_url_without_a_scheme():
    from wishrec.harness.endpoint_check import list_models

    result = list_models(EndpointConfig(base_url="localhost:8000/v1"))
    assert result["ok"] is False
    assert "must start with http:// or https://" in result["error"]
    assert "'localhost:8000/v1/models'" in result["error"]
