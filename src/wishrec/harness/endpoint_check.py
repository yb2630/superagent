"""Probe an OpenAI-compatible endpoint's tool-call transports before an R2 run.

The R2 notebook serves ``Qwen/Qwen3-8B-AWQ`` with vLLM and must record which tool protocol
works.  vLLM only returns native OpenAI tool calls when it was started with
``--enable-auto-tool-choice`` (and a matching tool-call parser); the published R2 command
does not pass those flags, so the JSON text fallback is the usual transport.  This module
checks both, selects the transport the harness should use, writes a JSON artifact and can
emit a ``configs/vllm.env`` file for the runner.  It never writes or prints the API key.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import (
    EndpointConfig,
    atomic_write_json,
    atomic_write_text,
    generator_config,
    utc_now_iso,
)
from .llm import LLMClient

ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo one short string back to the caller.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    },
}
PROBE_MESSAGES = [
    {
        "role": "system",
        "content": "You are a tool-use probe. Call the echo tool with value 'ok'.",
    },
    {
        "role": "user",
        "content": "Call the echo tool with value 'ok'. Do not answer in prose.",
    },
]


def list_models(config: EndpointConfig) -> dict:
    """GET ``{base_url}/models`` without using the chat client.

    Only ``http``/``https`` URLs are opened; any other scheme (``file:`` and the like) is
    reported as a failed probe rather than read.
    """
    url = config.base_url.rstrip("/") + "/models"
    try:
        scheme = urllib.parse.urlsplit(url).scheme
        if scheme not in {"http", "https"}:
            raise ValueError(f"endpoint URL must use http or https, not {scheme!r}")
        # S310: the scheme is restricted to http/https just above.
        request = urllib.request.Request(  # noqa: S310
            url,
            headers={
                "Authorization": f"Bearer {config.api_key or 'EMPTY'}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=min(config.timeout, 60)
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [str(item.get("id", "")) for item in payload.get("data", [])]
        return {"ok": True, "url": url, "models": [item for item in models if item]}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {"ok": False, "url": url, "models": [], "error": f"{type(exc).__name__}: {exc}"}


def _probe(config: EndpointConfig, tool_mode: str) -> dict:
    client = LLMClient(
        EndpointConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            timeout=config.timeout,
            max_retries=1,
            tool_mode=tool_mode,
            temperature=0.0,
            top_p=1.0,
            enable_thinking=config.enable_thinking,
            extra_headers=dict(config.extra_headers),
        ),
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,
        seed=0,
    )
    try:
        result = client.chat(PROBE_MESSAGES, tools=[ECHO_TOOL], purpose="endpoint_probe")
    except Exception as exc:  # noqa: BLE001 - a failed probe is the recorded result
        return {
            "ok": False,
            "transport": tool_mode,
            "error": f"{type(exc).__name__}: {exc}",
            "tool_calls": [],
            "content": "",
        }
    calls = [{"name": call.name, "arguments": call.arguments[:500]} for call in result.tool_calls]
    return {
        "ok": bool(calls),
        "transport": result.transport,
        "error": "",
        "tool_calls": calls,
        "content": (result.content or "")[:500],
        "usage": dict(result.usage or {}),
    }


def check_endpoint(config: EndpointConfig | None = None) -> dict:
    """Probe models, native tool calls and the text fallback; return the artifact."""
    config = config or generator_config()
    models = list_models(config)
    native = _probe(config, "native")
    text = _probe(config, "text")
    if native["ok"]:
        selected = "native"
    elif text["ok"]:
        selected = "text"
    else:
        selected = None
    return {
        "format_version": 1,
        "checked_at": utc_now_iso(),
        "endpoint": config.safe_identity,
        "models": models,
        "native": native,
        "text": text,
        "selected_transport": selected,
        "ok": selected is not None,
        "note": (
            "native OpenAI tool calls are available"
            if selected == "native"
            else (
                "native tool calls were unavailable or ignored; the JSON text fallback works"
                if selected == "text"
                else "neither transport produced a tool call; do not start the pilot"
            )
        ),
    }


def env_lines(config: EndpointConfig, transport: str) -> str:
    """Return a sourceable ``configs/vllm.env``.  A real key is never written."""
    lines = [
        "# Written by wishrec.harness.endpoint_check; safe to source.",
        f"VLLM_BASE_URL={config.base_url}",
        f"VLLM_MODEL={config.model}",
        f"VLLM_TIMEOUT={int(config.timeout)}",
        f"VLLM_MAX_RETRIES={config.max_retries}",
        f"VLLM_TOOL_MODE={transport}",
        f"VLLM_ENABLE_THINKING={'true' if config.enable_thinking else 'false'}",
        f"VLLM_TEMPERATURE={config.temperature}",
        f"VLLM_TOP_P={config.top_p}",
    ]
    if config.api_key and config.api_key != "EMPTY":
        lines.append("# VLLM_API_KEY omitted; keep the real key in the environment.")
    else:
        lines.append("VLLM_API_KEY=EMPTY")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Probe the configured endpoint, write the JSON artifact and optionally an env file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default results/runs/<run-id>/endpoint.json",
    )
    parser.add_argument(
        "--env-out",
        type=Path,
        default=None,
        help="write a sourceable VLLM_* env file (key omitted unless it is EMPTY)",
    )
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args(argv)

    config = generator_config()
    if args.timeout is not None:
        config = EndpointConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            timeout=args.timeout,
            max_retries=config.max_retries,
            tool_mode=config.tool_mode,
            temperature=config.temperature,
            top_p=config.top_p,
            enable_thinking=config.enable_thinking,
            extra_headers=dict(config.extra_headers),
        )
    artifact = check_endpoint(config)
    output = (
        Path(args.output)
        if args.output
        else Path("results") / "runs" / args.run_id / "endpoint.json"
    )
    atomic_write_json(output, artifact)

    print(f"endpoint: {config.base_url}  model: {config.model}")
    print(
        "models: "
        + (", ".join(artifact["models"]["models"]) if artifact["models"]["ok"] else "CAREFUL")
        + ("" if artifact["models"]["ok"] else f" ({artifact['models'].get('error')})")
    )
    print(
        f"native tool calls: {'ok' if artifact['native']['ok'] else 'failed'}"
        + (
            ""
            if artifact["native"]["ok"]
            else f" ({artifact['native'].get('error') or 'no tool call returned'})"
        )
    )
    print(
        f"text fallback: {'ok' if artifact['text']['ok'] else 'failed'}"
        + (
            ""
            if artifact["text"]["ok"]
            else f" ({artifact['text'].get('error') or 'no tool call returned'})"
        )
    )
    print(f"selected transport: {artifact['selected_transport'] or 'NONE'}")
    print(f"wrote {output}")
    if args.env_out is not None and artifact["selected_transport"]:
        atomic_write_text(args.env_out, env_lines(config, artifact["selected_transport"]))
        print(f"wrote {args.env_out}")
    if not artifact["ok"]:
        print("ERROR: no usable tool transport; fix the endpoint before the pilot")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ECHO_TOOL",
    "PROBE_MESSAGES",
    "check_endpoint",
    "env_lines",
    "list_models",
    "main",
]
