"""OpenAI-compatible vLLM chat client with a native and a text/JSON tool transport.

Qwen3-8B served by vLLM normally exposes OpenAI function calling.  A text protocol is
implemented as well because some builds or chat templates return only text: the model
writes one JSON object ``{"tool": ..., "arguments": ...}`` per turn.  In ``auto`` mode the
client starts with native tools, and if the first response carries no native call it
retries that turn with the text protocol and remembers the working transport.  Both paths
feed the same :class:`ToolCall` objects to the ReAct loop.

The client talks only to the configured endpoint.  There is no paid API fallback and no
credential logging; the API key is sent in the HTTP header and never recorded.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .config import EndpointConfig
from .tools import TOOL_NAMES

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

try:  # pragma: no cover - import guard for detector-only environments
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment]


@dataclass
class ToolCall:
    """One tool invocation requested by the model; ``arguments`` is a JSON string."""

    id: str
    name: str
    arguments: str


@dataclass
class ChatResult:
    """A parsed assistant turn: text, tool calls, token usage and the transport used."""

    content: str
    tool_calls: list[ToolCall]
    model: str
    usage: dict
    transport: str = "native"
    finish_reason: str = ""
    # Text transport only: the reply was the explicit ``{"tool": "finish"}``, or the JSON
    # error that stopped a tool call from parsing.  The loop uses both to decide whether a
    # turn without tool calls ends the episode or earns corrective feedback.
    finished: bool = False
    parse_error: str = ""
    raw: Any = field(default=None, repr=False)

    @property
    def input_tokens(self) -> int:
        """Prompt tokens reported by the endpoint."""
        return int(self.usage.get("input_tokens", 0) or 0)

    @property
    def output_tokens(self) -> int:
        """Completion tokens reported by the endpoint."""
        return int(self.usage.get("output_tokens", 0) or 0)

    def as_usage(self) -> dict:
        """Return the normalized input/output/total token counts."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": int(self.usage.get("total_tokens", 0) or 0),
        }


TEXT_PROTOCOL = """## Tool protocol (text fallback)

This endpoint does not return native tool calls. On every assistant turn reply with
exactly one JSON object and nothing else. Use the tools listed here:

{tools}

Format:
{{"tool": "<tool name>", "arguments": {{...}}}}

For write_file, arguments are {{"path": "solution.py", "content": "<complete file>"}}.
For run_visible_tests and read_error, arguments are {{}}.
For ask_stakeholder, arguments are {{"question": "<one question>"}}.
When you are finished, reply with {{"tool": "finish"}}."""


def _tool_catalog(tools: list[dict]) -> str:
    lines = []
    for schema in tools:
        function = schema.get("function") or {}
        name = function.get("name", "")
        description = (function.get("description") or "").strip().split("\n")[0]
        lines.append(f"- `{name}`: {description}")
    return "\n".join(lines)


def _inject_text_protocol(messages: list[dict], tools: list[dict]) -> list[dict]:
    block = TEXT_PROTOCOL.format(tools=_tool_catalog(tools))
    cloned = [dict(message) for message in messages]
    for message in cloned:
        if message.get("role") == "system":
            message["content"] = (message.get("content") or "") + "\n\n" + block
            return cloned
    cloned.insert(0, {"role": "system", "content": block})
    return cloned


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _candidate_objects(text: str) -> Iterator[str]:
    """Yield JSON-object candidates in priority order.

    Fenced blocks first, then the whole text when it starts with ``{``, then every object
    that decodes from an embedded ``{``.
    """
    for match in _JSON_FENCE.finditer(text or ""):
        yield match.group(1)
    stripped = (text or "").strip()
    if stripped.startswith("{"):
        yield stripped
    decoder = json.JSONDecoder()
    for index, character in enumerate(text or ""):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield json.dumps(value)


def parse_text_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Return tool calls encoded in an assistant text message.

    Accepts ``{"tool": name, "arguments": {...}}``, the flattened variant with the
    arguments beside ``tool``, and ``{"name": ..., "arguments": ...}`` for a known tool
    name (so ordinary ``"name"`` keys in data are not calls).  ``finish`` or a
    message without such an object yields no calls; :func:`text_reply_status` tells the
    two apart for the loop.
    """
    for candidate in _candidate_objects(content):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("tool") or payload.get("name")
        if name == "finish" or payload.get("final") is not None:
            return [], ""
        if not isinstance(name, str) or not name:
            continue
        if "tool" not in payload and name not in TOOL_NAMES:
            continue  # e.g. {"name": "Alice"} in example data, not a tool call
        arguments = payload.get("arguments")
        if arguments is None:
            arguments = {
                key: value for key, value in payload.items() if key not in {"tool", "name"}
            }
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                return [ToolCall(id="text-0", name=name, arguments=arguments)], ""
            arguments_text = arguments
        else:
            arguments_text = json.dumps(arguments, ensure_ascii=False)
        return [ToolCall(id="text-0", name=name, arguments=arguments_text)], ""
    return [], content


def text_reply_status(content: str) -> tuple[bool, str]:
    """Classify a text-protocol reply that yielded no tool call.

    Return ``(finished, parse_error)``: ``finished`` when the reply is the explicit
    ``finish`` object, else the first JSON error from an object-like candidate (empty when
    the reply holds no ``{`` at all, i.e. it is prose).
    """
    error = ""
    for candidate in _candidate_objects(content):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            error = error or str(exc)
            continue
        if isinstance(payload, dict) and (
            (payload.get("tool") or payload.get("name")) == "finish"
            or payload.get("final") is not None
        ):
            return True, ""
    text = content or ""
    if not error and "{" in text:
        try:
            json.JSONDecoder().raw_decode(text[text.index("{") :])
        except json.JSONDecodeError as exc:
            error = str(exc)
    return False, error


def _usage_dict(usage: object) -> dict:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
    }


class ChatClient(Protocol):
    """The chat interface the loop needs; :class:`LLMClient` and test fakes satisfy it."""

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        purpose: str | None = None,
        meta: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> ChatResult:
        """Send one chat turn and return the parsed result."""
        ...


class LLMClient:
    """One endpoint/model sampling configuration.

    ``chat`` accepts the same keyword arguments as the loop's fake clients, so tests can
    substitute a scripted object without touching the loop.
    """

    def __init__(
        self,
        config: EndpointConfig | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 1600,
        seed: int | None = None,
        purpose: str = "generator",
        timeout: float | None = None,
    ) -> None:
        self.config = config or EndpointConfig()
        self.temperature = self.config.temperature if temperature is None else temperature
        self.top_p = self.config.top_p if top_p is None else top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.purpose = purpose
        self.timeout = self.config.timeout if timeout is None else timeout
        self.max_retries = self.config.max_retries
        self._client: openai.OpenAI | None = None
        self._transport: str | None = None

    def _openai(self) -> openai.OpenAI:
        """Return the lazily constructed OpenAI client for the configured endpoint."""
        if openai is None:  # pragma: no cover
            raise RuntimeError("the openai package is not installed")
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.config.api_key or "EMPTY",
                base_url=self.config.base_url,
                timeout=self.timeout,
                max_retries=0,
                default_headers=self.config.extra_headers or None,
            )
        return self._client

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        purpose: str | None = None,
        meta: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> ChatResult:
        """Send one chat turn, negotiating the native or text tool transport in auto mode.

        Per-call sampling arguments override the client's defaults; ``purpose`` and
        ``meta`` are accepted for interface parity with the fakes and ignored.
        """
        del purpose, meta  # the caller records purpose in its trace
        transport = self._transport or self.config.tool_mode
        if transport == "text" and tools:
            return self._call(
                _inject_text_protocol(messages, tools),
                tools=None,
                temperature=self.temperature if temperature is None else temperature,
                top_p=self.top_p if top_p is None else top_p,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                seed=self.seed if seed is None else seed,
                transport="text",
            )
        negotiating = bool(self.config.tool_mode == "auto" and self._transport is None and tools)
        try:
            result = self._call(
                messages,
                tools=tools,
                temperature=self.temperature if temperature is None else temperature,
                top_p=self.top_p if top_p is None else top_p,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                seed=self.seed if seed is None else seed,
                transport="native",
            )
        except Exception as exc:
            # vLLM started without --enable-auto-tool-choice rejects any request that
            # carries tools with HTTP 400.  In auto mode that means "native unsupported":
            # switch to the text protocol for this turn and the rest of the session.
            if not negotiating or not tools or getattr(exc, "status_code", None) != 400:
                raise
            self._transport = "text"
            return self._call(
                _inject_text_protocol(messages, tools),
                tools=None,
                temperature=self.temperature if temperature is None else temperature,
                top_p=self.top_p if top_p is None else top_p,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                seed=self.seed if seed is None else seed,
                transport="text",
            )
        if not negotiating or not tools:
            return result
        self._transport = "native"
        if result.tool_calls:
            return result
        parsed, final_text = parse_text_tool_calls(result.content)
        if parsed:
            result.tool_calls = parsed
            result.transport = "text"
            result.content = final_text
            self._transport = "text"
            return result
        # The endpoint may have ignored the tools.  Retry this turn under the text
        # protocol once, remember the working transport, and continue from there.
        fallback = self._call(
            _inject_text_protocol(messages, tools),
            tools=None,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            seed=self.seed if seed is None else seed,
            transport="text",
        )
        self._transport = "text" if fallback.tool_calls else self._transport
        return fallback

    def _call(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None,
        transport: str,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.config.enable_thinking is not None:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": bool(self.config.enable_thinking)}
            }
        last_error: Exception | None = None
        for attempt in range(max(1, self.max_retries)):
            try:
                raw = self._openai().chat.completions.create(**kwargs)
                return self._parse(raw, transport)
            except Exception as exc:
                if not self._retryable(exc) or attempt >= self.max_retries - 1:
                    raise
                last_error = exc
                time.sleep(min(2.0**attempt, 8.0))
        raise RuntimeError(f"chat request failed after retries: {last_error!r}")

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if openai is None:  # pragma: no cover
            return False
        retryable = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
        if isinstance(exc, retryable):
            return True
        status = getattr(exc, "status_code", None)
        return bool(status == 429 or (isinstance(status, int) and status >= 500))

    @staticmethod
    def _parse(raw: ChatCompletion, transport: str) -> ChatResult:
        """Convert a chat completion into a :class:`ChatResult`, parsing text tool calls."""
        choice = raw.choices[0]
        message = choice.message
        content = getattr(message, "content", None) or ""
        calls: list[ToolCall] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            calls.append(
                ToolCall(
                    id=str(getattr(tool_call, "id", "") or f"call-{len(calls)}"),
                    name=str(getattr(function, "name", "")),
                    arguments=str(getattr(function, "arguments", "{}") or "{}"),
                )
            )
        finished, parse_error = False, ""
        if transport == "text" and not calls:
            finished, parse_error = text_reply_status(content)
            parsed, final_text = parse_text_tool_calls(content)
            calls = parsed
            content = final_text
        return ChatResult(
            content=content,
            tool_calls=calls,
            model=str(getattr(raw, "model", "") or ""),
            usage=_usage_dict(getattr(raw, "usage", None)),
            transport=transport,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            finished=finished,
            parse_error=parse_error,
            raw=raw,
        )


__all__ = [
    "TEXT_PROTOCOL",
    "ChatClient",
    "ChatResult",
    "LLMClient",
    "ToolCall",
    "extract_code_block",
    "parse_text_tool_calls",
    "text_reply_status",
]


def extract_code_block(text: str) -> str:
    """Return the first fenced Python block, or the whole text if there is none."""
    if not text:
        return ""
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            language = stripped[3:].strip().lower()
            if start is None and language in ("python", "py", ""):
                start = index + 1
                continue
            if start is not None:
                return "\n".join(lines[start:index]).strip("\n") + "\n"
    if start is not None:
        return "\n".join(lines[start:]).strip("\n") + "\n"
    return text.strip("\n") + "\n"
