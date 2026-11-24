"""Shared test fixtures: a tiny task, scripted clients and a local fake vLLM endpoint."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from wishrec.detector.trained import CategoryQuestion
from wishrec.harness.llm import ChatResult, ToolCall
from wishrec.harness.models import Task

SOLUTION = "def add(a, b):\n    return a + b\n"

VISIBLE_TEST = (
    "from solution import add\n"
    "assert add(1, 2) == 3\n"
    "assert add(-1, 1) == 0\n"
    "assert add(0, 0) == 0\n"
)

HIDDEN_TEST = (
    "from solution import add\n"
    "assert add(1, 2) == 3\n"
    "assert add(2, 3) == 5\n"
    "assert add(-2, -3) == -5\n"
)


def make_task(**overrides) -> Task:
    values = {
        "key": "test/add",
        "source": "humanevalplus",
        "source_task_id": "add",
        "entry_point": "add",
        "split": "evolution",
        "stub": "def add(a, b):\n    # TODO: implement this function.\n    pass\n",
        "fields": {
            "intent": "Add two integers.",
            "signature_types": "def add(a: int, b: int) -> int",
            "examples": ">>> add(1, 2)\n3",
            "constraints": "Inputs are integers.",
            "error_behaviour": "",
        },
        "field_presence": {
            "intent": True,
            "signature_types": True,
            "examples": True,
            "constraints": True,
            "error_behaviour": False,
        },
        "levels": {
            "L0": (
                "Intent:\nAdd two integers.\n\nSignature and types:\n"
                "```python\ndef add(a: int, b: int) -> int\n```\n\nExamples:\n"
                "```python\n>>> add(1, 2)\n3\n```\n\n"
                "Constraints and edge cases:\nInputs are integers."
            ),
            "L1": "Intent:\nAdd two integers.\n\nSignature:\n```python\ndef add(a, b)\n```",
            "L2": "Intent:\nAdd two integers.\n\nSignature:\n```python\ndef add(a, b)\n```",
            "L3": "Add two integers.",
        },
        "visible_asserts": "assert add(1, 2) == 3",
        "visible_asserts_count": 3,
        "visible_test": VISIBLE_TEST,
        "hidden_test": HIDDEN_TEST,
        "hidden_test_sha256": "0" * 64,
        "canonical_solution": SOLUTION,
        "canonical_sha256": "1" * 64,
        "requirements": [],
        "meta": {"visible_test_cases": ["one", "two", "three"]},
        "record": {},
    }
    values.update(overrides)
    return Task(**values)


def make_unittest_task(**overrides) -> Task:
    visible = (
        "from solution import add\nimport unittest\n\n"
        "class TestAdd(unittest.TestCase):\n"
        "    def test_one(self):\n        self.assertEqual(add(1, 2), 3)\n"
        "    def test_two(self):\n        self.assertEqual(add(-1, 1), 0)\n"
        "    def test_three(self):\n        self.assertEqual(add(0, 0), 0)\n"
    )
    hidden = visible + "\n"
    values = {
        "key": "test/bcb-add",
        "source": "bigcodebench",
        "source_task_id": "bcb-add",
        "visible_test": visible,
        "hidden_test": hidden,
    }
    values.update(overrides)
    return make_task(**values)


def text_result(content: str, *, model: str = "fake", transport: str = "native") -> ChatResult:
    return ChatResult(
        content=content,
        tool_calls=[],
        model=model,
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        transport=transport,
        finish_reason="stop",
    )


def tool_result(name: str, arguments: dict | str, *, tool_id: str = "call-1") -> ChatResult:
    return ChatResult(
        content="",
        tool_calls=[
            ToolCall(
                id=tool_id,
                name=name,
                arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
            )
        ],
        model="fake",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        transport="native",
        finish_reason="tool_calls",
    )


class FakeLLM:
    """Scripted client implementing the loop's ``chat`` interface."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if not self.responses:
            return text_result("done")
        response = self.responses.pop(0)
        if callable(response):
            return response(messages, kwargs)
        return response


class FakeDetector:
    def __init__(self, categories=("constraints", "error_behaviour")):
        self.categories = list(categories)

    def select_questions(self, wish: str, *, budget: int = 5):
        return [
            CategoryQuestion(
                category=category,
                label=category.replace("_", " "),
                score=0.9 - 0.1 * index,
                question=f"Question about {category}?",
            )
            for index, category in enumerate(self.categories)
        ][:budget]


def _usage(prompt: int = 10, completion: int = 5) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def native_response(content: str = "", tool_calls=None, model: str = "fake") -> dict:
    tool_calls = tool_calls or []
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments
                                if isinstance(arguments, str)
                                else json.dumps(arguments),
                            },
                        }
                        for index, (name, arguments) in enumerate(tool_calls)
                    ],
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": _usage(),
    }


def error_response(status: int, message: str) -> dict:
    """An OpenAI-style error body that FakeEndpoint returns with HTTP ``status``."""
    return {
        "__status__": status,
        "error": {"message": message, "type": "BadRequestError", "code": status},
    }


def text_response(content: str, model: str = "fake") -> dict:
    return native_response(content=content, model=model)


def generic_response(request: dict) -> dict:
    """A scripted model that can solve the tiny add task under every condition."""
    messages = request.get("messages") or []
    system = str(messages[0].get("content", "")) if messages else ""
    if "You are the stakeholder" in system:
        return text_response("Inputs are integers; return their sum.")
    if "reviewing an implementation" in system:
        return native_response(
            tool_calls=[("write_file", {"path": "solution.py", "content": SOLUTION})]
        )
    if not request.get("tools"):
        return text_response("```python\n" + SOLUTION + "```")
    if any(message.get("role") == "tool" for message in messages):
        return text_response("Done.")
    return native_response(
        tool_calls=[("write_file", {"path": "solution.py", "content": SOLUTION})]
    )


class FakeEndpoint:
    """Minimal OpenAI-compatible HTTP server used by the transport/loop tests."""

    def __init__(self, responses=None, handler=None, model: str = "fake"):
        self.responses = list(responses or [])
        self.handler = handler
        self.model = model
        self.requests: list[dict] = []
        self._server = None
        self._thread = None
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload: dict) -> None:
                payload = dict(payload)
                status = int(payload.pop("__status__", 200))
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    self._send_json(
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": endpoint.model,
                                    "object": "model",
                                    "created": 1,
                                    "owned_by": "fake",
                                }
                            ],
                        }
                    )
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                endpoint.requests.append(payload)
                if endpoint.handler is not None:
                    response = endpoint.handler(payload)
                elif endpoint.responses:
                    response = endpoint.responses.pop(0)
                    if callable(response):
                        response = response(payload)
                else:
                    response = text_response("done")
                self._send_json(response)

            def log_message(self, *args):
                return

        self._handler = Handler

    def __enter__(self) -> FakeEndpoint:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("endpoint is not running")
        return f"http://127.0.0.1:{self._server.server_port}/v1"
