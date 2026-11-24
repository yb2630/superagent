"""The four milestone tools: write_file, run_visible_tests, read_error, ask_stakeholder.

``write_file`` can only touch ``solution.py`` and, in H1 only, ``acceptance_tests.py``.
The visible and hidden suites are never writable, so a generated solution cannot change
the tests it is judged by.  Tool failures are returned to the model as data; they never
crash the loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .execution import VISIBLE_TIMEOUT, classify_exception, run_visible_suite

MAX_CONTENT_BYTES = 120_000
MAX_TOOL_OUTPUT = 6_000
MAX_ERROR_OUTPUT = 6_000


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... truncated {len(text) - limit} chars]"


def tool_schemas(*, allow_acceptance: bool = False, allow_ask: bool = False) -> list[dict]:
    paths = ["solution.py"]
    if allow_acceptance:
        paths.append("acceptance_tests.py")
    schemas: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Write Python source to a file in the task workspace. Only solution.py"
                    + (" and acceptance_tests.py" if allow_acceptance else "")
                    + " are writable. solution.py is the submitted implementation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "enum": paths,
                            "description": "File to overwrite.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete new contents of the file.",
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_visible_tests",
                "description": (
                    "Run the three visible tests against the current solution.py and return "
                    "pass/fail plus anything the tests print. Call read_error for the "
                    "traceback of the last failure."
                ),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_error",
                "description": "Return the traceback from the most recent failed visible-test run.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]
    if allow_ask:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "ask_stakeholder",
                    "description": (
                        "Ask the stakeholder one question about the requirements. The answer "
                        "comes only from the stakeholder's specification and consumes one "
                        "question from the shared budget."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "One question for the stakeholder.",
                            }
                        },
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    return schemas


@dataclass
class ToolSession:
    workspace: Path
    source: str = "humanevalplus"
    allow_acceptance: bool = False
    ask_fn: Callable[[str], str] | None = None
    ask_budget: int = 0
    ask_used: int = 0
    solution_written: bool = False
    last_error: str = ""
    calls: list[dict] = field(default_factory=list)
    visible_runs: list[dict] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        if not (self.workspace / "solution.py").exists():
            (self.workspace / "solution.py").write_text("", encoding="utf-8")

    def schemas(self) -> list[dict]:
        return tool_schemas(
            allow_acceptance=self.allow_acceptance,
            allow_ask=self.ask_fn is not None and self.ask_budget > self.ask_used,
        )

    @property
    def remaining_questions(self) -> int:
        return max(self.ask_budget - self.ask_used, 0)

    def execute(self, name: str, arguments: str | dict) -> str:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        except Exception as exc:  # noqa: BLE001 - malformed model output is data
            result = f"Error: tool arguments must be a JSON object ({exc})."
            self.calls.append({"name": name, "arguments": arguments, "result": result, "ok": False})
            return result
        if not isinstance(args, dict):
            result = "Error: tool arguments must be a JSON object."
            self.calls.append({"name": name, "arguments": arguments, "result": result, "ok": False})
            return result
        try:
            if name == "write_file":
                result = self._write_file(args)
            elif name == "run_visible_tests":
                result = self._run_visible_tests()
            elif name == "read_error":
                result = self._read_error()
            elif name == "ask_stakeholder":
                result = self._ask(args)
            else:
                result = f"Error: unknown tool {name!r}."
        except Exception as exc:  # noqa: BLE001 - tool errors are data for the agent
            result = f"Error: {type(exc).__name__}: {exc}"
        ok = not result.startswith("Error:")
        result = _truncate(result)
        self.calls.append({"name": name, "arguments": args, "result": result, "ok": ok})
        return result

    # -- individual tools ---------------------------------------------------------
    def _write_file(self, args: dict) -> str:
        path = str(args.get("path", ""))
        content = args.get("content")
        allowed = {"solution.py"}
        if self.allow_acceptance:
            allowed.add("acceptance_tests.py")
        if path not in allowed:
            return f"Error: path {path!r} is not writable. Choose one of {sorted(allowed)}."
        if not isinstance(content, str):
            return "Error: content must be a string."
        if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
            return f"Error: content is larger than {MAX_CONTENT_BYTES} bytes."
        (self.workspace / path).write_text(content, encoding="utf-8")
        if path == "solution.py":
            self.solution_written = True
        return f"Wrote {len(content)} characters to {path}."

    def _run_visible_tests(self) -> str:
        script = self.workspace / "visible_test.py"
        if not script.exists():
            return "Error: the visible test file is missing."
        result = run_visible_suite(self.source, self.workspace, VISIBLE_TIMEOUT)
        record = {
            "ok": result.ok,
            "exception": classify_exception(result),
            "duration_s": round(result.duration_s, 3),
            "output": _truncate(result.combined(), 2000),
        }
        self.visible_runs.append(record)
        if result.ok:
            self.last_error = ""
            printed = result.stdout.strip()
            if printed:
                return f"Visible tests passed. Printed output:\n{_truncate(printed, 2000)}"
            return "Visible tests passed."
        self.last_error = result.combined()
        return (
            f"Visible tests failed ({record['exception']}). Call read_error to see the traceback."
        )

    def _read_error(self) -> str:
        if not self.last_error:
            return "No error from the last visible-test run."
        return _truncate(self.last_error, MAX_ERROR_OUTPUT)

    def _ask(self, args: dict) -> str:
        question = str(args.get("question", "")).strip()
        if not question:
            return "Error: ask_stakeholder needs a non-empty question."
        if self.ask_fn is None or self.ask_budget <= 0:
            return "Error: no stakeholder is available for questions in this condition."
        if self.ask_used >= self.ask_budget:
            return f"Error: the question budget ({self.ask_budget}) is exhausted."
        self.ask_used += 1
        answer = self.ask_fn(question)
        self.questions.append({"index": self.ask_used, "question": question, "answer": answer})
        return answer

    # -- trace --------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "visible_runs": self.visible_runs,
            "questions": self.questions,
            "ask_used": self.ask_used,
            "solution_written": self.solution_written,
        }


__all__ = [
    "MAX_CONTENT_BYTES",
    "MAX_ERROR_OUTPUT",
    "MAX_TOOL_OUTPUT",
    "ToolSession",
    "tool_schemas",
]
