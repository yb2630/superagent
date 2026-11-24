"""Prompt templates for the R1 harness, kept in one module for auditability.

Only visible material reaches the generator: the level text, the stub, the three visible
tests, stakeholder answers and (in H4) sibling code.  Hidden tests and canonical solutions
never appear in a prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .conditions import Condition
from .config import QUESTION_BUDGET, TURN_CAP

if TYPE_CHECKING:
    from .models import Task

NOT_SPECIFIED = "That is not specified."

SYSTEM_TEMPLATE = (
    "You are a coding agent working on one Python task. You cannot browse a repository or "
    "read files; the task workspace contains only the files described below. The "
    "evaluated file is solution.py, which starts as a stub.\n\n"
    "Use write_file to submit code. Use run_visible_tests to run the three visible tests "
    "against the current solution.py; use read_error to see the traceback of the last "
    "failed run. {dialogue_clause}\n\n"
    "Work in at most {turn_cap} assistant turns. When you are done, stop calling tools "
    "and reply with a one-line summary. Do not put the solution in the chat; write it "
    "with write_file."
)

NO_DIALOGUE = "You cannot ask the stakeholder questions in this condition."

STAKEHOLDER_SYSTEM = (
    "You are the stakeholder who requested a Python function. You hold the complete "
    "specification below. Answer the developer's question using only this specification. "
    'If the specification does not contain the answer, reply exactly "That is not '
    'specified." Do not invent requirements, do not write code, and do not mention tests '
    "or hidden evaluation.\n\n"
    "## Complete specification\n\n"
    "{l0}"
)

SELF_CHECK_SYSTEM = (
    "You are reviewing an implementation against stakeholder requirements. The current "
    "solution.py and the answers are below. If the implementation violates a "
    "requirement, call write_file with the complete corrected file. Otherwise reply "
    '"OK" and do not call a tool.'
)

DIRECT_SYSTEM = (
    "You implement one Python function. Reply with the complete contents of solution.py "
    "in exactly one ```python code block and nothing else."
)

FRONTIER_SYSTEM = (
    "You implement one Python function from a short wish. Reply with the complete "
    "contents of solution.py in exactly one ```python code block and nothing else."
)

SMOOTH_SYSTEM = (
    "Rewrite the following wish as one or two sentences that preserve the user's intent "
    "without adding requirements, signatures, examples, constraints or error behaviour. "
    "Return only the rewritten wish."
)


def dialogue_clause(condition: Condition) -> str:
    """Return the system-prompt sentence describing the condition's stakeholder access."""
    if condition.name == "H2":
        return (
            "You may call ask_stakeholder with one question at a time, up to "
            f"{QUESTION_BUDGET} questions in total."
        )
    if condition.name == "H3":
        return (
            "The stakeholder's answers to the detector-selected questions are in the "
            "Requirements block below. No further stakeholder questions are available."
        )
    if condition.name == "DISCARD":
        return "The stakeholder is unavailable and no answers are available in this condition."
    return NO_DIALOGUE


def system_prompt(condition: Condition) -> str:
    """Return the loop's system prompt for ``condition``."""
    return SYSTEM_TEMPLATE.format(dialogue_clause=dialogue_clause(condition), turn_cap=TURN_CAP)


def spec_block(task: Task, condition: Condition) -> str:
    """Return the full L0 specification for B1, otherwise the L3 wish."""
    if condition.level == "L0":
        return f"## Specification (complete)\n\n{task.levels['L0']}"
    return f"## Wish\n\n{task.levels['L3']}"


def visible_block(task: Task) -> str:
    """Return the visible-test section shown to the generator."""
    body = (task.visible_test or task.visible_asserts).strip()
    return (
        "## Visible tests\n\nThese are the three tests the stakeholder remembers. "
        "run_visible_tests executes exactly this file:\n\n"
        f"```python\n{body}\n```"
    )


def answers_block(answers: list[dict], unanswered: list[str]) -> str:
    """Format stakeholder answers as requirements, listing unanswered categories."""
    lines = ["## Requirements (from stakeholder answers)"]
    lines.extend(f"- **{item['category']}**: {item['answer']}" for item in answers)
    if unanswered:
        lines.append("")
        lines.append("## Categories with no answer")
        lines.append("")
        lines.append(
            "No stakeholder answer was available for: "
            + ", ".join(unanswered)
            + ". Code defensively for these categories."
        )
    return "\n".join(lines)


def withheld_block(questions: list[dict]) -> str:
    """List the questions whose answers DISCARD withholds."""
    lines = [
        "## Questions asked to the stakeholder",
        "",
        (
            "The stakeholder was asked the following questions, but the answers are "
            "unavailable. Code defensively for these categories:"
        ),
        "",
    ]
    lines.extend(f"- **{item['category']}**: {item['question']}" for item in questions)
    return "\n".join(lines)


def sibling_block(siblings: list[dict]) -> str:
    """Format H4's sibling functions (truncated code and doctests) as context."""
    lines = [
        "## Nearby functions in the codebase (context, not your task)",
        "",
    ]
    for sibling in siblings:
        code = str(sibling.get("code") or "").strip()[:2500]
        examples = str(sibling.get("doctests") or "").strip()[:800]
        block = f"### `{sibling['entry_point']}`"
        if code:
            block += f"\n\n```python\n{code}\n```"
        if examples:
            block += f"\n\n```python\n{examples}\n```"
        lines.append(block + "\n")
    return "\n".join(lines).strip()


def condition_instruction(condition: Condition) -> str:
    """Return the closing instruction of the user prompt for ``condition``."""
    if condition.name == "H1":
        return (
            "Before implementing, write your acceptance tests to `acceptance_tests.py` "
            "with write_file. Then implement `solution.py` and run the visible tests."
        )
    if condition.name == "H2":
        return (
            "Before implementing, ask the stakeholder at least one generic clarifying "
            f"question, and at most {QUESTION_BUDGET} in total, one ask_stakeholder call "
            "per turn. Use the answers while implementing."
        )
    if condition.name == "H3":
        return (
            "Use the Requirements above as acceptance criteria. Implement `solution.py`, "
            "run the visible tests, and stop when done. No further questions are available."
        )
    if condition.name == "DISCARD":
        return (
            "The questions above were asked, but the answers are unavailable. Implement "
            "`solution.py`, coding defensively for those categories."
        )
    if condition.name == "B1":
        return "Implement `solution.py` from the complete specification."
    return "Implement `solution.py` from the wish, then run the visible tests."


def initial_messages(
    task: Task,
    condition: Condition,
    *,
    answers: list[dict] | None = None,
    unanswered: list[str] | None = None,
    withheld: list[dict] | None = None,
    siblings: list[dict] | None = None,
) -> list[dict]:
    """Build the system and first user message of a loop condition."""
    parts = ["## Task", "", spec_block(task, condition)]
    if condition.name in ("H3", "DISCARD"):
        if answers is not None:
            parts.extend(["", answers_block(answers, unanswered or [])])
        if withheld:
            parts.extend(["", withheld_block(withheld)])
    if condition.name == "H4" and siblings:
        parts.extend(["", sibling_block(siblings)])
    parts.extend(
        [
            "",
            "## Starting file (solution.py)",
            "",
            f"```python\n{task.stub.strip()}\n```",
            "",
            visible_block(task),
            "",
            condition_instruction(condition),
        ]
    )
    return [
        {"role": "system", "content": system_prompt(condition)},
        {"role": "user", "content": "\n".join(parts)},
    ]


def direct_messages(task: Task, *, include_visible: bool) -> list[dict]:
    """Build the single-shot messages for B2's direct samples."""
    parts = [
        "## Wish",
        "",
        task.levels["L3"],
        "",
        "## Starting file (solution.py)",
        "",
        f"```python\n{task.stub.strip()}\n```",
    ]
    if include_visible:
        parts.extend(["", visible_block(task)])
    parts.append("")
    parts.append("Write the complete contents of solution.py in one ```python code block.")
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def frontier_messages(task: Task) -> list[dict]:
    """B3 sees only the wish and the stub, with no visible tests or questions."""
    parts = [
        "## Wish",
        "",
        task.levels["L3"],
        "",
        "## Function stub (solution.py)",
        "",
        f"```python\n{task.stub.strip()}\n```",
        "",
        "Write the complete contents of solution.py in one ```python code block.",
    ]
    return [
        {"role": "system", "content": FRONTIER_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def stakeholder_messages(l0_spec: str, question: str) -> list[dict]:
    """Build the stakeholder call: the L0 specification as context, one question."""
    return [
        {"role": "system", "content": STAKEHOLDER_SYSTEM.format(l0=l0_spec)},
        {"role": "user", "content": question},
    ]


def self_check_messages(solution: str, answers: list[dict]) -> list[dict]:
    """Build the review call that checks a solution against the stakeholder answers."""
    requirements = answers_block(answers, [])
    user = (
        f"{requirements}\n\n## Current solution.py\n\n```python\n{solution.strip()}\n```\n\n"
        "If the implementation violates a requirement, call write_file with the complete "
        'corrected file. Otherwise reply with "OK" and do not call a tool.'
    )
    return [
        {"role": "system", "content": SELF_CHECK_SYSTEM},
        {"role": "user", "content": user},
    ]


def smoothing_messages(wish: str) -> list[dict]:
    """One separate endpoint call used by the deferred L3 smoothing step."""
    return [
        {"role": "system", "content": SMOOTH_SYSTEM},
        {"role": "user", "content": wish},
    ]


__all__ = [
    "DIRECT_SYSTEM",
    "FRONTIER_SYSTEM",
    "NOT_SPECIFIED",
    "NO_DIALOGUE",
    "SELF_CHECK_SYSTEM",
    "SMOOTH_SYSTEM",
    "STAKEHOLDER_SYSTEM",
    "answers_block",
    "condition_instruction",
    "dialogue_clause",
    "direct_messages",
    "frontier_messages",
    "initial_messages",
    "self_check_messages",
    "sibling_block",
    "smoothing_messages",
    "spec_block",
    "stakeholder_messages",
    "system_prompt",
    "visible_block",
    "withheld_block",
]
