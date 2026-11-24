"""The stakeholder simulator: a temperature-0 model that sees only the L0 specification.

The stakeholder is the only channel through which a condition can recover a deleted
requirement.  Its prompt contains the full specification and nothing else: no wish, no
hidden test, no canonical solution and no repository context.  An empty or evasive answer
is normalized to the documented refusal so the loop can record an unanswered category.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .prompts import NOT_SPECIFIED, stakeholder_messages

if TYPE_CHECKING:
    from .llm import ChatClient


class Stakeholder:
    """Answer clarification questions from the L0 specification alone."""

    def __init__(self, llm: ChatClient, l0_spec: str) -> None:
        if not str(l0_spec).strip():
            raise ValueError("the stakeholder requires a non-empty L0 specification")
        self.llm = llm
        self.l0_spec = str(l0_spec)

    def answer(self, question: str, *, meta: dict | None = None) -> str:
        """Return the stakeholder's answer, or the refusal for an empty question or reply."""
        question = str(question).strip()
        if not question:
            return NOT_SPECIFIED
        result = self.llm.chat(
            stakeholder_messages(self.l0_spec, question),
            purpose="stakeholder",
            meta=meta or {},
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
            seed=0,
        )
        text = (result.content or "").strip()
        return text or NOT_SPECIFIED


__all__ = ["NOT_SPECIFIED", "Stakeholder"]
