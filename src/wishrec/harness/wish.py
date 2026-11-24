"""L3 wish smoothing: one separate endpoint call, implemented but not run in R1.

The R1 detector trains on the raw first-sentence L3 from R0.  R2 will smooth the 500
training wishes with this prompt call, refit the detector and use the smoothed wishes for
the pilot.  Keeping the call here makes that step explicit and testable without changing
the R1 artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .prompts import smoothing_messages

if TYPE_CHECKING:
    from .llm import ChatClient


def smooth_wish(llm: ChatClient, wish: str, *, meta: dict | None = None) -> str:
    """Rewrite a wish with one deterministic call; an empty reply keeps the original."""
    wish = str(wish).strip()
    if not wish:
        raise ValueError("cannot smooth an empty wish")
    result = llm.chat(
        smoothing_messages(wish),
        purpose="wish_smoothing",
        meta=meta or {},
        temperature=0.0,
        top_p=1.0,
        max_tokens=160,
        seed=0,
    )
    text = (result.content or "").strip()
    return text or wish


__all__ = ["smooth_wish"]
