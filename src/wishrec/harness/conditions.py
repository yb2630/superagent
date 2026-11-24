"""The nine R1 conditions.

All conditions share the same tasks, tools and visible tests.  H5 (offline harness
evolution) is a plan item and is deliberately not implemented in R1.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import QUESTION_BUDGET, TURN_CAP


@dataclass(frozen=True)
class Condition:
    """One experimental condition: the wish level shown and the clarification it allows."""

    name: str
    level: str
    dialogue: bool
    question_budget: int
    description: str
    acceptance_first: bool = False
    siblings: bool = False
    targeted: bool = False
    discard: bool = False
    b2_samples: int = 0
    b3: bool = False

    @property
    def uses_loop(self) -> bool:
        """Whether the condition runs the ReAct loop (B2 and B3 sample directly)."""
        return not (self.b2_samples or self.b3)

    @property
    def pre_asks_detector(self) -> bool:
        """Whether detector-selected questions are asked before the loop starts."""
        return self.targeted


CONDITIONS: dict[str, Condition] = {
    "B1": Condition(
        name="B1",
        level="L0",
        dialogue=False,
        question_budget=0,
        description="Full L0 specification, no dialogue (ceiling).",
    ),
    "H0": Condition(
        name="H0",
        level="L3",
        dialogue=False,
        question_budget=0,
        description="Plain ReAct loop on the wish.",
    ),
    "H1": Condition(
        name="H1",
        level="L3",
        dialogue=False,
        question_budget=0,
        description="Acceptance tests first, no dialogue.",
        acceptance_first=True,
    ),
    "H2": Condition(
        name="H2",
        level="L3",
        dialogue=True,
        question_budget=QUESTION_BUDGET,
        description="Generic clarification with the shared question budget.",
    ),
    "H3": Condition(
        name="H3",
        level="L3",
        dialogue=False,
        question_budget=0,
        description="Detector-targeted clarification (2-3 questions asked up front).",
        targeted=True,
    ),
    "H4": Condition(
        name="H4",
        level="L3",
        dialogue=False,
        question_budget=0,
        description="Context-rich: five sibling functions with doctests.",
        siblings=True,
    ),
    "B2": Condition(
        name="B2",
        level="L3",
        dialogue=False,
        question_budget=0,
        description="Five direct samples instead of questions.",
        b2_samples=5,
    ),
    "B3": Condition(
        name="B3",
        level="L3",
        dialogue=False,
        question_budget=0,
        description="Frontier endpoint if configured, else Qwen3-8B at temperature 0.7.",
        b3=True,
    ),
    "DISCARD": Condition(
        name="DISCARD",
        level="L3",
        dialogue=False,
        question_budget=0,
        description="H3's detector questions are asked, but the answers are withheld.",
        targeted=True,
        discard=True,
    ),
}

CONDITION_ORDER = ("B1", "H0", "H1", "H2", "H3", "H4", "B2", "B3", "DISCARD")

# Import-time invariants of the R1 design; a violation is a programming error.
if len(CONDITION_ORDER) != 9 or any(name not in CONDITIONS for name in CONDITION_ORDER):
    raise RuntimeError("CONDITION_ORDER must list exactly the nine defined conditions")
if TURN_CAP != 12:
    raise RuntimeError(f"the R1 design fixes the turn cap at 12, not {TURN_CAP}")


def get_condition(name: str) -> Condition:
    """Return the condition called ``name``, with a helpful error for unknown names."""
    try:
        return CONDITIONS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown condition {name!r}; choose one of {list(CONDITION_ORDER)}"
        ) from exc


def resolve_conditions(names: list[str] | tuple[str, ...] | None) -> list[Condition]:
    """Return the named conditions in the given order, or all nine when none are named."""
    if not names:
        return [CONDITIONS[name] for name in CONDITION_ORDER]
    return [get_condition(name) for name in names]


__all__ = [
    "CONDITIONS",
    "CONDITION_ORDER",
    "Condition",
    "get_condition",
    "resolve_conditions",
]
