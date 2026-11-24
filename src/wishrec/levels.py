"""Mechanical level construction: L0-L3 and the single-field drops L2_c = L0 - c."""

from __future__ import annotations

from typing import Any

from .config import FIELD_KEYS, FIELD_LABELS, LEVELS

L1_DELETIONS = ["examples", "signature_types.annotations"]
L2_DELETIONS = ["examples", "signature_types.annotations", "constraints", "error_behaviour"]
L3_DELETIONS = ["signature_types", "examples", "constraints", "error_behaviour"]


def _code_block(text: str) -> str:
    return f"```python\n{text.strip()}\n```"


def _looks_like_example_code(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith((">>>", "...", "assert ", "def ", "import ", "from "))


def _render_examples(text: str) -> str:
    return _code_block(text) if _looks_like_example_code(text) else text.strip()


def _render_intent(text: str) -> str:
    return f"Intent:\n{text.strip()}"


def _render_signature(text: str, *, code: bool) -> str:
    if code:
        return f"Signature and types:\n{_code_block(text)}"
    return f"Signature and types:\n{text.strip()}"


def _render_untyped_signature(text: str) -> str:
    return f"Signature:\n{_code_block(text)}"


def _render_constraints(text: str) -> str:
    return f"Constraints and edge cases:\n{text.strip()}"


def _render_errors(text: str) -> str:
    return f"Error behaviour:\n{text.strip()}"


def _block(fields: dict[str, str], category: str, *, signature_is_code: bool) -> str:
    text = fields.get(category, "").strip()
    if not text:
        return ""
    if category == "intent":
        return _render_intent(text)
    if category == "signature_types":
        return _render_signature(text, code=signature_is_code)
    if category == "examples":
        return f"Examples:\n{_render_examples(text)}"
    if category == "constraints":
        return _render_constraints(text)
    if category == "error_behaviour":
        return _render_errors(text)
    raise KeyError(category)


def render_level(fields: dict[str, str], level: str, *, signature_is_code: bool) -> str:
    """Render one level.

    L0 keeps every present field.  L1 drops examples and type detail (the signature is
    kept, untyped).  L2 additionally drops constraints and error behaviour, keeping only
    intent and the untyped signature.  L3 keeps only the intent, i.e. the wish.
    """
    intent = fields.get("intent", "").strip()
    if level == "L3":
        if not intent:
            raise ValueError("L3 needs a non-empty intent")
        return intent
    if level not in LEVELS:
        raise ValueError(level)
    blocks: list[str] = []
    if intent:
        blocks.append(_render_intent(intent))
    if level in ("L0", "L1", "L2"):
        if level == "L0":
            block = _block(fields, "signature_types", signature_is_code=signature_is_code)
        else:
            untyped = fields.get("signature_untyped", "").strip()
            block = _render_untyped_signature(untyped) if untyped else ""
        if block:
            blocks.append(block)
    if level == "L0":
        block = _block(fields, "examples", signature_is_code=signature_is_code)
        if block:
            blocks.append(block)
    if level in ("L0", "L1"):
        for category in ("constraints", "error_behaviour"):
            block = _block(fields, category, signature_is_code=signature_is_code)
            if block:
                blocks.append(block)
    return "\n\n".join(blocks).strip()


def render_l0_minus(fields: dict[str, str], category: str, *, signature_is_code: bool) -> str:
    """L0 with exactly one field block removed (the paper's single-field drop)."""
    if category not in FIELD_KEYS:
        raise ValueError(category)
    blocks: list[str] = []
    for other in FIELD_KEYS:
        if other == category:
            continue
        block = _block(fields, other, signature_is_code=signature_is_code)
        if block:
            blocks.append(block)
    return "\n\n".join(blocks).strip()


def build_levels(fields: dict[str, str], *, signature_is_code: bool) -> dict:
    """Return level texts, deletion metadata and the L2_c single-field drops."""
    levels = {
        level: render_level(fields, level, signature_is_code=signature_is_code) for level in LEVELS
    }
    presence = {name: bool(fields.get(name, "").strip()) for name in FIELD_KEYS}
    l2_c = {
        category: {
            "definition": "L0 with exactly this field removed (single-label deletion)",
            "field_present": presence[category],
            "text": render_l0_minus(fields, category, signature_is_code=signature_is_code),
        }
        for category in FIELD_KEYS
    }
    return {
        "levels": levels,
        "deleted_fields": {
            "L0": [],
            "L1": list(L1_DELETIONS),
            "L2": list(L2_DELETIONS),
            "L3": list(L3_DELETIONS),
        },
        "L2_c": l2_c,
        "field_presence": presence,
    }


def level_labels(fields: dict[str, str], level: str) -> dict[str, int]:
    """Detector labels: 1 for a present category the level deletes.

    Absent source fields are 0 (nothing was deleted); the caller can additionally use
    ``field_presence`` as a mask.  L2_c labels are one-hot on the dropped category when
    the field is present.
    """
    presence = {name: bool(fields.get(name, "").strip()) for name in FIELD_KEYS}
    if level in ("L0", "L1", "L2", "L3"):
        deleted = {
            "L0": set(),
            "L1": {"examples", "signature_types"},
            "L2": {"examples", "signature_types", "constraints", "error_behaviour"},
            "L3": {"signature_types", "examples", "constraints", "error_behaviour"},
        }[level]
    else:
        raise ValueError(level)
    return {name: int(presence[name] and name in deleted) for name in FIELD_KEYS}


def l2_c_labels(fields: dict[str, str], category: str) -> dict[str, int]:
    """Detector labels for the L2_c variant: one-hot on ``category`` when it is present."""
    if category not in FIELD_KEYS:
        raise ValueError(category)
    presence = {name: bool(fields.get(name, "").strip()) for name in FIELD_KEYS}
    return {name: int(presence[name] and name == category) for name in FIELD_KEYS}


def level_char_lengths(built: dict[str, Any]) -> dict[str, int]:
    """Return the character length of each rendered level in a ``build_levels`` result."""
    return {level: len(text) for level, text in built["levels"].items()}


__all__ = [
    "FIELD_LABELS",
    "build_levels",
    "l2_c_labels",
    "level_char_lengths",
    "level_labels",
    "render_l0_minus",
    "render_level",
]
