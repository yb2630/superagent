"""Why loop units ended without writing a solution.

The ReAct loop as run in the R2b pilot ends on the first assistant turn that carries no
parseable tool call.  Under the text transport that covers an explicit ``{"tool": "finish"}``
but also malformed or truncated JSON and plain prose, all of which are scored ``NoSolution``
when ``solution.py`` was never written.  This analysis reads the committed pilot units and
classifies each ``NoSolution`` unit by how its last generator turn ended:

``api_error``
    the unit recorded an endpoint error.
``turn_cap``
    the twelve-turn cap was hit.
``truncated_output``
    the last reply stopped at the token limit (``finish_reason == "length"``).
``explicit_finish``
    the reply was a finish object: the text parser consumes it, leaving empty content
    after a non-zero number of output tokens.
``empty_reply``
    the model produced no output tokens.
``unparseable_tool_json``
    the reply is a JSON object or ``<tool_call>`` block that does not parse; the JSON error
    category is recorded.
``prose_reply``
    any other text, typically a plan ("Let's implement ...") that never called a tool.

It also counts units in which ``run_visible_tests`` reported a pass while ``solution.py`` was
still the stub (the inputs-only visible tests can pass on the stub), and recomputes pass@1
with the harness-terminated ``NoSolution`` units excluded.  That second number is a
descriptive sensitivity check, not a corrected result: the excluded units might have failed
anyway.

Run with ``python -m wishrec.analysis.nosolution_causes``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..harness.conditions import CONDITION_ORDER
from . import ANALYSIS_DIR, dumps_deterministic, fmt

DEFAULT_UNITS = Path("results/runs/r2b-pilot/units")

#: Display names used in the paper for internal condition ids.
DISPLAY = {"DISCARD": "H3⁻"}

#: Termination causes attributable to the harness ending on an unparsed reply.
HARNESS_TERMINATION = ("truncated_output", "unparseable_tool_json", "prose_reply", "empty_reply")

CAUSES = (
    "api_error",
    "turn_cap",
    "truncated_output",
    "explicit_finish",
    "empty_reply",
    "unparseable_tool_json",
    "prose_reply",
    "other",
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TOOL_CALL_TAG = re.compile(r"</?tool_call>")
_VISIBLE_OK_PREFIX = "Visible tests passed"


def load_units(root: Path) -> list[dict[str, Any]]:
    """Load every unit JSON under ``root`` in sorted path order."""
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(root).rglob("*.json"))
    ]


def units_digest(root: Path) -> str:
    """Return a SHA-256 over the sorted relative paths and bytes of every unit file."""
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*.json")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def json_error_category(content: str) -> str:
    """Return the JSON decoder's error message without its position, for a tool-call reply."""
    text = _TOOL_CALL_TAG.sub("", content).strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        return "no JSON object"
    try:
        json.loads(text[start:])
    except json.JSONDecodeError as error:
        return error.msg
    return "valid JSON without a recognised tool call"


def _generator_calls(unit: dict[str, Any]) -> list[dict[str, Any]]:
    return [call for call in unit.get("llm_calls") or [] if call.get("purpose") == "generator"]


def classify_termination(unit: dict[str, Any]) -> tuple[str, str | None]:
    """Classify how a loop unit's last generator turn ended.

    Returns ``(cause, json_error)`` where ``json_error`` is set only for
    ``unparseable_tool_json``.
    """
    if unit.get("api_error"):
        return "api_error", None
    if unit.get("turn_cap_hit"):
        return "turn_cap", None
    calls = _generator_calls(unit)
    if not calls:
        return "other", None
    last = calls[-1]
    if last.get("tool_calls"):
        return "other", None
    if last.get("finish_reason") == "length":
        return "truncated_output", None
    content = str(last.get("content") or "").strip()
    if not content:
        output_tokens = int((last.get("usage") or {}).get("output_tokens") or 0)
        return ("explicit_finish" if output_tokens > 0 else "empty_reply"), None
    bare = _TOOL_CALL_TAG.sub("", content).strip()
    fenced = _FENCE.match(bare)
    if fenced:
        bare = fenced.group(1).strip()
    if bare.startswith("{") or "<tool_call>" in content:
        return "unparseable_tool_json", json_error_category(content)
    return "prose_reply", None


def _write_path(arguments: object) -> str | None:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if isinstance(arguments, dict):
        path = arguments.get("path")
        return str(path) if path is not None else None
    return None


def stub_visible_pass(unit: dict[str, Any]) -> bool:
    """Return whether visible tests reported a pass before ``solution.py`` was written."""
    written = False
    for call in (unit.get("tools") or {}).get("calls") or []:
        name = call.get("name")
        if (
            name == "write_file"
            and call.get("ok")
            and _write_path(call.get("arguments")) == ("solution.py")
        ):
            written = True
        elif (
            name == "run_visible_tests"
            and not written
            and str(call.get("result") or "").startswith(_VISIBLE_OK_PREFIX)
        ):
            return True
    return False


def _passed_at_1(unit: dict[str, Any]) -> bool:
    return bool(unit.get("pass_at_1", unit.get("pass")))


def _ordered(conditions: Iterable[str]) -> list[str]:
    present = set(conditions)
    known = [condition for condition in CONDITION_ORDER if condition in present]
    return known + sorted(present - set(known))


def analyse(units: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate termination causes and pass@1 per condition over completed units."""
    completed = [unit for unit in units if unit.get("status") == "complete"]
    seen: set[tuple[str, str, int]] = set()
    for unit in completed:
        key = (str(unit["condition"]), str(unit["task_key"]), int(unit.get("sample_index", 0)))
        if key in seen:
            raise ValueError(f"duplicate unit {key}")
        seen.add(key)

    per_condition: dict[str, dict[str, Any]] = {}
    totals: Counter[str] = Counter()
    total_json_errors: Counter[str] = Counter()
    for condition in _ordered(str(unit["condition"]) for unit in completed):
        selected = [unit for unit in completed if unit["condition"] == condition]
        buckets = Counter(str(unit.get("bucket")) for unit in selected)
        causes: Counter[str] = Counter()
        json_errors: Counter[str] = Counter()
        after_tool_use = 0
        nosolution_stub_pass = 0
        excluded = 0
        for unit in selected:
            if unit.get("bucket") != "NoSolution":
                continue
            cause, json_error = classify_termination(unit)
            causes[cause] += 1
            if json_error is not None:
                json_errors[json_error] += 1
            if (unit.get("tools") or {}).get("calls"):
                after_tool_use += 1
            if stub_visible_pass(unit):
                nosolution_stub_pass += 1
            if cause in HARNESS_TERMINATION:
                excluded += 1
        passes = sum(1 for unit in selected if _passed_at_1(unit))
        kept = [
            unit
            for unit in selected
            if not (
                unit.get("bucket") == "NoSolution"
                and classify_termination(unit)[0] in HARNESS_TERMINATION
            )
        ]
        per_condition[condition] = {
            "display": DISPLAY.get(condition, condition),
            "units": len(selected),
            "failures": len(selected) - passes,
            "buckets": dict(sorted(buckets.items())),
            "nosolution": buckets.get("NoSolution", 0),
            "nosolution_causes": {cause: causes[cause] for cause in CAUSES if causes[cause]},
            "nosolution_json_errors": dict(sorted(json_errors.items())),
            "nosolution_after_tool_use": after_tool_use,
            "nosolution_with_stub_visible_pass": nosolution_stub_pass,
            "units_with_stub_visible_pass": sum(1 for unit in selected if stub_visible_pass(unit)),
            "pass_at_1": round(passes / len(selected), 4) if selected else None,
            "sensitivity_excluding_harness_terminated": {
                "excluded_units": excluded,
                "units": len(kept),
                "pass_at_1": (
                    round(sum(1 for unit in kept if _passed_at_1(unit)) / len(kept), 4)
                    if kept
                    else None
                ),
            },
        }
        totals.update(causes)
        total_json_errors.update(json_errors)
    return {
        "description": (
            "Termination causes of NoSolution units in the committed pilot, and pass@1 with "
            "harness-terminated NoSolution units excluded (a descriptive sensitivity check, "
            "not a corrected result)."
        ),
        "harness_termination_causes": list(HARNESS_TERMINATION),
        "units": len(completed),
        "conditions": per_condition,
        "totals": {
            "nosolution": sum(item["nosolution"] for item in per_condition.values()),
            "nosolution_causes": {cause: totals[cause] for cause in CAUSES if totals[cause]},
            "nosolution_json_errors": dict(sorted(total_json_errors.items())),
            "nosolution_with_stub_visible_pass": sum(
                item["nosolution_with_stub_visible_pass"] for item in per_condition.values()
            ),
            "units_with_stub_visible_pass": sum(
                item["units_with_stub_visible_pass"] for item in per_condition.values()
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as a short Markdown summary."""
    shown = [cause for cause in CAUSES if report["totals"]["nosolution_causes"].get(cause)]
    lines = [
        "# Why loop units ended without a solution",
        "",
        f"{report['units']} completed pilot units. A NoSolution unit never wrote `solution.py`; "
        "the cause is how its last generator turn ended. Harness-terminated causes are "
        + ", ".join(f"`{cause}`" for cause in report["harness_termination_causes"])
        + ".",
        "",
        "| condition | failures | NoSolution | "
        + " | ".join(shown)
        + " | NoSolution after stub passed visible tests |",
        "| --- | --- | --- |" + " --- |" * len(shown) + " --- |",
    ]
    for item in report["conditions"].values():
        if not item["nosolution"]:
            continue
        cells = [str(item["nosolution_causes"].get(cause, 0)) for cause in shown]
        lines.append(
            f"| {item['display']} | {item['failures']} | {item['nosolution']} | "
            + " | ".join(cells)
            + f" | {item['nosolution_with_stub_visible_pass']} |"
        )
    totals = report["totals"]
    lines.append(
        f"| **total** | | {totals['nosolution']} | "
        + " | ".join(str(totals["nosolution_causes"].get(cause, 0)) for cause in shown)
        + f" | {totals['nosolution_with_stub_visible_pass']} |"
    )
    lines += [
        "",
        "JSON errors among `unparseable_tool_json`: "
        + (
            ", ".join(
                f"{message} ({count})"
                for message, count in totals["nosolution_json_errors"].items()
            )
            or "none"
        )
        + ".",
        "",
        "## pass@1 sensitivity (descriptive, not a corrected result)",
        "",
        "| condition | pass@1 as recorded | excluded units | pass@1 excluding them |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["conditions"].values():
        sensitivity = item["sensitivity_excluding_harness_terminated"]
        lines.append(
            f"| {item['display']} | {fmt(item['pass_at_1'])} | {sensitivity['excluded_units']} | "
            f"{fmt(sensitivity['pass_at_1'])} |"
        )
    lines += [
        "",
        (
            "Excluding units changes the denominator and assumes nothing about how they would "
            "have scored; it bounds how much the termination rule could move each condition."
        ),
        "",
    ]
    return "\n".join(lines)


def run(units_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the analysis and write ``nosolution_causes.json`` and ``.md`` to ``output_dir``."""
    report = analyse(load_units(units_dir))
    report["inputs"] = {"units": {"path": str(units_dir), "sha256": units_digest(units_dir)}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nosolution_causes.json").write_text(
        dumps_deterministic(report), encoding="utf-8"
    )
    (output_dir / "nosolution_causes.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args(argv)
    run(args.units, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
