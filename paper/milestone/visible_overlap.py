#!/usr/bin/env python3
"""Measure the overlap between docstring examples and visible tests.

The R0 artifact describes the visible tests as a fixed subset of the hidden suite.
This script checks the other half of the design claim: that a visible test is not
also present in the ``examples`` field.  It reads only ``results/levels/tasks.jsonl``,
uses the task's ``entry_point`` and ``visible_asserts`` fields, and counts a
visible call as matching when its literal argument tuple also occurs in a call in
``fields.examples``.

Calls with non-literal arguments (for example a DataFrame variable in a
BigCodeBench test) are not comparable and are omitted rather than guessed.  The
reported rate is therefore a lower bound.

Run from the repository root:

    python3 paper/milestone/visible_overlap.py --json paper/milestone/visible_overlap.json
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "results" / "levels" / "tasks.jsonl"


def _literal(node: ast.AST) -> tuple[str, str]:
    try:
        return ("literal", repr(ast.literal_eval(node)))
    except Exception:  # noqa: BLE001 - any non-literal node is not comparable
        return ("ast", ast.dump(node, annotate_fields=False))


def _is_literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except Exception:  # noqa: BLE001
        return False


def _arguments(args: list[ast.AST], env: dict[str, ast.AST]) -> tuple[tuple[str, str], ...] | None:
    """Return a canonical argument tuple, or None when an argument is not literal."""
    result: list[tuple[str, str]] = []
    for arg in args:
        if isinstance(arg, ast.Starred):
            value = arg.value
            if isinstance(value, ast.Name) and value.id in env:
                value = env[value.id]
            if not isinstance(value, (ast.List, ast.Tuple)) or not all(
                _is_literal(item) for item in value.elts
            ):
                return None
            result.extend(_literal(item) for item in value.elts)
        elif isinstance(arg, ast.Name) and arg.id in env:
            result.append(_literal(env[arg.id]))
        elif _is_literal(arg):
            result.append(_literal(arg))
        else:
            return None
    return tuple(result)


def _calls(node: ast.AST, entry: str, env: dict[str, ast.AST]) -> list[tuple[tuple[str, str], ...]]:
    found = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == entry
        ):
            arguments = _arguments(child.args, env)
            if arguments is not None:
                found.append(arguments)
    return found


def _example_calls(text: str, entry: str) -> list[tuple[tuple[str, str], ...]]:
    env: dict[str, ast.AST] = {}
    calls: list[tuple[tuple[str, str], ...]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">>>"):
            expression = line[3:].strip()
            try:
                node = ast.parse(expression, mode="eval").body
            except SyntaxError:
                continue
            if isinstance(node, ast.Assign):
                target = node.targets[0] if len(node.targets) == 1 else None
                if isinstance(target, ast.Name) and _is_literal(node.value):
                    env[target.id] = node.value
                calls.extend(_calls(node.value, entry, env))
            elif isinstance(node, ast.Call):
                calls.extend(_calls(node, entry, env))
            elif isinstance(node, ast.Expr):
                calls.extend(_calls(node.value, entry, env))
        elif line.startswith("assert "):
            try:
                node = ast.parse(line).body[0]
            except SyntaxError:
                continue
            calls.extend(_calls(node, entry, env))
    return calls


def _visible_calls(source: str, entry: str) -> list[tuple[tuple[str, str], ...]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    calls: list[tuple[tuple[str, str], ...]] = []

    def walk(body: list[ast.stmt], env: dict[str, ast.AST]) -> None:
        local = dict(env)
        for statement in body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and _is_literal(statement.value)
            ):
                local[statement.targets[0].id] = statement.value
            calls.extend(_calls(statement, entry, local))

    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef):
            walk(statement.body, {})
        else:
            walk([statement], {})
    return calls


def _measure(rows: list[dict], split: str | None) -> dict:
    per_source: dict[str, dict] = defaultdict(
        lambda: {
            "tasks": 0,
            "tasks_with_examples": 0,
            "tasks_with_literal_visible_calls": 0,
            "tasks_with_overlap": 0,
            "literal_visible_calls": 0,
            "matching_calls": 0,
        }
    )
    for row in rows:
        if split is not None and row["split"] != split:
            continue
        source = row["source"]
        entry = row["entry_point"]
        examples = _example_calls(row["fields"].get("examples", ""), entry)
        visible = _visible_calls(row.get("visible_asserts", ""), entry)
        block = per_source[source]
        block["tasks"] += 1
        if examples:
            block["tasks_with_examples"] += 1
        if not visible:
            continue
        block["tasks_with_literal_visible_calls"] += 1
        example_set = set(examples)
        matches = sum(1 for call in visible if call in example_set)
        block["literal_visible_calls"] += len(visible)
        block["matching_calls"] += matches
        if matches:
            block["tasks_with_overlap"] += 1
    totals = {
        key: sum(block[key] for block in per_source.values())
        for key in next(iter(per_source.values()))
    }
    totals["rate"] = (
        round(totals["matching_calls"] / totals["literal_visible_calls"], 4)
        if totals["literal_visible_calls"]
        else None
    )
    by_source = {
        source: {
            **block,
            "rate": (
                round(block["matching_calls"] / block["literal_visible_calls"], 4)
                if block["literal_visible_calls"]
                else None
            ),
        }
        for source, block in sorted(per_source.items())
    }
    return {"split": split or "all", "totals": totals, "by_source": by_source}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the measurement as JSON (default: print it)",
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line]
    result = {
        "method": (
            "a visible call matches when its literal argument tuple also appears in a call "
            "in fields.examples; calls with non-literal arguments are omitted"
        ),
        "source": str(TASKS.relative_to(ROOT)),
        "all_tasks": _measure(rows, None),
        "pilot_control": _measure(rows, "pilot"),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.write_text(text, encoding="utf-8")
        print(f"wrote {args.json}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
