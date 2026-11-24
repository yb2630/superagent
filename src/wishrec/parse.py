"""Mechanical parsing of the three pinned code benchmarks into the five milestone fields.

The five fields are exactly the detector labels: intent, signature and types, examples,
constraints and edge cases, error behaviour.  Fields the source does not contain stay
empty; nothing is invented.  Hidden tests and canonical solutions are kept for build-time
validation and for the R1 harness evaluator, and are never shown to the generator or the
stakeholder.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import sys
import textwrap
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_VISIBLE_TESTS, FIELD_KEYS, VISIBLE_TEST_MODES

ERROR_CUE = re.compile(
    r"\b(raise|raises|raised|exception|exceptions|error|errors|invalid|illegal|None)\b",
    re.IGNORECASE,
)

# Benchmark test modules contain regex strings with invalid Python escapes; parsing them
# is intentional, and the SyntaxWarnings are noise rather than data-quality signals.
warnings.filterwarnings("ignore", category=SyntaxWarning)
SECTION_LABEL = re.compile(
    r"^\s*(Note|Notes|Constraints?|Preconditions?|Important|Hints?|Variables|Output|"
    r"Args|Arguments|Returns?|Raises|Exceptions?)\s*:\s*",
    re.IGNORECASE,
)
EXAMPLE_LABEL = re.compile(
    r"^\s*(?:(?:Example\s+\d+|Examples?)\s*:?\s*|For\s+example\s*:\s*)",
    re.IGNORECASE,
)

_STDLIB = set(sys.stdlib_module_names)


class ParseError(ValueError):
    """A source row cannot be turned into a complete, mechanical task."""


@dataclass
class ParsedTask:
    """One benchmark row split into specification fields, stub and test sources."""

    key: str
    source: str
    source_task_id: str
    entry_point: str
    fields: dict[str, str]
    stub: str
    visible_test: str
    visible_asserts: str
    visible_asserts_count: int
    hidden_test: str
    canonical_solution: str
    requirements: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def fields_present(self) -> dict[str, bool]:
        """Return whether each specification field has non-blank content."""
        return {name: bool(self.fields.get(name, "").strip()) for name in FIELD_KEYS}


# --------------------------------------------------------------------------------------
# Small source helpers
# --------------------------------------------------------------------------------------
def _find_function(tree: ast.AST, name: str | None = None) -> ast.FunctionDef | None:
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if name is not None:
        for node in candidates:
            if node.name == name:
                return node
    return candidates[0] if candidates else None


def _function_signature_source(source: str, fn: ast.FunctionDef) -> str:
    """Return the source of a def header (possibly multi-line), through its colon."""
    lines = source.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(rf"^\s*(async\s+)?def\s+{re.escape(fn.name)}\s*\(", line):
            start = index
            break
    if start is None:
        return ""
    depth = 0
    chunks: list[str] = []
    for line in lines[start:]:
        chunks.append(line)
        for char in line:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
        if depth == 0 and line.rstrip().endswith(":"):
            break
    return textwrap.dedent("\n".join(chunks)).strip()


def _strip_annotations(def_source: str) -> str:
    """Strip annotations and one-line the def header, keeping defaults and names."""
    synthetic = def_source.rstrip().rstrip(":") + ":\n    pass\n"
    tree = ast.parse(synthetic)
    fn = _find_function(tree)
    if not isinstance(fn, ast.FunctionDef):
        raise ParseError(f"could not parse def header: {def_source!r}")
    fn.returns = None
    fn.decorator_list = []
    for argument in list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs):
        argument.annotation = None
    if fn.args.vararg is not None:
        fn.args.vararg.annotation = None
    if fn.args.kwarg is not None:
        fn.args.kwarg.annotation = None
    ast.fix_missing_locations(fn)
    return ast.unparse(fn).split("\n", 1)[0].strip()


def _top_level_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    return [
        ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def _imported_modules(source: str) -> set[str]:
    """All top-level module names imported anywhere in ``source``."""
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def _requirements(*sources: str) -> list[str]:
    modules: set[str] = set()
    for source in sources:
        if not source.strip():
            continue
        modules |= _imported_modules(source)
    third_party = {
        module
        for module in modules
        if module not in _STDLIB and module not in {"solution", "__future__"}
    }
    return sorted(third_party)


def _top_level_preamble(source: str, entry_point: str) -> list[str]:
    """Top-level imports/constants/helpers the entry function may rely on.

    BigCodeBench's official harness concatenates the solution and the test module, so
    module-level constants and imports are part of the solution's scaffolding.  The stub
    keeps them, with the entry function's body replaced by ``pass``.
    """
    tree = ast.parse(source)
    return [
        ast.unparse(node)
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name == entry_point)
    ]


def make_stub(preamble: list[str], signature_untyped: str) -> str:
    """Return the agent's starting file: the preamble plus an unimplemented signature."""
    parts = []
    if preamble:
        parts.append("\n\n".join(preamble))
    parts.append(signature_untyped.rstrip() + "\n    # TODO: implement this function.\n    pass")
    return "\n\n".join(parts) + "\n"


# --------------------------------------------------------------------------------------
# Prose and docstring parsing
# --------------------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    protected = re.sub(r"\b(e\.g|i\.e|etc|vs|Mr|Mrs|Dr)\.", r"\1<DOT>", text)
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", protected)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    out = []
    for part in parts:
        sentence = part.replace("<DOT>", ".").strip()
        if sentence:
            out.append(sentence)
    return out


def _strip_section_label(sentence: str) -> tuple[str, bool]:
    match = SECTION_LABEL.match(sentence)
    if not match:
        return sentence, False
    return sentence[match.end() :].strip(), True


INTENT_VERBS = re.compile(
    r"\b(?:return|create|write|compute|find|count|implement)\w*\b",
    re.IGNORECASE,
)
MIN_INTENT_TOKENS = 5


def _validate_intent(intent: str) -> None:
    """Reject a wish that is too short or contains no requirement verb."""
    if not intent:
        raise ParseError("empty intent after docstring parsing")
    if len(intent.split()) < MIN_INTENT_TOKENS:
        raise ParseError(f"intent has fewer than {MIN_INTENT_TOKENS} tokens: {intent!r}")
    if not INTENT_VERBS.search(intent):
        raise ParseError(f"intent has no requirement verb: {intent!r}")


def _classify_prose(prose: str) -> tuple[str, str, str]:
    """Return (intent, constraints, error_behaviour) from the prose block.

    The intent is the whole prose block before the examples, with newlines collapsed to
    spaces.  Later unlabelled sentences are part of that request, not hidden constraints;
    constraints and error behaviour are extra classifications of sentences that carry an
    explicit label or an error cue.  In particular, the first sentence is never allowed
    to swallow a requirement that the following prose states.
    """
    sentences = _split_sentences(prose)
    if not sentences:
        return "", "", ""
    intent = re.sub(r"\s+", " ", prose).strip()
    intent = re.sub(r"^Task\s*:?\s*", "", intent, flags=re.IGNORECASE).strip()
    constraints: list[str] = []
    errors: list[str] = []
    for raw in sentences:
        sentence, from_label = _strip_section_label(raw)
        if not sentence:
            continue
        if from_label and re.search(r"\b(Raises?|Exceptions?)\b", raw, re.IGNORECASE):
            errors.append(sentence)
        elif from_label and re.search(
            r"\b(Note|Notes|Constraints?|Preconditions?)\b", raw, re.IGNORECASE
        ):
            constraints.append(sentence)
        elif ERROR_CUE.search(sentence):
            errors.append(sentence)
        # Unlabelled prose is the request: it stays in the intent, not in constraints.
    return (
        intent,
        "\n".join(f"- {item}" for item in constraints),
        "\n".join(f"- {item}" for item in errors),
    )


def _extract_docstring_parts(docstring: str) -> tuple[str, str]:
    """Split a docstring into (prose, examples).

    Handles ``>>>`` doctests and ``Examples:`` / ``Example N:`` / ``For example:``
    sections.  The examples field collects the example text verbatim (minus indentation);
    the prose is passed to :func:`_classify_prose` with its section labels intact, so
    explicit labels can still be recognised as constraints or error behaviour.
    """
    doc = textwrap.dedent(docstring).strip("\n")
    lines = doc.splitlines()
    prose: list[str] = []
    examples: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith(">>>"):
            first = True
            while index < len(lines):
                current = lines[index]
                current_stripped = current.strip()
                if not first and (
                    not current_stripped
                    or SECTION_LABEL.match(current)
                    or EXAMPLE_LABEL.match(current)
                ):
                    break
                examples.append(current_stripped)
                first = False
                index += 1
            continue
        label_match = EXAMPLE_LABEL.match(line)
        if label_match:
            remainder = line[label_match.end() :].strip()
            if remainder:
                examples.append(remainder)
            index += 1
            while index < len(lines):
                current = lines[index]
                current_stripped = current.strip()
                next_nonblank = ""
                for later in range(index + 1, len(lines)):
                    if lines[later].strip():
                        next_nonblank = lines[later]
                        break
                if (
                    current_stripped
                    and SECTION_LABEL.match(current)
                    and not EXAMPLE_LABEL.match(current)
                ):
                    break
                if current_stripped == "" and next_nonblank and SECTION_LABEL.match(next_nonblank):
                    break
                examples.append(current_stripped)
                index += 1
            continue
        prose.append(line)
        index += 1
    prose_text = "\n".join(prose)
    return prose_text.strip(), "\n".join(examples).strip("\n")


def _validate_visible_tests(visible_tests: str) -> None:
    if visible_tests not in VISIBLE_TEST_MODES:
        raise ValueError(
            f"visible_tests must be one of {VISIBLE_TEST_MODES}, got {visible_tests!r}"
        )


def _humaneval_visible_test(
    test_source: str,
    entry_point: str,
    cases: int = 3,
    *,
    visible_tests: str = DEFAULT_VISIBLE_TESTS,
) -> tuple[str, str, int]:
    """Return (visible test source, visible assertion text, case count).

    ``full`` keeps the benchmark's expected results in the visible file.  ``inputs_only``
    calls the entry point on the same first ``cases`` inputs and prints each observed
    result, so the agent never sees the expected values.
    """
    _validate_visible_tests(visible_tests)
    tree = ast.parse(test_source)
    check = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check"),
        None,
    )
    if check is None:
        raise ParseError("HumanEval+ test has no check() function")
    inputs: list[ast.expr] | None = None
    results: list[ast.expr] | None = None
    for node in ast.walk(check):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name in ("inputs", "results") and isinstance(node.value, ast.List):
                if name == "inputs":
                    inputs = node.value.elts
                    node.value.elts = node.value.elts[:cases]
                else:
                    results = node.value.elts
                    node.value.elts = node.value.elts[:cases]
    if not inputs or not results:
        raise ParseError("HumanEval+ test has no literal inputs/results lists")
    if len(inputs) < cases or len(results) < cases:
        raise ParseError(f"HumanEval+ test has fewer than {cases} cases")
    truncated = copy.deepcopy(tree)
    inputs_assign = None
    if visible_tests == "inputs_only":
        copied_check = next(
            node
            for node in truncated.body
            if isinstance(node, ast.FunctionDef) and node.name == "check"
        )
        inputs_assign = next(
            node
            for node in copied_check.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "inputs"
        )
        # Keep the literal inputs and imports, but drop the expected results and all
        # comparisons that would reveal them.  The test only exercises the entry point.
        copied_check.body = [
            inputs_assign,
            ast.parse("for inp in inputs:\n    print(candidate(*inp))").body[0],
        ]
    truncated.body.append(
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id="check", ctx=ast.Load()),
                args=[ast.Name(id=entry_point, ctx=ast.Load())],
                keywords=[],
            )
        )
    )
    ast.fix_missing_locations(truncated)
    visible_source = ast.unparse(truncated)
    visible = f"from solution import {entry_point}\n{visible_source}\n"
    if visible_tests == "full":
        assertions = "\n".join(
            f"assertion({entry_point}(*{ast.unparse(inp)}), {ast.unparse(res)}, 0)"
            for inp, res in zip(inputs[:cases], results[:cases], strict=True)
        )
    elif inputs_assign is not None:
        assertions = (
            f"for inp in {ast.unparse(inputs_assign.value)}:\n    print({entry_point}(*inp))"
        )
    else:
        raise ValueError(f"unhandled visible_tests mode {visible_tests!r}")
    return visible, assertions, cases


def _mbpp_imports(test_imports: list[str], asserts: str) -> list[str]:
    imports = [str(item).strip() for item in test_imports if str(item).strip()]
    inferred_modules = {
        "sys": "import sys",
        "math": "import math",
        "re": "import re",
        "collections": "import collections",
        "itertools": "import itertools",
        "functools": "import functools",
        "heapq": "import heapq",
        "random": "import random",
        "string": "import string",
        "datetime": "import datetime",
        "operator": "import operator",
        "numpy": "import numpy as np",
        "pandas": "import pandas as pd",
    }
    for module, statement in inferred_modules.items():
        alias = "np" if module == "numpy" else "pd" if module == "pandas" else module
        if re.search(rf"\b{re.escape(alias)}\.", asserts) and not re.search(
            rf"\b(import\s+{re.escape(module)}|from\s+{re.escape(module)}\s+import)",
            "\n".join(imports),
        ):
            imports.append(statement)
    return imports


def _mbpp_entry_point(test_source: str, canonical_code: str) -> str:
    tree = ast.parse(test_source)
    names: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assertion"
            and node.args
        ):
            first = node.args[0]
            if isinstance(first, ast.Call) and isinstance(first.func, ast.Name):
                names.append(first.func.id)
    code_tree = ast.parse(canonical_code)
    defined = {node.name for node in ast.walk(code_tree) if isinstance(node, ast.FunctionDef)}
    for name in names:
        if name in defined:
            return name
    for name in names:
        return name
    raise ParseError("MBPP+ hidden test does not call assertion(entry(...))")


# --------------------------------------------------------------------------------------
# HumanEval+
# --------------------------------------------------------------------------------------
def parse_humanevalplus(row: dict, *, visible_tests: str = DEFAULT_VISIBLE_TESTS) -> ParsedTask:
    """Parse one HumanEval+ row; raise ``ParseError`` when it cannot be made mechanical."""
    _validate_visible_tests(visible_tests)
    source_task_id = str(row["task_id"])
    entry_point = str(row["entry_point"])
    prompt = str(row["prompt"])
    prompt_tree = ast.parse(prompt)
    fn = _find_function(prompt_tree, entry_point)
    if not isinstance(fn, ast.FunctionDef):
        raise ParseError(f"entry point {entry_point!r} not found in prompt")
    typed_def = _function_signature_source(prompt, fn)
    if not typed_def:
        raise ParseError("could not locate the HumanEval+ signature")
    imports = _top_level_imports(prompt)
    signature_types = "\n\n".join([*imports, typed_def]) if imports else typed_def
    signature_untyped = _strip_annotations(typed_def)
    docstring = ast.get_docstring(fn, clean=True) or ""
    prose, examples = _extract_docstring_parts(docstring)
    intent, constraints, errors = _classify_prose(prose)
    _validate_intent(intent)

    test_source = str(row["test"])
    if "ref_func" in test_source:
        # The visible test is derived from the same test module; a reference
        # implementation in that module would leak a solution into the visible tests.
        raise ParseError(
            "hidden test embeds ref_func; visible tests would leak a reference implementation"
        )
    visible, visible_asserts, count = _humaneval_visible_test(
        test_source, entry_point, visible_tests=visible_tests
    )
    hidden = f"from solution import {entry_point}\n{test_source}\ncheck({entry_point})\n"
    canonical = (prompt.rstrip("\n") + "\n" + str(row["canonical_solution"])).strip("\n") + "\n"
    fields = {
        "intent": intent,
        "signature_types": signature_types,
        "signature_untyped": signature_untyped,
        "examples": examples,
        "constraints": constraints,
        "error_behaviour": errors,
    }
    return ParsedTask(
        key=f"humanevalplus/{source_task_id}",
        source="humanevalplus",
        source_task_id=source_task_id,
        entry_point=entry_point,
        fields=fields,
        stub=make_stub(_top_level_preamble(prompt, entry_point), signature_untyped),
        visible_test=visible,
        visible_asserts=visible_asserts,
        visible_asserts_count=count,
        hidden_test=hidden,
        canonical_solution=canonical,
        requirements=_requirements(prompt, canonical, test_source),
        meta={
            "visible_test_cases": f"check() on first {count} inputs",
            "visible_tests": visible_tests,
            "docstring_chars": len(docstring),
        },
    )


# --------------------------------------------------------------------------------------
# MBPP+
# --------------------------------------------------------------------------------------
def _first_entry_call(tree: ast.AST, entry_point: str) -> ast.Call | None:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == entry_point
        ):
            return node
    return None


def _mbpp_inputs_only_asserts(assertions: list[str], entry_point: str) -> str:
    """Turn the first three MBPP+ assertions into ``print(entry(...))`` calls."""
    lines = []
    for assertion in assertions:
        call = _first_entry_call(ast.parse(assertion), entry_point)
        if call is None:
            raise ParseError(
                f"MBPP+ visible assertion does not call {entry_point!r}: {assertion!r}"
            )
        lines.append(f"print({ast.unparse(call)})")
    return "\n".join(lines)


def parse_mbppplus(row: dict, *, visible_tests: str = DEFAULT_VISIBLE_TESTS) -> ParsedTask:
    """Parse one MBPP+ row; raise ``ParseError`` when it cannot be made mechanical."""
    _validate_visible_tests(visible_tests)
    source_task_id = str(row["task_id"])
    canonical = str(row["code"]).strip("\n") + "\n"
    prompt = str(row["prompt"]).strip()
    test_source = str(row["test"])
    entry_point = _mbpp_entry_point(test_source, canonical)
    code_tree = ast.parse(canonical)
    fn = _find_function(code_tree, entry_point)
    if not isinstance(fn, ast.FunctionDef):
        raise ParseError(f"entry point {entry_point!r} not found in canonical code")
    signature_types = _function_signature_source(canonical, fn)
    if not signature_types:
        raise ParseError("could not locate the MBPP+ canonical signature")
    signature_untyped = _strip_annotations(signature_types)
    sentences = _split_sentences(prompt)
    intent = sentences[0] if sentences else prompt
    constraints: list[str] = []
    errors: list[str] = []
    for sentence in sentences[1:]:
        if ERROR_CUE.search(sentence):
            errors.append(sentence)
        else:
            constraints.append(sentence)
    test_list = [str(item).strip() for item in row.get("test_list") or []]
    if len(test_list) < 3:
        raise ParseError("MBPP+ task has fewer than three test_list assertions")
    test_imports = [
        str(item).strip() for item in row.get("test_imports") or [] if str(item).strip()
    ]
    examples = "\n".join(test_list)
    if visible_tests == "inputs_only":
        visible_asserts = _mbpp_inputs_only_asserts(test_list[:3], entry_point)
    else:
        visible_asserts = "\n".join(test_list[:3])
    visible_imports = _mbpp_imports(test_imports, visible_asserts)
    visible_lines = [f"from solution import {entry_point}"]
    visible_lines.extend(visible_imports)
    visible_lines.append(visible_asserts)
    hidden_lines = [f"from solution import {entry_point}", "candidate = " + entry_point]
    hidden_lines.extend(test_imports)
    hidden_lines.append(test_source)
    fields = {
        "intent": intent,
        "signature_types": signature_types,
        "signature_untyped": signature_untyped,
        "examples": examples,
        "constraints": "\n".join(f"- {item}" for item in constraints),
        "error_behaviour": "\n".join(f"- {item}" for item in errors),
    }
    return ParsedTask(
        key=f"mbppplus/{source_task_id}",
        source="mbppplus",
        source_task_id=source_task_id,
        entry_point=entry_point,
        fields=fields,
        stub=make_stub(_top_level_preamble(canonical, entry_point), signature_untyped),
        visible_test="\n".join(visible_lines) + "\n",
        visible_asserts=visible_asserts,
        visible_asserts_count=3,
        hidden_test="\n".join(hidden_lines) + "\n",
        canonical_solution=canonical,
        requirements=_requirements(canonical, test_source, visible_asserts),
        meta={
            "prompt": prompt,
            "source_file": str(row.get("source_file", "")),
            "visible_test_cases": "test_list[0:3]",
            "visible_tests": visible_tests,
            "test_list_size": len(test_list),
        },
    )


# --------------------------------------------------------------------------------------
# BigCodeBench
# --------------------------------------------------------------------------------------
def _bcb_field_text(items: str | Iterable[object] | None) -> list[str]:
    if not items:
        return []
    if isinstance(items, str):
        return [items]
    return [str(item).strip() for item in items if str(item).strip()]


def _split_bcb_description(raw: list[str]) -> tuple[str, list[str], list[str]]:
    """Split the few doc_struct descriptions that mix Args/doctest lines into fields.

    Six BigCodeBench rows carry their argument descriptions (and a couple carry doctest
    lines) inside ``description`` while leaving ``params`` empty.  Keeping those lines in
    the intent would put an argument list in the wish, so they are moved to their proper
    field; when ``params``/``examples`` are already populated they win and the extras are
    dropped as duplicates.
    """
    intent_lines: list[str] = []
    extra_params: list[str] = []
    extra_examples: list[str] = []
    in_args = False
    in_examples = False
    for line in raw:
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered in ("args:", "arguments:") or lowered.startswith(("args:", "arguments:")):
            in_args = True
            in_examples = False
            continue
        if stripped.startswith((">>>", "...")):
            extra_examples.append(stripped)
            in_args = False
            in_examples = True
            continue
        if in_examples:
            extra_examples.append(stripped)
            continue
        if in_args:
            extra_params.append(stripped.lstrip("- ").strip())
            continue
        intent_lines.append(stripped)
    intent = " ".join(intent_lines)
    intent = re.sub(r"\s+", " ", intent).strip()
    return intent, extra_params, extra_examples


def _bcb_doc_fields(doc: dict) -> dict[str, str]:
    description_lines = _bcb_field_text(doc.get("description"))
    description, extra_params, extra_examples = _split_bcb_description(description_lines)
    params = _bcb_field_text(doc.get("params")) or extra_params
    returns = _bcb_field_text(doc.get("returns"))
    signature_parts = []
    if params:
        signature_parts.append("Args:\n" + "\n".join(f"- {item}" for item in params))
    if returns:
        signature_parts.append("Returns:\n" + "\n".join(f"- {item}" for item in returns))
    constraints_parts = []
    notes = _bcb_field_text(doc.get("notes"))
    reqs = _bcb_field_text(doc.get("reqs"))
    if notes:
        constraints_parts.append("Notes:\n" + "\n".join(f"- {item}" for item in notes))
    if reqs:
        constraints_parts.append("Requirements:\n" + "\n".join(f"- {item}" for item in reqs))
    raises = _bcb_field_text(doc.get("raises"))
    return {
        "intent": description,
        "signature_types": "\n\n".join(signature_parts),
        "examples": "\n".join(_bcb_field_text(doc.get("examples")) or extra_examples),
        "constraints": "\n\n".join(constraints_parts),
        "error_behaviour": "\n".join(f"- {item}" for item in raises),
    }


def _bcb_main_test_class(tree: ast.Module) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(
            isinstance(child, ast.FunctionDef) and child.name.startswith("test_")
            for child in node.body
        ):
            return node
    raise ParseError("BigCodeBench test has no unittest class with test_* methods")


def _bcb_visible_test(test_source: str, entry_point: str) -> tuple[str, str, list[str], int]:
    tree = ast.parse(test_source)
    main_class = _bcb_main_test_class(tree)
    tests = [
        child
        for child in main_class.body
        if isinstance(child, ast.FunctionDef) and child.name.startswith("test_")
    ]
    if len(tests) < 3:
        raise ParseError(f"BigCodeBench task has only {len(tests)} test_* methods")
    selected = tests[:3]
    new_class = copy.deepcopy(main_class)
    source_body = main_class.body
    selected_ids = {id(node) for node in selected}
    new_body: list[ast.stmt] = []
    for original, copied in zip(source_body, new_class.body, strict=True):
        if isinstance(original, ast.FunctionDef) and original.name.startswith("test_"):
            if id(original) in selected_ids:
                new_body.append(copied)
        else:
            new_body.append(copied)
    new_class.body = new_body
    module_body: list[ast.stmt] = []
    for node in tree.body:
        if node is main_class:
            module_body.append(new_class)
        elif isinstance(node, ast.ClassDef) and any(
            isinstance(child, ast.FunctionDef) and child.name.startswith("test_")
            for child in node.body
        ):
            # Other test classes are part of the hidden suite, not the visible three.
            continue
        else:
            module_body.append(copy.deepcopy(node))
    module = ast.Module(body=module_body, type_ignores=[])
    ast.fix_missing_locations(module)
    visible_source = ast.unparse(module)
    visible = (
        f"from solution import {entry_point}\n"
        "from solution import *  # BigCodeBench official harness runs test and solution\n"
        "                          # in one namespace; this keeps module constants resolved.\n"
        f"{visible_source}\n"
    )
    assertions = "\n\n".join(ast.unparse(node) for node in selected)
    return visible, assertions, [node.name for node in selected], len(tests)


def parse_bigcodebench(row: dict, *, visible_tests: str = DEFAULT_VISIBLE_TESTS) -> ParsedTask:
    """Parse one BigCodeBench task.

    The ``visible_tests`` mode is recorded for provenance, but BigCodeBench's visible
    suite is a selection of ``unittest`` methods whose expectations are assertions rather
    than literal ``(inputs, results)`` pairs, so it is kept as-is in both modes.
    """
    _validate_visible_tests(visible_tests)
    source_task_id = str(row["task_id"])
    entry_point = str(row["entry_point"])
    complete_prompt = str(row["complete_prompt"])
    prompt_tree = ast.parse(complete_prompt)
    fn = _find_function(prompt_tree, entry_point)
    if not isinstance(fn, ast.FunctionDef):
        raise ParseError(f"entry point {entry_point!r} not found in complete_prompt")
    typed_def = _function_signature_source(complete_prompt, fn)
    signature_untyped = _strip_annotations(typed_def)
    stub = make_stub(_top_level_preamble(complete_prompt, entry_point), signature_untyped)
    canonical = (
        complete_prompt.rstrip("\n") + "\n" + str(row["canonical_solution"]).rstrip("\n") + "\n"
    )
    try:
        doc = json.loads(row["doc_struct"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ParseError(f"doc_struct is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ParseError("doc_struct is not a JSON object")
    doc_fields = _bcb_doc_fields(doc)
    if not doc_fields["intent"]:
        raise ParseError("empty BigCodeBench description")
    test_source = str(row["test"])
    visible, visible_asserts, method_names, total_tests = _bcb_visible_test(
        test_source, entry_point
    )
    hidden = (
        f"from solution import {entry_point}\n"
        "from solution import *  # BigCodeBench official harness runs test and solution\n"
        "                          # in one namespace; this keeps module constants resolved.\n"
        f"{test_source.rstrip()}\n"
    )
    try:
        libs = json.loads(row.get("libs") or "[]")
    except (TypeError, json.JSONDecodeError):
        libs = []
    fields = {
        "intent": doc_fields["intent"],
        "signature_types": doc_fields["signature_types"],
        "signature_untyped": signature_untyped,
        "examples": doc_fields["examples"],
        "constraints": doc_fields["constraints"],
        "error_behaviour": doc_fields["error_behaviour"],
    }
    return ParsedTask(
        key=f"bigcodebench/{source_task_id}",
        source="bigcodebench",
        source_task_id=source_task_id,
        entry_point=entry_point,
        fields=fields,
        stub=stub,
        visible_test=visible,
        visible_asserts=visible_asserts,
        visible_asserts_count=3,
        hidden_test=hidden,
        canonical_solution=canonical,
        requirements=_requirements(complete_prompt, canonical, test_source),
        meta={
            "libs": [str(item) for item in libs] if isinstance(libs, list) else [],
            "visible_test_cases": method_names,
            "visible_tests": visible_tests,
            "test_method_count": total_tests,
            "doc_raises_count": len(_bcb_field_text(doc.get("raises"))),
        },
    )


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def parse_task(source: str, row: dict, *, visible_tests: str = DEFAULT_VISIBLE_TESTS) -> ParsedTask:
    """Dispatch a raw row to its source-specific parser."""
    _validate_visible_tests(visible_tests)
    if source == "humanevalplus":
        return parse_humanevalplus(row, visible_tests=visible_tests)
    if source == "mbppplus":
        return parse_mbppplus(row, visible_tests=visible_tests)
    if source == "bigcodebench":
        return parse_bigcodebench(row, visible_tests=visible_tests)
    raise ParseError(f"unknown source {source!r}")


def target_tests(task: ParsedTask) -> tuple[str, str]:
    """Return (visible runner, hidden runner) command styles for validation."""
    if task.source == "bigcodebench":
        return ("unittest", "unittest")
    return ("script", "script")
