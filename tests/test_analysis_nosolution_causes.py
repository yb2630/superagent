from __future__ import annotations

import json
from pathlib import Path

import pytest

from wishrec.analysis import nosolution_causes as nc

COMMITTED = Path("results/analysis/nosolution_causes.json")


def _unit(
    condition="H1",
    task="t/1",
    sample=0,
    *,
    bucket="NoSolution",
    passed=False,
    content="",
    output_tokens=7,
    finish_reason="stop",
    tool_calls=None,
    tools=None,
    **extra,
):
    unit = {
        "condition": condition,
        "task_key": task,
        "sample_index": sample,
        "status": "complete",
        "bucket": bucket,
        "pass": passed,
        "api_error": "",
        "turn_cap_hit": False,
        "llm_calls": [
            {
                "purpose": "generator",
                "content": content,
                "finish_reason": finish_reason,
                "tool_calls": tool_calls or [],
                "usage": {"output_tokens": output_tokens},
            }
        ],
        "tools": {"calls": tools or []},
    }
    unit.update(extra)
    return unit


@pytest.mark.parametrize(
    ("kwargs", "cause"),
    [
        ({"api_error": "boom"}, "api_error"),
        ({"turn_cap_hit": True}, "turn_cap"),
        ({"finish_reason": "length", "content": '{"tool": "write_file"'}, "truncated_output"),
        ({"content": "", "output_tokens": 7}, "explicit_finish"),
        ({"content": "", "output_tokens": 0}, "empty_reply"),
        (
            {"content": '{"tool": "write_file", "arguments": {"a": 1 "b": 2}}'},
            "unparseable_tool_json",
        ),
        (
            {"content": '<tool_call> {"name": "write_file", "x": } </tool_call>'},
            "unparseable_tool_json",
        ),
        ({"content": "```json\n{bad json}\n```"}, "unparseable_tool_json"),
        ({"content": "Let's implement the function correctly."}, "prose_reply"),
        ({"tool_calls": [{"name": "write_file"}]}, "other"),
    ],
)
def test_termination_causes(kwargs, cause):
    assert nc.classify_termination(_unit(**kwargs))[0] == cause


def test_missing_llm_calls_is_other():
    unit = _unit()
    del unit["llm_calls"]
    assert nc.classify_termination(unit) == ("other", None)


def test_json_error_category_drops_the_position():
    assert nc.json_error_category('{"a": 1 "b": 2}') == "Expecting ',' delimiter"
    assert nc.json_error_category("no braces") == "no JSON object"
    assert nc.json_error_category('{"a": 1}') == "valid JSON without a recognised tool call"


def test_stub_visible_pass_depends_on_order():
    run_pass = {"name": "run_visible_tests", "ok": True, "result": "Visible tests passed."}
    write = {"name": "write_file", "ok": True, "arguments": {"path": "solution.py"}}
    acceptance = {"name": "write_file", "ok": True, "arguments": '{"path": "acceptance_tests.py"}'}
    assert nc.stub_visible_pass(_unit(tools=[acceptance, run_pass]))
    assert not nc.stub_visible_pass(_unit(tools=[write, run_pass]))
    assert not nc.stub_visible_pass(_unit(tools=None))
    failed = {"name": "run_visible_tests", "ok": True, "result": "Visible tests failed (X)."}
    assert not nc.stub_visible_pass(_unit(tools=[failed]))


def test_analyse_aggregates_and_runs_the_sensitivity_check():
    stub_pass = [{"name": "run_visible_tests", "ok": True, "result": "Visible tests passed."}]
    units = [
        _unit("H1", "t/1", content="", output_tokens=7, tools=stub_pass),
        _unit("H1", "t/2", content="Plan: write it."),
        _unit("H1", "t/3", bucket="None", passed=True),
        _unit("H1", "t/4", bucket="AssertionError"),
        _unit("H0", "t/1", bucket="None", passed=True, pass_at_1=True),
        {**_unit("H0", "t/2"), "status": "running"},
    ]
    report = nc.analyse(units)
    h1 = report["conditions"]["H1"]
    assert h1["units"] == 4
    assert h1["failures"] == 3
    assert h1["nosolution"] == 2
    assert h1["nosolution_causes"] == {"explicit_finish": 1, "prose_reply": 1}
    assert h1["nosolution_with_stub_visible_pass"] == 1
    assert h1["pass_at_1"] == 0.25
    # Only the prose reply is harness-terminated; the explicit finish stays in.
    assert h1["sensitivity_excluding_harness_terminated"] == {
        "excluded_units": 1,
        "units": 3,
        "pass_at_1": round(1 / 3, 4),
    }
    assert report["conditions"]["H0"]["units"] == 1  # incomplete unit ignored
    assert list(report["conditions"]) == ["H0", "H1"]  # paper order
    assert report["totals"]["nosolution"] == 2


def test_display_name_for_withheld_answers_condition():
    report = nc.analyse([_unit("DISCARD", content="")])
    assert report["conditions"]["DISCARD"]["display"] == "H3⁻"


def test_duplicate_units_are_refused():
    with pytest.raises(ValueError, match="duplicate unit"):
        nc.analyse([_unit(), _unit()])


def test_run_writes_deterministic_outputs(tmp_path):
    units = tmp_path / "units" / "H1" / "0"
    units.mkdir(parents=True)
    (units / "t__1.json").write_text(json.dumps(_unit()), encoding="utf-8")
    nc.run(tmp_path / "units", tmp_path / "one")
    nc.run(tmp_path / "units", tmp_path / "two")
    for name in ("nosolution_causes.json", "nosolution_causes.md"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_committed_analysis_reproduces_from_committed_units(tmp_path):
    nc.run(nc.DEFAULT_UNITS, tmp_path)
    assert (tmp_path / "nosolution_causes.json").read_bytes() == COMMITTED.read_bytes()
    assert (tmp_path / "nosolution_causes.md").read_bytes() == COMMITTED.with_suffix(
        ".md"
    ).read_bytes()
