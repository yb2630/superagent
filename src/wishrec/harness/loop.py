"""The ReAct loop and the direct-sample conditions.

Every loop condition shares one tool session type, one turn cap (12) and one per-task
workspace.  H3/DISCARD ask a detector-selected 2-3 questions before the loop starts; H2
lets the agent ask through ``ask_stakeholder`` within its five-question budget.  H3's
answers are also used by one self-check-and-repair call.  The hidden evaluation runs only
after the loop, in a fresh workspace containing solution.py and the hidden suite.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .config import (
    GENERATOR_MAX_TOKENS,
    GENERATOR_SEED,
    GENERATOR_TEMPERATURE,
    GENERATOR_TOP_P,
    TURN_CAP,
    utc_now_iso,
)
from .conditions import Condition
from .detector import questions_for
from .execution import (
    HIDDEN_TIMEOUT,
    classify_exception,
    run_hidden_suite,
)
from .llm import ChatResult, extract_code_block
from .prompts import (
    NOT_SPECIFIED,
    direct_messages,
    frontier_messages,
    initial_messages,
    self_check_messages,
    stakeholder_messages,
)
from .stakeholder import Stakeholder
from .tools import ToolSession, tool_schemas


def _totals() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "cached_calls": 0,
    }


def _accumulate(totals: dict, result: ChatResult) -> None:
    totals["calls"] += 1
    usage = result.usage or {}
    totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
    totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
    totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    if getattr(result, "cached", False):
        totals["cached_calls"] += 1


def _call_record(result: ChatResult, purpose: str) -> dict:
    return {
        "purpose": purpose,
        "model": result.model,
        "transport": result.transport,
        "finish_reason": result.finish_reason,
        "content": (result.content or "")[:3000],
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments[:2000]} for call in result.tool_calls
        ],
        "usage": dict(result.usage or {}),
    }


def evaluate_solution(task, solution_source: str, *, timeout: float = HIDDEN_TIMEOUT) -> dict:
    """Run the full hidden suite in a fresh workspace with only the two files."""
    with tempfile.TemporaryDirectory(prefix="wish-eval-") as temporary:
        workspace = Path(temporary)
        (workspace / "solution.py").write_text(solution_source, encoding="utf-8")
        (workspace / "hidden_test.py").write_text(task.hidden_test, encoding="utf-8")
        result = run_hidden_suite(task.source, workspace, timeout)
    bucket = "None" if result.ok else classify_exception(result)
    return {
        "pass": bool(result.ok),
        "bucket": bucket,
        "exception": "" if result.ok else bucket,
        "output_tail": result.combined()[-3000:],
        "duration_s": round(result.duration_s, 3),
        "timed_out": result.timed_out,
    }


def _assistant_message(result: ChatResult) -> dict:
    message: dict = {"role": "assistant", "content": result.content or ""}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in result.tool_calls
        ]
    return message


def _detector_prepass(
    task,
    condition: Condition,
    llm,
    *,
    detector,
    sample_index: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    """Ask the detector-selected questions; answers are withheld for DISCARD."""
    selected = detector.select_questions(task.wish)
    question_records = questions_for(selected)
    answers: list[dict] = []
    withheld: list[dict] = []
    ask_records: list[dict] = []
    calls: list[dict] = []
    totals = _totals()
    unanswered: list[str] = []
    for item in question_records:
        result = llm.chat(
            stakeholder_messages(task.levels["L0"], item["question"]),
            purpose="stakeholder",
            meta={
                "condition": condition.name,
                "task_key": task.key,
                "sample_index": sample_index,
                "detector_category": item["category"],
            },
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
            seed=0,
        )
        answer = (result.content or "").strip() or NOT_SPECIFIED
        _accumulate(totals, result)
        calls.append(_call_record(result, "stakeholder"))
        record = {
            "category": item["category"],
            "label": item["label"],
            "score": item["score"],
            "question": item["question"],
            "answer": None if condition.discard else answer,
        }
        ask_records.append(record)
        if condition.discard:
            withheld.append(record)
        else:
            answers.append(record)
            if answer == NOT_SPECIFIED:
                unanswered.append(item["category"])
    return answers, withheld, ask_records, calls, {"totals": totals, "unanswered": unanswered}


def run_repl_unit(
    task,
    condition: Condition,
    llm,
    *,
    detector=None,
    siblings: list[dict] | None = None,
    seed: int = GENERATOR_SEED,
    run_id: str = "",
    sample_index: int = 0,
) -> dict:
    started = time.monotonic()
    started_at = utc_now_iso()
    totals = _totals()
    calls: list[dict] = []
    trace_events: list[dict] = []
    answers: list[dict] = []
    withheld: list[dict] = []
    detector_questions: list[dict] = []
    unanswered: list[str] = []
    api_error = ""

    if condition.targeted:
        if detector is None:
            raise ValueError(f"{condition.name} requires a trained detector")
        answers, withheld, detector_questions, prepass_calls, prepass = _detector_prepass(
            task, condition, llm, detector=detector, sample_index=sample_index
        )
        calls.extend(prepass_calls)
        for key in totals:
            totals[key] += prepass["totals"][key]
        unanswered = prepass["unanswered"]
        trace_events.append({"event": "detector_prepass", "questions": detector_questions})

    with tempfile.TemporaryDirectory(prefix="wish-loop-") as temporary:
        workspace = Path(temporary)
        (workspace / "solution.py").write_text(task.stub, encoding="utf-8")
        (workspace / "visible_test.py").write_text(task.visible_test, encoding="utf-8")

        ask_fn = None
        if condition.dialogue and condition.question_budget > 0:
            stakeholder = Stakeholder(llm, task.levels["L0"])
            counter = {"n": 0}

            def ask_fn(question: str) -> str:  # noqa: ANN001
                counter["n"] += 1
                return stakeholder.answer(
                    question,
                    meta={
                        "condition": condition.name,
                        "task_key": task.key,
                        "sample_index": sample_index,
                        "follow_up": counter["n"],
                    },
                )

        session = ToolSession(
            workspace=workspace,
            source=task.source,
            allow_acceptance=condition.acceptance_first,
            ask_fn=ask_fn,
            ask_budget=condition.question_budget,
        )
        messages = initial_messages(
            task,
            condition,
            answers=answers if condition.targeted and not condition.discard else None,
            unanswered=unanswered if condition.targeted and not condition.discard else None,
            withheld=withheld if condition.discard else None,
            siblings=siblings if condition.siblings else None,
        )

        turns = 0
        turn_cap_hit = False
        last_had_calls = False
        while turns < TURN_CAP:
            try:
                result = llm.chat(
                    messages,
                    tools=session.schemas(),
                    purpose="generator",
                    meta={
                        "condition": condition.name,
                        "task_key": task.key,
                        "sample_index": sample_index,
                        "turn": turns + 1,
                    },
                    temperature=GENERATOR_TEMPERATURE,
                    top_p=GENERATOR_TOP_P,
                    max_tokens=GENERATOR_MAX_TOKENS,
                    seed=seed,
                )
            except Exception as exc:  # noqa: BLE001 - a failed unit is an artifact
                api_error = f"{type(exc).__name__}: {exc}"
                trace_events.append({"event": "api_error", "error": api_error})
                break
            turns += 1
            _accumulate(totals, result)
            calls.append(_call_record(result, "generator"))
            messages.append(_assistant_message(result))
            last_had_calls = bool(result.tool_calls)
            if not result.tool_calls:
                break
            for tool_call in result.tool_calls:
                output = session.execute(tool_call.name, tool_call.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    }
                )
        if turns >= TURN_CAP and last_had_calls:
            turn_cap_hit = True
            trace_events.append({"event": "turn_cap_hit", "turns": turns})

        self_check = None
        if condition.name == "H3" and answers:
            solution_before = (workspace / "solution.py").read_text(encoding="utf-8")
            check_tools = [
                schema for schema in tool_schemas() if schema["function"]["name"] == "write_file"
            ]
            try:
                check_result = llm.chat(
                    self_check_messages(solution_before, answers),
                    tools=check_tools,
                    purpose="self_check",
                    meta={
                        "condition": condition.name,
                        "task_key": task.key,
                        "sample_index": sample_index,
                    },
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=GENERATOR_MAX_TOKENS,
                    seed=seed,
                )
                _accumulate(totals, check_result)
                calls.append(_call_record(check_result, "self_check"))
                wrote = False
                for tool_call in check_result.tool_calls:
                    if tool_call.name == "write_file":
                        output = session.execute(tool_call.name, tool_call.arguments)
                        wrote = wrote or output.startswith("Wrote")
                self_check = {
                    "content": (check_result.content or "")[:2000],
                    "tool_calls": [call.name for call in check_result.tool_calls],
                    "wrote": wrote,
                }
                trace_events.append({"event": "self_check", **self_check})
            except Exception as exc:  # noqa: BLE001
                api_error = api_error or f"self_check {type(exc).__name__}: {exc}"
                trace_events.append({"event": "self_check_error", "error": str(exc)[:1000]})

        final_solution = (workspace / "solution.py").read_text(encoding="utf-8")
        evaluation = evaluate_solution(task, final_solution)
        if not session.solution_written and not evaluation["pass"]:
            evaluation["bucket"] = "NoSolution"
        session_snapshot = session.snapshot()

    if api_error:
        trace_events.append({"event": "api_error_final", "error": api_error})

    questions_asked = len(detector_questions) + session_snapshot["ask_used"]
    return {
        "condition": condition.name,
        "task_key": task.key,
        "source": task.source,
        "source_task_id": task.source_task_id,
        "entry_point": task.entry_point,
        "sample_index": sample_index,
        "seed": seed,
        "run_id": run_id,
        "pass": bool(evaluation["pass"]),
        "bucket": evaluation["bucket"],
        "hidden_output_tail": evaluation["output_tail"],
        "hidden_duration_s": evaluation["duration_s"],
        "turns": turns,
        "turn_cap_hit": turn_cap_hit,
        "questions": questions_asked,
        "detector_questions": detector_questions,
        "follow_up_questions": session_snapshot["questions"],
        "stakeholder_answers": [
            {
                "category": item["category"],
                "answer": item["answer"],
            }
            for item in detector_questions
        ],
        "solution_written": session_snapshot["solution_written"],
        "final_solution": final_solution,
        "evaluation": evaluation,
        "self_check": self_check,
        "llm_calls": calls,
        "trace_events": trace_events,
        "messages": messages,
        "tools": session_snapshot,
        "totals": totals,
        "api_error": api_error,
        "started_at": started_at,
        "duration_s": round(time.monotonic() - started, 3),
        "status": "complete",
    }


def _direct_one(task, condition: Condition, llm, *, seed: int, include_visible: bool) -> dict:
    messages = (
        frontier_messages(task)
        if condition.b3
        else direct_messages(task, include_visible=include_visible)
    )
    temperature = (
        getattr(llm, "temperature", GENERATOR_TEMPERATURE)
        if condition.b3
        else GENERATOR_TEMPERATURE
    )
    try:
        result = llm.chat(
            messages,
            purpose="b3_frontier" if condition.b3 else "b2_sample",
            meta={"condition": condition.name, "task_key": task.key, "seed": seed},
            temperature=temperature,
            top_p=GENERATOR_TOP_P,
            max_tokens=4096,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "seed": seed,
            "pass": False,
            "bucket": "APIError",
            "error": f"{type(exc).__name__}: {exc}",
            "code": "",
            "totals": _totals(),
            "calls": [],
            "evaluation": {},
            "content": "",
            "messages": messages,
        }
    code = extract_code_block(result.content)
    evaluation = evaluate_solution(task, code)
    totals = _totals()
    _accumulate(totals, result)
    return {
        "seed": seed,
        "pass": bool(evaluation["pass"]),
        "bucket": evaluation["bucket"],
        "code": code,
        "evaluation": evaluation,
        "totals": totals,
        "calls": [_call_record(result, "b3_frontier" if condition.b3 else "b2_sample")],
        "content": result.content,
        "messages": messages,
        "error": "",
    }


def run_direct_unit(
    task,
    condition: Condition,
    llm,
    *,
    seed: int = GENERATOR_SEED,
    run_id: str = "",
    sample_index: int = 0,
) -> dict:
    started = time.monotonic()
    samples = []
    count = 1 if condition.b3 else max(condition.b2_samples, 1)
    for index in range(count):
        samples.append(
            _direct_one(
                task,
                condition,
                llm,
                seed=seed + index,
                include_visible=not condition.b3,
            )
        )
    totals = _totals()
    calls: list[dict] = []
    for sample in samples:
        for key in totals:
            totals[key] += int(sample["totals"].get(key, 0) or 0)
        calls.extend(sample["calls"])
    passed = [sample["pass"] for sample in samples]
    first = samples[0]
    return {
        "condition": condition.name,
        "task_key": task.key,
        "source": task.source,
        "source_task_id": task.source_task_id,
        "entry_point": task.entry_point,
        "sample_index": sample_index,
        "seed": seed,
        "run_id": run_id,
        "pass": any(passed),
        "pass_at_1": bool(first["pass"]),
        "bucket": "None" if any(passed) else first["bucket"],
        "hidden_output_tail": first.get("evaluation", {}).get("output_tail", ""),
        "turns": len(samples),
        "turn_cap_hit": False,
        "questions": 0,
        "detector_questions": [],
        "follow_up_questions": [],
        "solution_written": bool(first["code"]),
        "final_solution": first["code"],
        "evaluation": first.get("evaluation", {}),
        "self_check": None,
        "llm_calls": calls,
        "trace_events": [],
        "messages": [],
        "tools": {},
        "samples": [
            {
                "seed": sample["seed"],
                "pass": sample["pass"],
                "bucket": sample["bucket"],
                "code": sample.get("code", ""),
                "error": sample.get("error", ""),
                "totals": sample["totals"],
                "evaluation": sample.get("evaluation", {}),
                "messages": sample.get("messages", []),
            }
            for sample in samples
        ],
        "totals": totals,
        "api_error": first.get("error", ""),
        "started_at": utc_now_iso(),
        "duration_s": round(time.monotonic() - started, 3),
        "status": "complete",
    }


def run_unit(
    task,
    condition: Condition,
    llm,
    *,
    detector=None,
    siblings: list[dict] | None = None,
    seed: int = GENERATOR_SEED,
    run_id: str = "",
    sample_index: int = 0,
) -> dict:
    if condition.uses_loop:
        return run_repl_unit(
            task,
            condition,
            llm,
            detector=detector,
            siblings=siblings,
            seed=seed,
            run_id=run_id,
            sample_index=sample_index,
        )
    return run_direct_unit(
        task,
        condition,
        llm,
        seed=seed,
        run_id=run_id,
        sample_index=sample_index,
    )


__all__ = [
    "evaluate_solution",
    "run_direct_unit",
    "run_repl_unit",
    "run_unit",
]
