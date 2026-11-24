#!/usr/bin/env python3
"""Local fake-endpoint dry run of the Colab harness path.

This script is the local counterpart of ``notebooks/colab_a100_vllm.ipynb``.  It runs the
same pipeline against an in-process OpenAI-compatible fake endpoint instead of vLLM, using
the real committed 100-task pilot split, the real derived hidden tests and the real
Qwen3-Embedding encoder (CPU; download/cache only, no GPU):

1. endpoint probe (native tool calls and JSON text fallback);
2. L3 smoothing of the 500 detector-train and 100 pilot wishes;
3. detector refit on the smoothed train wishes;
4. a 5-task x 9-condition dry run over two config-stable shards;
5. the aggregate pilot report (pass@1, recovery, per-category recovery, failure buckets).

No GPU, paid API, vLLM server or Colab runtime is used, and nothing is written outside
``--output-dir`` (plus the gitignored encoder embedding cache).  The vLLM/Colab path itself
still has to be verified by the operator; see ``COLAB.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import override

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wishrec.detector.train import train_detector  # noqa: E402
from wishrec.harness.conditions import resolve_conditions  # noqa: E402
from wishrec.harness.config import EndpointConfig  # noqa: E402
from wishrec.harness.endpoint_check import check_endpoint  # noqa: E402
from wishrec.harness.llm import LLMClient  # noqa: E402
from wishrec.harness.models import Task, load_tasks  # noqa: E402
from wishrec.harness.report import build_report, render_report  # noqa: E402
from wishrec.harness.runner import run_units  # noqa: E402
from wishrec.harness.smoothing import smooth_tasks  # noqa: E402

SOLUTION = "def solution(*args, **kwargs):\n    return None\n"


def _usage(prompt: int = 12, completion: int = 6) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _completion(*, content: str = "", tool_calls: list[tuple[str, dict]] | None = None) -> dict:
    tool_calls = tool_calls or []
    return {
        "id": "chatcmpl-local-dry-run",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "fake",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                        for index, (name, arguments) in enumerate(tool_calls)
                    ],
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": _usage(),
    }


def _first_tool(payload: dict) -> tuple[str, dict] | None:
    for schema in payload.get("tools") or []:
        function = schema.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        if name == "write_file":
            return name, {"path": "solution.py", "content": SOLUTION}
        if name == "ask_stakeholder":
            return name, {"question": "What should the function do?"}
        if name == "echo":
            return name, {"value": "ok"}
    return None


def _fake_response(payload: dict) -> dict:
    messages = payload.get("messages") or []
    system = str(messages[0].get("content", "")) if messages else ""
    if "Rewrite the following wish" in system:
        wish = str(messages[-1].get("content", "")).strip()
        return _completion(content=f"A user asks: {wish}")
    if "You are the stakeholder" in system:
        return _completion(content="The function should be deterministic.")
    if "reviewing an implementation" in system:
        return _completion(content="OK")
    if "## Tool protocol (text fallback)" in system:
        if any(message.get("role") == "tool" for message in messages):
            return _completion(content=json.dumps({"tool": "finish"}))
        if "echo" in system:
            return _completion(content=json.dumps({"tool": "echo", "arguments": {"value": "ok"}}))
        return _completion(
            content=json.dumps(
                {
                    "tool": "write_file",
                    "arguments": {
                        "path": "solution.py",
                        "content": SOLUTION,
                    },
                }
            )
        )
    if payload.get("tools"):
        if any(message.get("role") == "tool" for message in messages):
            return _completion(content="Done.")
        call = _first_tool(payload)
        if call is not None:
            return _completion(tool_calls=[call])
        return _completion(content="Done.")
    return _completion(content=f"```python\n{SOLUTION}```")


class FakeEndpoint:
    """In-process OpenAI-compatible HTTP endpoint that records every request.

    Use it as a context manager: entering starts a threaded server on an ephemeral
    localhost port, and exiting shuts it down.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.rstrip("/").endswith("/models"):
                    self._send(
                        {
                            "object": "list",
                            "data": [{"id": "fake", "object": "model", "created": 1}],
                        }
                    )
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                endpoint.requests.append(payload)
                self._send(_fake_response(payload))

            @override
            def log_message(self, fmt: str, *args: object) -> None:
                """Silence the per-request access log."""

        self._handler = Handler
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> FakeEndpoint:
        server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server = server
        self._thread = thread
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        """Return the ``/v1`` base URL of the running server."""
        if self._server is None:
            raise RuntimeError("FakeEndpoint is not running; use it as a context manager")
        return f"http://127.0.0.1:{self._server.server_port}/v1"


def _pilot_tasks(limit: int) -> list[Task]:
    tasks = load_tasks(split="pilot", require_derived=True)
    # Prefer the two stdlib-only sources so the dry run does not need the pinned
    # BigCodeBench validation library set; fall back to any pilot task.
    simple = [task for task in tasks if task.source in ("humanevalplus", "mbppplus")]
    chosen = (simple + [task for task in tasks if task not in simple])[:limit]
    if len(chosen) < limit:
        raise SystemExit(f"pilot split has only {len(chosen)} tasks")
    return chosen


def main(argv: list[str] | None = None) -> int:
    """Run the five dry-run stages against the fake endpoint and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".local-dry-run")
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoothing-workers", type=int, default=4)
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="reuse the committed R1 detector instead of the real CPU refit",
    )
    parser.add_argument(
        "--tool-mode",
        choices=("auto", "native", "text"),
        default="text",
        help="harness transport for the dry run (default: the JSON text fallback)",
    )
    args = parser.parse_args(argv)
    started = time.time()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    with FakeEndpoint() as endpoint:
        os.environ["VLLM_BASE_URL"] = endpoint.base_url
        os.environ["VLLM_API_KEY"] = "EMPTY"
        os.environ["VLLM_MODEL"] = "fake"
        os.environ["VLLM_TIMEOUT"] = "30"
        os.environ["VLLM_MAX_RETRIES"] = "1"
        os.environ["VLLM_TOOL_MODE"] = args.tool_mode
        config = EndpointConfig(
            base_url=endpoint.base_url,
            model="fake",
            timeout=30,
            max_retries=1,
        )

        print("== 1. endpoint probe (native + text fallback)")
        probe = check_endpoint(config)
        (out / "endpoint.json").write_text(
            json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not probe["ok"] or not probe["native"]["ok"] or not probe["text"]["ok"]:
            raise SystemExit("fake endpoint probe failed; it must support both transports")
        print(
            f"native tool calls: {probe['native']['ok']}, "
            f"text fallback: {probe['text']['ok']}, "
            f"selected: {probe['selected_transport']}"
        )
        print(f"dry-run harness transport: {args.tool_mode}")

        print("== 2. L3 smoothing (500 train + 100 pilot)")
        smoothing_tasks = load_tasks(split="train", require_derived=False) + load_tasks(
            split="pilot", require_derived=False
        )
        artifact = smooth_tasks(
            smoothing_tasks,
            LLMClient(config, temperature=0.0),
            out / "smoothing",
            workers=args.smoothing_workers,
        )
        print(
            f"smoothing records: {len(artifact.records)} "
            f"(splits {artifact.splits}), fallbacks {artifact.fallbacks}"
        )

        print("== 3. detector refit on smoothed wishes (real encoder, CPU)")
        if args.skip_detector:
            detector_path = REPO_ROOT / "results" / "detector" / "model.json"
            print(f"using committed detector: {detector_path}")
        else:
            refit = train_detector(
                smoothing_path=out / "smoothing" / "wishes.jsonl",
                output_dir=out / "detector-smoothed",
                run_id="r2-local-dry-run",
            )
            detector_path = Path(refit["output_dir"]) / "model.json"
            metrics = refit["metrics"]["detection"]["encoder"]
            print(
                "refit macro-F1 "
                f"{metrics['macro_f1']:.3f}, constraints AUROC "
                f"{metrics['per_label']['constraints']['auroc']:.3f}"
            )

        tasks = _pilot_tasks(args.tasks)
        print(
            f"== 4. {len(tasks)}-task x 9-condition dry run "
            f"({args.shards} shards): {[task.key for task in tasks]}"
        )
        run_dir = out / "run"
        for shard in range(args.shards):
            print(f"-- shard {shard + 1}/{args.shards}")
            result = run_units(
                tasks=tasks,
                conditions=resolve_conditions(None),
                run_id="r2-local-dry-run",
                output_dir=run_dir,
                samples=1,
                workers=args.workers,
                smoothing_path=out / "smoothing" / "wishes.jsonl",
                detector_path=detector_path,
                shard=shard,
                shards=args.shards,
                progress=lambda _message: None,
            )
            print(
                f"   units {result['summary']['units']}/"
                f"{result['summary']['planned_units']} "
                f"status={result['manifest']['status']}"
            )

        print("== 5. aggregate report")
        summary = build_report([run_dir])
        report_path = out / "run" / "report.md"
        (out / "run" / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report_path.write_text(render_report(summary), encoding="utf-8")
        print(render_report(summary))
    elapsed = time.time() - started
    print(f"local fake-endpoint dry run finished in {elapsed:.1f}s")
    print(f"artifacts: {out}")
    if not summary["complete"] or summary["heldout_tasks_used"]:
        print("ERROR: dry run is incomplete or touched held-out tasks")
        return 1
    print("OK: all planned units completed with the fake endpoint; no GPU was used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
