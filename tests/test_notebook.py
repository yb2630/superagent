from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

NOTEBOOK = Path("notebooks/colab_a100_vllm.ipynb")
RUNBOOK = Path("COLAB.md")
LEVELS = Path("results/levels")


def _sources() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def test_notebook_is_valid_json_with_compiling_code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 20
    code_cells = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) >= 10
    for index, source in enumerate(code_cells):
        compile(source, f"notebook-code-{index}", "exec")


def test_notebook_covers_the_corrected_r2b_pipeline_steps():
    sources = _sources()
    text = "\n".join(sources)
    for marker in (
        "ENABLE_NATIVE_TOOL_CALLS",
        "PERSIST_TO_DRIVE",
        "wishrec.materialize",
        "vllm serve",
        "Qwen/Qwen3-8B-AWQ",
        "Qwen/Qwen3-Embedding-0.6B",
        "wishrec.harness.endpoint_check",
        "wishrec.harness.smoothing",
        "wishrec.detector.train",
        "wishrec.harness.runner",
        "wishrec.harness.report",
        "--shards",
        "--smoothed",
        "files.download",
        "r2b-artifacts",
        "EXPECTED_LEVELS_SHA256",
        "ALLOW_LEVELS_DRIFT",
        "levels tasks.jsonl sha256",
        "derived hidden/visible",
        "INSTALL_VALIDATION_REQUIREMENTS",
        "requirements-validation.txt",
        "MPLBACKEND",
        "Agg",
        "PYTHONPATH",
        "dry-run passes",
        "ModuleNotFoundError units",
        "CLEAR_CACHE_FOR_DRY_RUN",
        "dry run finished in",
        "data_provenance.json",
        "r2b-pilot",
        "VISIBLE_TESTS",
        "inputs_only",
        "--visible-tests",
        "BigCodeBench visible tests",
        "run_visible_tests",
        "Printed output",
    ):
        assert marker in text, marker
    assert "--split heldout" not in text


def test_notebook_never_embeds_a_token_literal():
    text = "\n".join(_sources())
    patterns = (
        r"ghp_[A-Za-z0-9]{20,}",
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"hf_[A-Za-z0-9]{20,}",
        r"sk-[A-Za-z0-9]{20,}",
    )
    for pattern in patterns:
        assert not re.search(pattern, text), pattern
    # The clone must not put a token in a URL.
    assert "x-access-token:$" not in text
    assert "x-access-token@" not in text


def test_runbook_covers_resume_troubleshooting_and_operator_checks():
    text = RUNBOOK.read_text(encoding="utf-8")
    for marker in (
        "What is verified locally vs by the operator",
        "## 1. Before you start",
        "## 2. Run order and expected wall-clock",
        "## 3. Step details",
        "## 4. Resume after a disconnect or session end",
        "## 5. Alternative: run vLLM in Colab, harness locally",
        "## 6. Troubleshooting",
        "## 7. Operator checklist",
        "GPU-hours",
        "cloudflared",
        "ngrok",
        "held-out",
        "make levels",
        "vLLM",
        "token",
        "r2b-pilot",
        "inputs_only",
        "BigCodeBench",
        "visible tests",
    ):
        assert marker in text, marker
    assert "2.902 GPU-hours" in text


def test_notebook_pins_the_committed_corrected_levels():
    manifest = json.loads((LEVELS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parsing"] == {"visible_tests": "inputs_only"}
    sha = hashlib.sha256((LEVELS / "tasks.jsonl").read_bytes()).hexdigest()
    text = "\n".join(_sources())
    assert f'EXPECTED_LEVELS_SHA256 = "{sha}"' in text
    assert 'VISIBLE_TESTS = "inputs_only"' in text


def test_repo_docs_point_at_the_r2_notebook_and_runbook():
    plan = Path("PLAN.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    for text in (plan, readme):
        assert "notebooks/colab_a100_vllm.ipynb" in text
        assert "COLAB.md" in text
    assert "local_fake_dry_run" in readme
    assert ".local-dry-run" in Path(".gitignore").read_text(encoding="utf-8")
