from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_submitted_milestone_is_byte_identical():
    submitted = ROOT / "paper" / "milestone" / "submitted"
    assert _sha256(submitted / "main.tex") == (
        "760bf3807ed5b21d25542a2a4a68b890c7a3051811f433a5c3569d8c187bec2b"
    )
    assert _sha256(submitted / "main.pdf") == (
        "bb151114e8ec16e19af0b94be07fbc64fc53b1638ea364ffcf5c4798aa2ed668"
    )


def test_final_milestone_and_extended_report_exist():
    for relative in (
        "paper/milestone/main.tex",
        "paper/milestone/main.pdf",
        "paper/main.tex",
        "paper/main.pdf",
        "paper/neurips_2019.sty",
        "paper/references.bib",
    ):
        assert (ROOT / relative).is_file(), relative
    milestone = (ROOT / "paper" / "milestone" / "main.tex").read_text(encoding="utf-8")
    assert "Noah Foster" in milestone
    assert "H3$^-$" in milestone
    # The accepted layout puts title/abstract on page 1, the body on pages 2--3,
    # and the free table (all nine conditions, with Turns) at the end.
    assert "\\clearpage" in milestone
    assert "Turns" in milestone
    for condition in ("B1", "H0", "H1", "H2", "H3", "H4", "B2", "B3", "H3$^-$"):
        assert condition in milestone
    assert "0.623" in milestone
    assert "0.537" in milestone
    assert "0.427" in milestone
    assert "53.7" in milestone
    assert "42.7" in milestone


def test_final_detector_artifact_matches_the_reported_metrics():
    artifact = ROOT / "results" / "detector-smoothed"
    for name in (
        "model.json",
        "metrics.json",
        "manifest.json",
        "cv_predictions.jsonl",
        "report.md",
    ):
        assert (artifact / name).is_file(), name
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    encoder = metrics["detection"]["encoder"]
    assert round(encoder["macro_f1"], 3) == 0.792
    assert round(encoder["per_label"]["constraints"]["auroc"], 3) == 0.952
    assert round(encoder["per_label"]["error_behaviour"]["auroc"], 3) == 0.744
    assert encoder["per_label"]["error_behaviour"]["positives"] == 85


def test_superseded_state_is_gone():
    assert not (ROOT / "results" / "runs" / "r2-pilot").exists()
    assert not (ROOT / "paper" / "milestone" / "r2").exists()
    assert not (ROOT / "results" / "levels_pre_r2b_split.json").exists()


def test_documents_do_not_carry_the_superseded_state():
    stale = (
        "58.3",
        "47.7",
        "H3-NA",
        "0.777",
        "24 GB",
        "18 GPU-hours",
        "results/runs/r2-pilot",
        "paper/milestone/r2",
        "DISCARD",
    )
    documents = (
        "README.md",
        "PLAN.md",
        "PROJECT.md",
        "COLAB.md",
        "paper/main.tex",
        "paper/operationalization-notes.md",
    )
    for relative in documents:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for marker in stale:
            assert marker not in text, f"{relative} still contains {marker!r}"
