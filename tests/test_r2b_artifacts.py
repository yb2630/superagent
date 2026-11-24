import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "runs" / "r2b-pilot"
LEVELS = ROOT / "results" / "levels"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows():
    return [
        json.loads(line)
        for line in (RUN / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tasks():
    return [
        json.loads(line)
        for line in (LEVELS / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_r2b_artifacts_are_committed():
    for name in (
        "summary.json",
        "report.md",
        "results.jsonl",
        "manifest.json",
        "environment.json",
        "endpoint.json",
        "gpu_time.json",
        "data_provenance.json",
        "r1_detector.json",
        "bootstrap.json",
        "bootstrap.md",
        "COLAB_RUN.md",
        "colab_a100_vllm.ipynb",
        "README.md",
    ):
        assert (RUN / name).is_file(), f"missing {name}"
    assert len(list((RUN / "units").rglob("*.json"))) == 2700
    assert (RUN / "detector-smoothed" / "metrics.json").is_file()
    assert (RUN / "detector-smoothed" / "model.json").is_file()
    assert (RUN / "smoothing" / "wishes.jsonl").is_file()
    assert (RUN / "smoothing" / "manifest.json").is_file()
    assert (RUN / "dry-run" / "summary.json").is_file()
    assert len(list((RUN / "dry-run" / "units").rglob("*.json"))) == 45


def test_r2b_run_is_complete_and_holds_out_nothing():
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True
    assert summary["completed_units"] == summary["expected_units"] == 2700
    assert summary["heldout_tasks_used"] == 0
    assert summary["tasks"] == 100
    assert summary["samples"] == 3
    assert summary["missing_units"] == {}
    assert manifest["status"] == "complete"
    assert manifest["units_completed"] == manifest["units_planned"] == 2700
    assert manifest["run_id"] == "r2b-pilot"

    splits = {task["key"]: task["split"] for task in _tasks()}
    pilot = {key for key, split in splits.items() if split == "pilot"}
    heldout = {key for key, split in splits.items() if split == "heldout"}
    assert len(pilot) == 100
    assert len(heldout) == 300
    assert set(manifest["config"]["task_keys"]) == pilot
    task_keys = {row["task_key"] for row in _rows()}
    assert task_keys == pilot
    assert not (task_keys & heldout)


def test_r2b_summary_matches_the_unit_table():
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    rows = _rows()
    assert len(rows) == 2700
    assert len({(r["condition"], r["task_key"], r["sample_index"]) for r in rows}) == 2700

    for condition, item in summary["conditions_summary"].items():
        selected = [row for row in rows if row["condition"] == condition]
        assert item["units"] == len(selected) == 300
        assert item["pass_at_1"] == pytest.approx(
            sum(bool(r["pass_at_1"]) for r in selected) / len(selected), abs=5e-4
        )
        assert item["pass_rate"] == pytest.approx(
            sum(bool(r["pass"]) for r in selected) / len(selected), abs=5e-4
        )

    for item in summary["by_category"].values():
        assert item["tasks"] > 0
        for values in item["conditions"].values():
            # The category-to-task mapping is not in results.jsonl, so only the overall
            # condition aggregates are cross-checked here; the per-category rows are
            # checked against the committed summary by the bootstrap test below.
            assert values["units"] == item["tasks"] * 3
            assert 0.0 <= values["pass_at_1"] <= 1.0
            assert 0.0 <= values["pass_rate"] <= 1.0


def test_r2b_units_reproduce_the_result_table():
    rows = {(row["condition"], row["sample_index"], row["task_key"]): row for row in _rows()}
    units = {}
    for path in (RUN / "units").rglob("*.json"):
        unit = json.loads(path.read_text(encoding="utf-8"))
        key = (unit["condition"], unit["sample_index"], unit["task_key"])
        assert key not in units, f"duplicate unit {key}"
        assert path.parts[-3] == unit["condition"]
        assert path.parts[-2] == str(unit["sample_index"])
        units[key] = unit
    assert set(units) == set(rows)
    for key, unit in units.items():
        assert bool(unit["pass"]) == bool(rows[key]["pass"])


def test_r2b_bootstrap_matches_units_and_summary():
    bootstrap = json.loads((RUN / "bootstrap.json").read_text(encoding="utf-8"))
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    assert bootstrap["results_sha256"] == _sha256(RUN / "results.jsonl")
    assert bootstrap["summary_sha256"] == _sha256(RUN / "summary.json")
    assert bootstrap["tasks"] == 100
    assert bootstrap["bootstrap"]["resamples"] == 2000
    assert bootstrap["bootstrap"]["seed"] == 20261121

    for contrast in bootstrap["contrasts"]:
        a = summary["conditions_summary"][contrast["a"]]["pass_at_1"]
        b = summary["conditions_summary"][contrast["b"]]["pass_at_1"]
        assert contrast["difference"] == pytest.approx(a - b, abs=1e-3)
        assert contrast["ci_low"] <= contrast["difference"] <= contrast["ci_high"]

    by_name = {item["name"]: item for item in bootstrap["contrasts"]}
    assert set(by_name) == {
        "B1-H0",
        "H2-H0",
        "H3-H0",
        "H4-H0",
        "H1-H0",
        "H3-H2",
        "H3-H4",
        "DISCARD-H3",
        "B2-H0",
    }
    assert by_name["B1-H0"]["excludes_zero"] is True
    assert by_name["H1-H0"]["excludes_zero"] is True
    assert by_name["H2-H0"]["excludes_zero"] is True
    assert by_name["H3-H2"]["excludes_zero"] is True
    assert by_name["DISCARD-H3"]["excludes_zero"] is True
    assert by_name["H3-H0"]["excludes_zero"] is False
    assert by_name["H3-H4"]["excludes_zero"] is False

    categories = bootstrap["by_category"]
    assert set(categories) == {"signature_types", "examples", "constraints", "error_behaviour"}
    for name, item in categories.items():
        reported = summary["by_category"][name]
        assert item["tasks"] == reported["tasks"]
        for condition in ("B1", "H0", "H2", "H3", "H4"):
            assert item["conditions"][condition]["pass_at_1"] == pytest.approx(
                reported["conditions"][condition]["pass_at_1"]
            )
        assert item["conditions"]["H3"]["recovery"] == pytest.approx(
            reported["conditions"]["H3"]["recovery"]
        )


def test_r2b_provenance_matches_the_corrected_levels():
    provenance = json.loads((RUN / "data_provenance.json").read_text(encoding="utf-8"))
    environment = json.loads((RUN / "environment.json").read_text(encoding="utf-8"))
    assert provenance["run_id"] == "r2b-pilot"
    assert provenance["levels_parsing"] == {"visible_tests": "inputs_only"}
    assert provenance["levels_tasks_sha256"] == _sha256(LEVELS / "tasks.jsonl")
    # The manifest file hash and the canonical manifest hash below are provenance from the
    # measured run; a clean `make levels` rebuild changes created_at/git/timing fields, so
    # the task hash is the stable reproducibility invariant.
    assert len(provenance["levels_manifest_sha256"]) == 64
    assert environment["visible_tests"] == "inputs_only"
    assert environment["transport"] == "text"
    assert environment["gpu"] == "NVIDIA A100-SXM4-40GB"

    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["level_manifest_sha256"]) == 64
