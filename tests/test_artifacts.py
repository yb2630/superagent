import hashlib
import json
from pathlib import Path

from wishrec.artifacts import check_records, load_jsonl
from wishrec.config import FIELD_KEYS, SOURCES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "levels"


def _load():
    tasks = load_jsonl(RESULTS / "tasks.jsonl")
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((RESULTS / "stats.json").read_text(encoding="utf-8"))
    return tasks, manifest, stats


def test_committed_artifacts_exist():
    for name in ("tasks.jsonl", "manifest.json", "stats.json"):
        assert (RESULTS / name).is_file(), f"missing committed artifact {name}"


def test_task_records_are_internally_consistent():
    tasks, manifest, _ = _load()
    split_counts = check_records(tasks)
    assert len(tasks) == manifest["counts"]["kept"]
    assert split_counts == {
        "train": 500,
        "evolution": 200,
        "heldout": 300,
        "pilot": 100,
    }
    assert manifest["split"]["counts"] == split_counts
    assert manifest["counts"]["shortfall"] == 0


def test_split_is_disjoint_and_pilot_is_disjoint():
    tasks, _, stats = _load()
    splits = {}
    for task in tasks:
        splits.setdefault(task["split"], set()).add(task["key"])
    assert set(splits) == {"train", "evolution", "heldout", "pilot"}
    for name, keys in splits.items():
        assert len(keys) == stats["splits"]["counts"][name]
    ordered = list(splits)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            assert splits[first].isdisjoint(splits[second])
    assert splits["pilot"].isdisjoint(splits["heldout"])
    assert splits["heldout"].isdisjoint(splits["train"] | splits["evolution"])


def test_source_pins_match_configuration():
    tasks, manifest, _ = _load()
    seen = {task["source"] for task in tasks}
    assert seen == set(SOURCES)
    for record in manifest["sources"]:
        source = SOURCES[record["name"]]
        assert record["repo"] == source.repo
        assert record["revision"] == source.revision
        assert record["sha256"] == source.sha256
        assert record["bytes"] > 0
    assert manifest["environment"]["python"].startswith("3.12")


def test_visible_and_hidden_hash_fields():
    tasks, _, _ = _load()
    for task in tasks:
        for field in (
            "stub_sha256",
            "visible_test_sha256",
            "hidden_test_sha256",
            "canonical_sha256",
        ):
            assert len(task[field]) == 64
            assert all(character in "0123456789abcdef" for character in task[field])
        assert task["visible_asserts_count"] == 3
        assert task["hidden_test_bytes"] > 0
        assert task["hidden_test_shared_across_levels"] is True
        assert task["visible_asserts"].strip()
        assert task["validated"] is True


def test_levels_and_l2_c_are_consistent():
    tasks, _, _ = _load()
    labels = {
        "intent": "Intent:",
        "signature_types": "Signature and types:",
        "examples": "Examples:",
        "constraints": "Constraints and edge cases:",
        "error_behaviour": "Error behaviour:",
    }
    for task in tasks:
        assert task["levels"]["L3"] == task["fields"]["intent"].strip()
        assert task["deleted_fields"]["L0"] == []
        assert "examples" in task["deleted_fields"]["L1"]
        assert set(task["L2_c"]) == set(FIELD_KEYS)
        for category, drop in task["L2_c"].items():
            if drop["field_present"]:
                assert labels[category] not in drop["text"], (task["key"], category)
            else:
                assert drop["text"] == task["levels"]["L0"]


def test_field_presence_matches_stats():
    tasks, _, stats = _load()
    counts = dict.fromkeys(FIELD_KEYS, 0)
    for task in tasks:
        assert set(task["field_presence"]) == set(FIELD_KEYS)
        for field in FIELD_KEYS:
            assert task["field_presence"][field] == bool(task["fields"][field].strip())
            counts[field] += int(task["field_presence"][field])
    assert stats["field_presence"]["kept"]["counts"] == counts
    assert stats["field_presence"]["kept"]["denominator"] == len(tasks)


def test_bigcodebench_raises_expectation():
    _, _, stats = _load()
    raises = stats["bigcodebench_raises"]
    assert raises["rows_with_raises"] == 296
    assert raises["expected_rows_with_raises"] == 296
    assert raises["matched_expected"] is True
    assert raises["valid_tasks_with_error_behaviour"] > 0


def test_checksums_and_counts_in_manifest():
    tasks, manifest, _ = _load()
    raw = (RESULTS / "tasks.jsonl").read_text(encoding="utf-8")
    assert hashlib.sha256(raw.encode("utf-8")).hexdigest() == manifest["artifacts"]["tasks_sha256"]
    stats_raw = (RESULTS / "stats.json").read_text(encoding="utf-8")
    assert (
        hashlib.sha256(stats_raw.encode("utf-8")).hexdigest()
        == manifest["artifacts"]["stats_sha256"]
    )
    assert manifest["artifacts"]["task_count"] == len(tasks) == 1100
    assert manifest["counts"]["valid"] >= manifest["counts"]["kept"]
    kept_keys = {task["key"] for task in tasks}
    assert all(
        key not in kept_keys for keys in manifest["unused_task_keys"].values() for key in keys
    )


def test_error_behaviour_prevalence_is_reported_per_source():
    _, manifest, stats = _load()
    for block in stats["field_presence"]["kept"]["by_source"].values():
        assert block["denominator"] > 0
        assert block["counts"]["intent"] == block["denominator"]
    assert "by_source" in stats["library_requirements"]
    assert stats["exclusions"]["total"] == len(manifest["exclusions"])


def test_corrected_protocol_uses_inputs_only_visible_tests():
    tasks, manifest, _ = _load()
    assert manifest["parser_version"] == "0.2.0"
    assert manifest["parsing"] == {"visible_tests": "inputs_only"}
    for task in tasks:
        assert task["meta"]["visible_tests"] == "inputs_only"
        if task["source"] == "bigcodebench":
            # Documented residual: BigCodeBench's visible suite is assertion-based and
            # is kept as-is in both modes, so its examples stay visible to the agent.
            continue
        assert "print(" in task["visible_asserts"]
        assert "assert" not in task["visible_asserts"]
    assert any(task["source"] == "bigcodebench" for task in tasks)


def test_exclusion_details_are_portable():
    _, manifest, _ = _load()
    for item in manifest["exclusions"]:
        detail = item.get("detail", "")
        assert "/Users/" not in detail, item["key"]
        assert "/private/var/folders/" not in detail, item["key"]
        assert "wishrec-validate-" not in detail, item["key"]


def test_portable_detail_rewrites_local_paths():
    from pathlib import Path

    from wishrec.build import portable_detail

    home = str(Path.home())
    detail = (
        '  File "/private/var/folders/9n/abc/T/wishrec-validate-czco4ya9/visible_test.py", line 16\n'
        '  File "/Users/alice/work/.venv/lib/python3.12/site-packages/pandas/core/x.py", line 3\n'
        '  File "/Users/alice/.local/share/uv/python/cpython-3.12.15-none/lib/python3.12/re.py", line 1\n'
        f"data at {home}/nltk_data\n"
        "PosixPath('/private/var/folders/9n/phmsx2[90 chars]act')\n"
    )
    portable = portable_detail(detail)
    assert "/Users/" not in portable
    assert "/private/var/folders/" not in portable
    assert "<workspace>/visible_test.py" in portable
    assert "<venv>/lib/python3.12/site-packages/pandas/core/x.py" in portable
    assert "<python>/lib/python3.12/re.py" in portable
    assert "~/nltk_data" in portable
    assert portable_detail("") == ""
