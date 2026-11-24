"""Summarise the committed R0 artifacts for review/PR text."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "levels"


def main() -> None:
    tasks = [
        json.loads(line)
        for line in (RESULTS / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    stats = json.loads((RESULTS / "stats.json").read_text())
    splits = ("train", "evolution", "heldout", "pilot")
    sources = ("humanevalplus", "mbppplus", "bigcodebench")
    print("counts:", json.dumps(manifest["counts"], indent=2))
    print("\nsplit x source")
    for split in splits:
        row = stats["splits"]["split_by_source"][split]
        print(split, stats["splits"]["counts"][split], {s: row.get(s, 0) for s in sources})
    print("\nsource totals:")
    for source in sources:
        print(source, sum(1 for task in tasks if task["source"] == source))
    print("\nfield presence kept:")
    for field in ("intent", "signature_types", "examples", "constraints", "error_behaviour"):
        block = stats["field_presence"]["kept"]
        print(field, f"{block['counts'][field]}/{block['denominator']}", block["rates"][field])
        for source in sources:
            item = block["by_source"][source]
            print("   ", source, f"{item['counts'][field]}/{item['denominator']}")
    print("\nfield presence pool:")
    for field in ("intent", "signature_types", "examples", "constraints", "error_behaviour"):
        block = stats["field_presence"]["pool"]
        print(field, f"{block['counts'][field]}/{block['denominator']}", block["rates"][field])
        for source in sources:
            item = block["by_source"][source]
            print("   ", source, f"{item['counts'][field]}/{item['denominator']}")
    print("\nbigcodebench raises:", json.dumps(stats["bigcodebench_raises"], indent=2))
    print("\nexclusions:")
    print(json.dumps(stats["exclusions"], indent=2))
    print("\nexclusion list (first 40):")
    for item in manifest["exclusions"][:40]:
        print(item["key"], item["stage"], item.get("exception"), item["reason"][:180])
    print("\nlibrary requirements (top 40):")
    libs = stats["library_requirements"]["distributions"]
    for name, count in sorted(libs.items(), key=lambda item: -item[1])[:40]:
        print(name, count)
    print("\nmissing libraries:")
    print(json.dumps(stats["library_requirements"]["missing_libraries"], indent=2))
    print("\nlevel chars:")
    print(json.dumps(stats["level_chars"]["overall"], indent=2))
    print("\nvalidation:", json.dumps(manifest["validation"], indent=2))
    print("\nenvironment:", json.dumps(manifest["environment"], indent=2)[:2000])
    print("\nartifact sizes:")
    for name in ("tasks.jsonl", "manifest.json", "stats.json"):
        print(name, (RESULTS / name).stat().st_size)
    print("\ntop-level record keys:", sorted(tasks[0]))
    print(
        "excluded keys:",
        len(manifest["exclusions"]),
        "unused:",
        {k: len(v) for k, v in manifest["unused_task_keys"].items()},
    )
    print("source kept counts:", {s: sum(1 for t in tasks if t["source"] == s) for s in sources})


if __name__ == "__main__":
    main()
