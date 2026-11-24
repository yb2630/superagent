from types import SimpleNamespace

from wishrec.split import apportion, assign_splits, select_kept, split_targets


def make_tasks(counts: dict[str, int]) -> list[SimpleNamespace]:
    tasks = []
    for source, count in counts.items():
        for index in range(count):
            tasks.append(
                SimpleNamespace(
                    source=source,
                    source_task_id=f"{index:04d}",
                    key=f"{source}/{index:04d}",
                )
            )
    return tasks


def test_apportion_is_exact_and_deterministic():
    first = apportion(10, {"a": 1, "b": 1, "c": 1})
    second = apportion(10, {"a": 1, "b": 1, "c": 1})
    assert first == second
    assert sum(first.values()) == 10
    assert first == {"a": 4, "b": 3, "c": 3}


def test_apportion_respects_caps():
    allocation = apportion(5, {"a": 10, "b": 1}, caps={"a": 0, "b": 10})
    assert allocation == {"a": 0, "b": 5}


def test_split_targets_sum_to_total():
    for total in (0, 1, 7, 100, 1000, 1100):
        targets = split_targets(total)
        assert sum(targets.values()) == total
        assert set(targets) == {"train", "evolution", "heldout", "pilot"}
    assert split_targets(1100) == {
        "train": 500,
        "evolution": 200,
        "heldout": 300,
        "pilot": 100,
    }


def test_select_kept_uses_all_small_sources_and_samples_bigcodebench():
    tasks = make_tasks({"humanevalplus": 20, "mbppplus": 40, "bigcodebench": 100})
    kept, info = select_kept(tasks, target=120, seed=20261121)
    assert len(kept) == 120
    assert info["shortfall"] == 0
    assert info["kept_counts"] == {
        "humanevalplus": 20,
        "mbppplus": 40,
        "bigcodebench": 60,
    }
    assert info["unused_counts"]["bigcodebench"] == 40
    again, _ = select_kept(tasks, target=120, seed=20261121)
    assert [task.key for task in kept] == [task.key for task in again]


def test_select_kept_reports_shortfall_without_inventing_tasks():
    tasks = make_tasks({"humanevalplus": 20, "mbppplus": 40, "bigcodebench": 100})
    kept, info = select_kept(tasks, target=200, seed=20261121)
    assert len(kept) == 160
    assert info["shortfall"] == 40
    assert info["kept_counts"] == {
        "humanevalplus": 20,
        "mbppplus": 40,
        "bigcodebench": 100,
    }


def test_assign_splits_is_exact_stratified_and_disjoint():
    tasks = make_tasks({"humanevalplus": 110, "mbppplus": 330, "bigcodebench": 660})
    assigned, info = assign_splits(tasks, seed=20261121)
    assert {split: len(items) for split, items in assigned.items()} == {
        "train": 500,
        "evolution": 200,
        "heldout": 300,
        "pilot": 100,
    }
    keys = [task.key for items in assigned.values() for task in items]
    assert len(keys) == len(set(keys)) == 1100
    for source in ("humanevalplus", "mbppplus", "bigcodebench"):
        total = sum(1 for items in assigned.values() for task in items if task.source == source)
        assert total == info["split_by_source"]["train"].get(source, 0) + info["split_by_source"][
            "evolution"
        ].get(source, 0) + info["split_by_source"]["heldout"].get(source, 0) + info[
            "split_by_source"
        ]["pilot"].get(source, 0)
    assert info["split_by_source"]["train"] == {
        "humanevalplus": 50,
        "mbppplus": 150,
        "bigcodebench": 300,
    }
    again, _ = assign_splits(tasks, seed=20261121)
    for split in assigned:
        assert {task.key for task in assigned[split]} == {task.key for task in again[split]}


def test_assign_splits_pilot_is_disjoint_from_heldout():
    tasks = make_tasks({"humanevalplus": 10, "mbppplus": 10, "bigcodebench": 80})
    assigned, _ = assign_splits(tasks, seed=1)
    pilot = {task.key for task in assigned["pilot"]}
    heldout = {task.key for task in assigned["heldout"]}
    assert pilot.isdisjoint(heldout)
