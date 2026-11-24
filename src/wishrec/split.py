"""Deterministic task selection and the source-stratified 500/200/300 + 100 split."""

from __future__ import annotations

import random
import re
from collections import defaultdict

from .config import (
    SOURCE_ORDER,
    SPLIT_ORDER,
    SPLIT_SEED,
    SPLIT_SIZES,
    TARGET_TASKS,
)


def _natural_key(text: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def _sorted_pool(tasks: list) -> list:
    return sorted(tasks, key=lambda task: _natural_key(task.source_task_id))


def apportion(
    total: int, weights: dict[str, int], caps: dict[str, int] | None = None
) -> dict[str, int]:
    """Largest-remainder apportionment of ``total`` over weighted buckets.

    ``caps`` limits each bucket (used for the per-split source quotas).  Ties are broken
    by bucket name so the result is deterministic.
    """
    if total < 0:
        raise ValueError("total must be non-negative")
    names = list(weights)
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive number")
    cap_map = dict.fromkeys(names, total) if caps is None else dict(caps)
    if sum(cap_map.values()) < total:
        raise ValueError("caps cannot hold the requested total")
    exact = {name: total * weights[name] / weight_sum for name in names}
    allocation = {name: min(int(exact[name]), cap_map[name]) for name in names}
    remainder = total - sum(allocation.values())
    # Fractional parts after flooring; tied buckets are ordered by name.
    order = sorted(
        names,
        key=lambda name: (-(exact[name] - int(exact[name])), name),
    )
    while remainder > 0:
        progressed = False
        for name in order:
            if allocation[name] < cap_map[name]:
                allocation[name] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
        if not progressed:
            raise ValueError("could not apportion total within caps")
    return allocation


def select_kept(
    valid_tasks: list,
    *,
    target: int = TARGET_TASKS,
    seed: int = SPLIT_SEED,
) -> tuple[list, dict]:
    """Keep every valid HumanEval+/MBPP+ task and sample BigCodeBench up to ``target``.

    HumanEval+ and MBPP+ are small, hand-written benchmark families; using all of them
    avoids throwing away scarce real tasks, and BigCodeBench fills the remainder.  The
    BigCodeBench sample is a seeded uniform sample.  If the valid pool cannot reach the
    target, all valid tasks are kept and the exact shortfall is reported rather than
    inventing tasks.
    """
    by_source: dict[str, list] = defaultdict(list)
    for task in valid_tasks:
        by_source[task.source].append(task)
    for source in by_source:
        if source not in SOURCE_ORDER:
            raise ValueError(f"unknown source {source!r}")

    small = []
    for source in ("humanevalplus", "mbppplus"):
        small.extend(_sorted_pool(by_source.get(source, [])))
    bigcodebench = _sorted_pool(by_source.get("bigcodebench", []))

    if len(small) >= target:
        rng = random.Random(f"{seed}:kept:small")
        pool = list(small)
        rng.shuffle(pool)
        kept = pool[:target]
        shortfall = 0
    else:
        kept = list(small)
        need = target - len(kept)
        if len(bigcodebench) >= need:
            rng = random.Random(f"{seed}:kept:bigcodebench")
            pool = list(bigcodebench)
            rng.shuffle(pool)
            kept.extend(pool[:need])
            shortfall = 0
        else:
            kept.extend(bigcodebench)
            shortfall = target - len(kept)

    kept = sorted(kept, key=lambda task: task.key)
    pool_counts = {source: len(by_source.get(source, [])) for source in SOURCE_ORDER}
    kept_counts = {
        source: sum(1 for task in kept if task.source == source) for source in SOURCE_ORDER
    }
    unused_counts = {source: pool_counts[source] - kept_counts[source] for source in SOURCE_ORDER}
    info = {
        "selection_seed": seed,
        "target_tasks": target,
        "policy": (
            "retain all valid HumanEval+ and MBPP+ tasks; fill the target with a seeded "
            "uniform BigCodeBench sample; never invent tasks to cover a shortfall"
        ),
        "pool_counts": pool_counts,
        "kept_counts": kept_counts,
        "unused_counts": unused_counts,
        "unused_task_keys": {
            source: sorted(
                task.key
                for task in by_source.get(source, [])
                if task.key not in {kept_task.key for kept_task in kept}
            )
            for source in SOURCE_ORDER
        },
        "shortfall": shortfall,
    }
    return kept, info


def split_targets(total: int, sizes: dict[str, int] = SPLIT_SIZES) -> dict[str, int]:
    """Split sizes for ``total`` tasks; exact when total matches the milestone sizes."""
    weights = {name: sizes[name] for name in SPLIT_ORDER}
    return apportion(total, weights)


def assign_splits(
    tasks: list,
    *,
    sizes: dict[str, int] = SPLIT_SIZES,
    seed: int = SPLIT_SEED,
) -> tuple[dict[str, list], dict]:
    """Assign each task to exactly one of the milestone splits.

    Allocation is sequential: each split's size is spread over the remaining per-source
    pools in proportion to their remaining size (largest remainder, capped by each
    pool), so every split is source-stratified and the totals are exact.  Each source's
    seeded shuffled order is consumed front-to-back, making assignment deterministic
    and the splits disjoint.
    """
    if not tasks:
        raise ValueError("no tasks to split")
    targets = split_targets(len(tasks), sizes)
    order_by_source: dict[str, list] = {}
    for source in SOURCE_ORDER:
        pool = _sorted_pool([task for task in tasks if task.source == source])
        rng = random.Random(f"{seed}:{source}:split")
        rng.shuffle(pool)
        order_by_source[source] = pool
    remaining = {source: len(order_by_source[source]) for source in SOURCE_ORDER}
    consumed = dict.fromkeys(SOURCE_ORDER, 0)
    assigned: dict[str, list] = {name: [] for name in SPLIT_ORDER}
    by_source: dict[str, dict[str, int]] = {name: {} for name in SPLIT_ORDER}
    for split in SPLIT_ORDER:
        size = targets[split]
        quotas = apportion(size, remaining, caps=remaining)
        for source in SOURCE_ORDER:
            count = quotas[source]
            if count:
                start = consumed[source]
                assigned[split].extend(order_by_source[source][start : start + count])
                consumed[source] += count
                remaining[source] -= count
            by_source[split][source] = count
    if sum(len(assigned[split]) for split in SPLIT_ORDER) != len(tasks):
        raise AssertionError("split lost tasks")
    if any(remaining.values()):
        raise AssertionError(f"split left tasks unassigned: {remaining}")
    for split in SPLIT_ORDER:
        if len(assigned[split]) != targets[split]:
            raise AssertionError(
                f"{split}: {len(assigned[split])} tasks, expected {targets[split]}"
            )
    info = {
        "split_seed": seed,
        "split_sizes": targets,
        "split_by_source": by_source,
        "algorithm": (
            "seeded per-source shuffle; sequential largest-remainder allocation of each "
            "split size over the remaining source pools, capped by each pool"
        ),
        "heldout_untouched": "held-out tasks are never assigned to train, evolution or pilot",
        "pilot_disjoint": True,
    }
    return assigned, info
