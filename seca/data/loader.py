"""Shared data loading — single entry point for all datasets."""
from __future__ import annotations
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from seca.data.problem import CodeProblem

# Each worker just waits on a subprocess (I/O-bound), so heavy oversub is fine.
_DEFAULT_WORKERS = 500


def load_problems(data_cfg: dict) -> list[CodeProblem]:
    ds = data_cfg["dataset"]
    filter_bad_gold = data_cfg.get("filter_bad_gold", False)

    if ds == "apps":
        from seca.data.apps import load_apps
        probs = load_apps(
            split=data_cfg.get("split", "test"),
            difficulty=data_cfg.get("difficulty"),
            max_problems=data_cfg.get("max_problems"),
        )
        if filter_bad_gold:
            probs = _filter_bad_gold(probs)
        return probs
    elif ds == "livecodebench":
        from seca.data.livecodebench import load_livecodebench
        probs = load_livecodebench(
            split=data_cfg.get("split", "test"),
            max_problems=data_cfg.get("max_problems"),
        )
        if filter_bad_gold:
            probs = _filter_bad_gold(probs)
        return probs
    elif ds == "kernelbench":
        from seca.data.kernelbench import load_kernelbench
        probs = load_kernelbench(
            split=data_cfg.get("split", "test"),
            level=data_cfg.get("level"),
            max_problems=data_cfg.get("max_problems"),
        )
        if filter_bad_gold:
            probs = _filter_bad_gold(probs)
        return probs
    else:
        raise ValueError(f"Unknown dataset: {ds}. Use: apps | livecodebench | kernelbench")


def _check_one(problem: CodeProblem) -> tuple[CodeProblem, bool]:
    """Run one problem's gold solution against its tests. Returns (problem, passed)."""
    from seca.sandbox.executor import execute_code
    fb = execute_code(problem.gold_solution, problem, extract=False)
    return (problem, fb.all_passed)


def _filter_bad_gold(
    problems: list[CodeProblem],
    max_workers: int | None = None,
) -> list[CodeProblem]:
    """Drop problems whose gold solution fails its own tests (parallel)."""
    workers = max_workers or _DEFAULT_WORKERS

    # Pre-filter obviously unusable problems (no gold / no tests)
    candidates = [p for p in problems if p.gold_solution.strip() and p.test_cases]
    total = len(candidates)
    print(f"[filter_bad_gold] Testing {total} gold solutions with {workers} workers …")
    sys.stdout.flush()

    kept: list[CodeProblem] = []
    done = 0
    passed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_check_one, p): p for p in candidates}
        for fut in as_completed(futures):
            done += 1
            prob, ok = fut.result()
            if ok:
                kept.append(prob)
                passed += 1
            else:
                failed += 1
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] ✓ {passed}  ✗ {failed}")
                sys.stdout.flush()

    # Preserve original ordering
    kept_ids = {id(p) for p in kept}
    kept_ordered = [p for p in candidates if id(p) in kept_ids]

    print(f"[filter_bad_gold] Kept {len(kept_ordered)}/{total} problems.")
    sys.stdout.flush()
    return kept_ordered
