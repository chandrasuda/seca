"""Shared data loading — single entry point for all datasets."""
from __future__ import annotations
from seca.data.problem import CodeProblem


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


def _filter_bad_gold(problems: list[CodeProblem]) -> list[CodeProblem]:
    """Drop problems whose gold solution fails its own tests."""
    from seca.sandbox.executor import execute_code

    kept: list[CodeProblem] = []
    for p in problems:
        if not p.gold_solution.strip() or not p.test_cases:
            continue
        fb = execute_code(p.gold_solution, p, extract=False)
        if fb.all_passed:
            kept.append(p)
    return kept
