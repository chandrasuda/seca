"""Shared data loading — single entry point for all datasets."""
from __future__ import annotations
from seca.data.problem import CodeProblem


def load_problems(data_cfg: dict) -> list[CodeProblem]:
    ds = data_cfg["dataset"]

    if ds == "apps":
        from seca.data.apps import load_apps
        return load_apps(
            split=data_cfg.get("split", "test"),
            difficulty=data_cfg.get("difficulty"),
            max_problems=data_cfg.get("max_problems"),
        )
    elif ds == "livecodebench":
        from seca.data.livecodebench import load_livecodebench
        return load_livecodebench(
            split=data_cfg.get("split", "test"),
            max_problems=data_cfg.get("max_problems"),
        )
    elif ds == "kernelbench":
        from seca.data.kernelbench import load_kernelbench
        return load_kernelbench(
            split=data_cfg.get("split", "test"),
            level=data_cfg.get("level"),
            max_problems=data_cfg.get("max_problems"),
        )
    else:
        raise ValueError(f"Unknown dataset: {ds}. Use: apps | livecodebench | kernelbench")
