"""APPS loader — codeparrot/apps (10k problems, 3 difficulty tiers)."""
from __future__ import annotations
import json
from datasets import load_dataset
from seca.data.problem import CodeProblem, TestCase


def load_apps(
    split: str = "test",
    difficulty: str | None = None,
    max_problems: int | None = None,
) -> list[CodeProblem]:
    ds = load_dataset("codeparrot/apps", split=split)
    if difficulty:
        ds = ds.filter(lambda x: x.get("difficulty") == difficulty)
    if max_problems:
        ds = ds.select(range(min(max_problems, len(ds))))

    problems: list[CodeProblem] = []
    for row in ds:
        # parse solutions list
        sols = _safe_json(row.get("solutions", "[]"))
        gold = sols[0] if isinstance(sols, list) and sols else ""

        # parse test IO
        io = _safe_json(row.get("input_output", "{}"))
        tests = _build_tests(io)

        problems.append(CodeProblem(
            problem_id=str(row.get("problem_id", "")),
            prompt=row.get("question", ""),
            gold_solution=gold,
            test_cases=tests,
            starter_code=row.get("starter_code", ""),
            difficulty=row.get("difficulty", ""),
            source="apps",
        ))
    return problems


def _safe_json(raw) -> any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _build_tests(io: dict | str) -> list[TestCase]:
    if isinstance(io, str):
        io = _safe_json(io)
    if not isinstance(io, dict):
        return []
    inputs = io.get("inputs", [])
    outputs = io.get("outputs", [])
    return [
        TestCase(input=str(i), expected_output=str(o))
        for i, o in zip(inputs, outputs)
    ]
