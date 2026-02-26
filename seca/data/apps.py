"""APPS loader — local JSONL first, HuggingFace fallback."""
from __future__ import annotations
import json
from pathlib import Path
from seca.data.problem import CodeProblem, TestCase

LOCAL_DIR = Path(__file__).resolve().parents[2] / "data"


def load_apps(
    split: str = "test",
    difficulty: str | None = None,
    max_problems: int | None = None,
) -> list[CodeProblem]:
    rows = _load_rows(split, difficulty)
    if max_problems:
        rows = rows[:max_problems]

    problems: list[CodeProblem] = []
    for row in rows:
        sols = _safe_json(row.get("solutions", "[]"))
        gold = sols[0] if isinstance(sols, list) and sols else ""
        tests = _build_tests(_safe_json(row.get("input_output", "{}")))
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


def _load_rows(split: str, difficulty: str | None) -> list[dict]:
    # try local JSONL first (e.g. data/apps_test_introductory.jsonl)
    if difficulty:
        local = LOCAL_DIR / f"apps_{split}_{difficulty}.jsonl"
        if local.exists():
            return [json.loads(l) for l in local.read_text().splitlines() if l.strip()]

    local_full = LOCAL_DIR / f"apps_{split}.jsonl"
    if local_full.exists():
        rows = [json.loads(l) for l in local_full.read_text().splitlines() if l.strip()]
        if difficulty:
            rows = [r for r in rows if r.get("difficulty") == difficulty]
        return rows

    # fallback: download from HuggingFace
    from datasets import load_dataset
    ds = load_dataset("codeparrot/apps", split=split)
    if difficulty:
        ds = ds.filter(lambda x: x.get("difficulty") == difficulty)
    return list(ds)


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
