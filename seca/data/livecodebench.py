"""LiveCodeBench v6 loader — livecodebench/code_generation_lite."""
from __future__ import annotations
import json
from datasets import load_dataset
from seca.data.problem import CodeProblem, TestCase


def load_livecodebench(
    split: str = "test",
    max_problems: int | None = None,
) -> list[CodeProblem]:
    ds = load_dataset("livecodebench/code_generation_lite", split=split)
    if max_problems:
        ds = ds.select(range(min(max_problems, len(ds))))

    problems: list[CodeProblem] = []
    for row in ds:
        tests = _parse_tests(row.get("public_test_cases", "[]"))
        problems.append(CodeProblem(
            problem_id=str(row.get("question_id", row.get("id", ""))),
            prompt=row.get("question_content", row.get("question", "")),
            gold_solution=row.get("python_solution", row.get("solution", "")),
            test_cases=tests,
            starter_code=row.get("starter_code", ""),
            difficulty=str(row.get("difficulty", "")),
            source="livecodebench",
        ))
    return problems


def _parse_tests(raw: str) -> list[TestCase]:
    """Parse JSON-encoded test cases."""
    try:
        cases = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return []
    out: list[TestCase] = []
    for c in (cases if isinstance(cases, list) else []):
        inp = c.get("input", c.get("stdin", ""))
        exp = c.get("expected_output", c.get("stdout", c.get("output", "")))
        out.append(TestCase(input=str(inp), expected_output=str(exp)))
    return out
