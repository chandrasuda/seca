"""KernelBench loader — ScalingIntelligence/KernelBench (CUDA kernels)."""
from __future__ import annotations
from datasets import load_dataset
from seca.data.problem import CodeProblem, TestCase


def load_kernelbench(
    split: str = "test",
    level: int | None = None,
    max_problems: int | None = None,
) -> list[CodeProblem]:
    ds = load_dataset("ScalingIntelligence/KernelBench", split=split)
    if level is not None:
        ds = ds.filter(lambda x: x.get("level") == level)
    if max_problems:
        ds = ds.select(range(min(max_problems, len(ds))))

    problems: list[CodeProblem] = []
    for row in ds:
        # KernelBench test = the validation script itself
        test_script = row.get("test_script", row.get("check_script", ""))
        tests = []
        if test_script:
            tests = [TestCase(input=test_script, expected_output="PASS")]

        problems.append(CodeProblem(
            problem_id=str(row.get("name", row.get("problem_id", ""))),
            prompt=row.get("prompt", row.get("description", "")),
            gold_solution=row.get("reference_code", row.get("solution", "")),
            test_cases=tests,
            starter_code=row.get("starter_code", ""),
            difficulty=f"level_{row.get('level', '?')}",
            source="kernelbench",
            meta={"level": row.get("level")},
        ))
    return problems
