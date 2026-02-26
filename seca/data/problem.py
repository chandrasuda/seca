"""Core data abstraction: a coding problem with tests + gold solution."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestCase:
    input: str
    expected_output: str


@dataclass
class CodeProblem:
    problem_id: str
    prompt: str
    gold_solution: str
    test_cases: list[TestCase] = field(default_factory=list)
    starter_code: str = ""
    difficulty: str = ""
    source: str = ""
    fn_name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def format_prompt(self) -> str:
        parts = [self.prompt]
        if self.starter_code:
            parts.append(f"\n```python\n{self.starter_code}\n```")
        parts.append(
            "\nWrite a complete Python solution. Your response must contain ONLY executable code "
            "enclosed between <start_code> and <end_code> tokens. No explanation, no markdown. "
            "Example format:\n<start_code>\n# your code here\n<end_code>"
        )
        return "\n".join(parts)
