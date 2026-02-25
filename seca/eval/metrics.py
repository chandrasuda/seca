"""Evaluation: Pass@k (unbiased) and execution success rate."""
from __future__ import annotations
import math
import torch
from seca.data.problem import CodeProblem
from seca.sandbox.executor import execute_code


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k estimator (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


@torch.no_grad()
def evaluate_problems(model, problems: list[CodeProblem], n_samples: int = 10,
                      k_values: list[int] | None = None, temperature: float = 0.8,
                      timeout: float = 10.0) -> dict:
    k_values = k_values or [1, 5, 10]
    all_pass_at: dict[int, list[float]] = {k: [] for k in k_values}
    total_pass_rate = 0.0

    for problem in problems:
        prompts = [problem.format_prompt()] * n_samples
        completions = model.generate(
            prompts, temperature=temperature, do_sample=True,
        )

        # execute each completion
        n_correct = 0
        problem_pass_rates: list[float] = []
        for code in completions:
            fb = execute_code(code, problem, timeout=timeout)
            if fb.all_passed:
                n_correct += 1
            problem_pass_rates.append(fb.pass_rate)

        total_pass_rate += sum(problem_pass_rates) / len(problem_pass_rates)

        for k in k_values:
            all_pass_at[k].append(pass_at_k(n_samples, n_correct, min(k, n_samples)))

    n = len(problems)
    results = {
        f"pass@{k}": sum(v) / n if n else 0.0
        for k, v in all_pass_at.items()
    }
    results["mean_test_pass_rate"] = total_pass_rate / n if n else 0.0
    results["n_problems"] = n
    return results
