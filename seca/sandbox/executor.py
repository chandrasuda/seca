"""Sandboxed code execution — runs student code against test cases."""
from __future__ import annotations
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from seca.data.problem import CodeProblem


def extract_code(text: str) -> str:
    """Extract Python code from completion (handles ```python ... ``` blocks)."""
    text = text.strip()
    # Try ```python ... ``` or ``` ... ```
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


@dataclass
class ExecResult:
    """Result of running code against a single test case."""
    passed: bool
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class FeedbackBundle:
    """Full execution feedback for one (problem, code) pair."""
    results: list[ExecResult]
    summary: str               # human-readable for teacher conditioning
    pass_rate: float

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)


def execute_code(
    code: str,
    problem: CodeProblem,
    timeout: float = 10.0,
    extract: bool = True,
) -> FeedbackBundle:
    """Run *code* against all test cases in *problem*; return feedback.
    If extract=True, extracts Python from ``` blocks."""
    if extract:
        code = extract_code(code)
    if not code.strip():
        return FeedbackBundle(
            results=[ExecResult(passed=False, stdout="", stderr="Empty code")],
            summary="Error: no executable code found in the completion.",
            pass_rate=0.0,
        )
    if not problem.test_cases:
        # no tests → just try to compile/run
        r = _run_snippet(code, stdin="", timeout=timeout)
        r.passed = not r.timed_out and not r.stderr
        summary = "No test cases. " + ("Compiled OK." if r.passed else f"Error: {r.stderr[:500]}")
        return FeedbackBundle(results=[r], summary=summary,
                              pass_rate=1.0 if r.passed else 0.0)

    results: list[ExecResult] = []
    for tc in problem.test_cases:
        r = _run_snippet(code, stdin=tc.input, timeout=timeout)
        actual = "\n".join(l.strip() for l in r.stdout.strip().splitlines())
        expected = "\n".join(l.strip() for l in tc.expected_output.strip().splitlines())
        r.passed = (actual == expected) and not r.timed_out
        results.append(r)

    passed = sum(r.passed for r in results)
    total = len(results)
    rate = passed / total if total else 0.0

    # build teacher-readable summary
    lines = [f"Passed {passed}/{total} test cases."]
    for i, r in enumerate(results):
        if not r.passed:
            lines.append(f"  Test {i+1}: FAIL")
            if r.timed_out:
                lines.append(f"    Reason: Timed out ({timeout}s)")
            elif r.stderr:
                lines.append(f"    Stderr: {r.stderr[:300]}")
            else:
                lines.append(f"    Expected: {problem.test_cases[i].expected_output[:100]}")
                lines.append(f"    Got:      {r.stdout[:100]}")
    summary = "\n".join(lines)

    return FeedbackBundle(results=results, summary=summary, pass_rate=rate)


# ── internal ──

def _run_snippet(code: str, stdin: str = "", timeout: float = 10.0) -> ExecResult:
    """Execute a Python snippet in a subprocess."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        path = f.name

    try:
        proc = subprocess.run(
            ["python3", path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecResult(
            passed=False,  # set by caller
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(passed=False, stdout="", stderr="", timed_out=True)
    finally:
        Path(path).unlink(missing_ok=True)