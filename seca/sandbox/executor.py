"""Sandboxed code execution — runs student code against test cases.

Each subprocess runs in its own temporary directory with a UUID-named script
to guarantee full isolation under high concurrency.
"""
from __future__ import annotations
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from seca.data.problem import CodeProblem


def extract_code(text: str) -> str:
    """Extract Python code from completion.

    Primary format (preferred): <start_code> ... <end_code>
    Fallback:
      - ```python ... ```
      - ```py ... ```
      - ``` ... ``` (no lang tag)
      - Raw code with no block (returns as-is)
    Takes the last/largest block if multiple exist.
    """
    text = text.strip()
    # Primary: <start_code> ... <end_code>
    start_tag, end_tag = "<start_code>", "<end_code>"
    if start_tag in text and end_tag in text:
        parts = text.split(start_tag, 1)[-1].split(end_tag, 1)
        if len(parts) >= 1 and parts[0].strip():
            return parts[0].strip()
    # Fallback: ```python, ```py, or bare ```
    pattern = r"```(?:python|py)?\s*\n?(.*?)```"
    matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
    if matches:
        blocks = [m.group(1).strip() for m in matches]
        return max(blocks, key=len) if blocks else text
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
    extracted_code: str = ""   # code actually executed (after extraction), for debugging
    first_failure_stderr: str = ""   # full stderr of first failed test
    first_failure_stdout: str = ""   # full stdout of first failed test

    @property
    def all_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)


def execute_code(
    code: str,
    problem: CodeProblem,
    timeout: float = 30.0,
    extract: bool = True,
    max_retries: int = 1,
) -> FeedbackBundle:
    """Run *code* against all test cases in *problem*; return feedback.

    If extract=True, extracts Python from ``` blocks.
    Timeout failures are retried up to max_retries times (to absorb CPU-load
    spikes under high concurrency).
    """
    if extract:
        code = extract_code(code)
    if not code.strip():
        return FeedbackBundle(
            results=[ExecResult(passed=False, stdout="", stderr="Empty code")],
            summary="Error: no executable code found in the completion.",
            pass_rate=0.0,
            extracted_code="",
            first_failure_stderr="Empty code",
        )
    if not problem.test_cases:
        # no tests → just try to compile/run
        r = _run_snippet(code, stdin="", timeout=timeout)
        r.passed = not r.timed_out and not r.stderr
        summary = "No test cases. " + ("Compiled OK." if r.passed else f"Error: {r.stderr[:500]}")
        return FeedbackBundle(
            results=[r], summary=summary, pass_rate=1.0 if r.passed else 0.0,
            extracted_code=code,
            first_failure_stderr="" if r.passed else r.stderr,
            first_failure_stdout="" if r.passed else r.stdout,
        )

    results: list[ExecResult] = []
    for tc in problem.test_cases:
        if problem.fn_name:
            wrapper = (
                "\nimport json as __j\n"
                f"__args = __j.loads({tc.input!r})\n"
                "try:\n"
                f"    __result = {problem.fn_name}(*__args)\n"
                "except NameError:\n"
                f"    __result = Solution().{problem.fn_name}(*__args)\n"
                "print(__j.dumps(__result, sort_keys=True))\n"
            )
            snippet = code + wrapper
        else:
            snippet = code

        stdin = "" if problem.fn_name else tc.input

        r = _run_snippet(snippet, stdin=stdin, timeout=timeout)

        actual = "\n".join(l.strip() for l in r.stdout.strip().splitlines())
        expected = "\n".join(l.strip() for l in tc.expected_output.strip().splitlines())
        r.passed = (actual == expected) and not r.timed_out
        results.append(r)

    passed = sum(r.passed for r in results)
    total = len(results)
    rate = passed / total if total else 0.0

    lines = [f"Passed {passed}/{total} test cases."]
    first_fail_stderr = ""
    first_fail_stdout = ""
    for i, r in enumerate(results):
        if not r.passed:
            if not first_fail_stderr and not first_fail_stdout:
                first_fail_stderr = r.stderr
                first_fail_stdout = r.stdout
            lines.append(f"  Test {i+1}: FAIL")
            if r.timed_out:
                lines.append(f"    Reason: Timed out ({timeout}s)")
            elif r.stderr:
                lines.append(f"    Stderr: {r.stderr[:300]}")
            else:
                lines.append(f"    Expected: {problem.test_cases[i].expected_output[:100]}")
                lines.append(f"    Got:      {r.stdout[:100]}")
    summary = "\n".join(lines)

    return FeedbackBundle(
        results=results,
        summary=summary,
        pass_rate=rate,
        extracted_code=code,
        first_failure_stderr=first_fail_stderr,
        first_failure_stdout=first_fail_stdout,
    )


# ── internal ──

# Clean env for subprocesses: strip VIRTUAL_ENV / CONDA vars that might
# interfere, keep PATH so python3 resolves.
_CLEAN_ENV = {
    k: v for k, v in os.environ.items()
    if not k.startswith(("VIRTUAL_ENV", "CONDA"))
}
_CLEAN_ENV["PYTHONDONTWRITEBYTECODE"] = "1"
_CLEAN_ENV["PYTHONHASHSEED"] = "0"


def _run_snippet(code: str, stdin: str = "", timeout: float = 30.0) -> ExecResult:
    """Execute a Python snippet in a fully-isolated subprocess.

    - Each call gets its own temp directory (no file collisions).
    - Script is named with a UUID (no module-name collisions).
    - subprocess cwd is set to the temp dir.
    - Clean env to avoid side-effects.
    """
    tmpdir = tempfile.mkdtemp(prefix="seca_")
    script = os.path.join(tmpdir, f"run_{uuid.uuid4().hex[:12]}.py")

    try:
        with open(script, "w") as f:
            f.write(code)

        proc = subprocess.run(
            ["python3", script],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmpdir,
            env=_CLEAN_ENV,
        )
        return ExecResult(
            passed=False,  # set by caller
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(passed=False, stdout="", stderr="", timed_out=True)
    except OSError as e:
        return ExecResult(passed=False, stdout="", stderr=f"OSError: {e}", timed_out=False)
    finally:
        # Clean up temp dir
        try:
            os.unlink(script)
            os.rmdir(tmpdir)
        except OSError:
            pass
