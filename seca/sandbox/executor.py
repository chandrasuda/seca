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

    Primary format (preferred): <start_code> ... <end_code> or </start_code>
    Models sometimes output XML-style </start_code> instead of <end_code>.
    Fallback:
      - <start_code> without end tag → take everything after last start_tag
      - ```python ... ```
      - ```py ... ```
      - ``` ... ``` (no lang tag)
      - Raw code with no block (returns as-is)
    Takes the last/largest block if multiple exist.
    Strips tags from output so they never end up in executed code (SyntaxError).
    """
    text = text.strip()
    start_tag = "<start_code>"
    # Accept both <end_code> and XML-style </start_code> as end delimiters
    end_tags = ("<end_code>", "</start_code>")

    if start_tag in text:
        after_start = text.split(start_tag)[-1]
        # Find content before first occurrence of any end tag
        earliest = len(after_start)
        for et in end_tags:
            idx = after_start.find(et)
            if idx >= 0 and idx < earliest:
                earliest = idx
        before_end = after_start[:earliest] if earliest < len(after_start) else after_start
        extracted = before_end.strip()
        if extracted:
            # Strip any tag lines that snuck through (leading/trailing)
            for tag in (start_tag,) + end_tags:
                extracted = extracted.replace(tag, "")
            extracted = extracted.strip()
            if extracted:
                return extracted
        # Partial: has start_tag but empty/no valid content before end — try rest
        if any(et in after_start for et in end_tags):
            return ""  # content between tags was empty
        return after_start.strip() if after_start.strip() else ""

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


def _typed_arg_positions(problem: CodeProblem, type_token: str) -> list[int]:
    """Infer which function args mention a given type token in starter_code."""
    if not problem.fn_name or not problem.starter_code:
        return []
    sig_line = None
    for line in problem.starter_code.splitlines():
        if f"def {problem.fn_name}" in line:
            sig_line = line.strip()
            break
    if not sig_line or "(" not in sig_line or ")" not in sig_line:
        return []
    raw_params = [p.strip() for p in sig_line.split("(", 1)[1].rsplit(")", 1)[0].split(",")]
    params = [p for p in raw_params if p and p not in {"self", "cls"}]
    positions: list[int] = []
    for idx, p in enumerate(params):
        if type_token in p:
            positions.append(idx)
    return positions


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
    harness = problem.meta.get("leetcode_test_harness", "")
    entry_point = problem.meta.get("leetcode_entry_point", "")
    if harness and entry_point:
        return _execute_with_harness(
            code=code,
            harness=harness,
            entry_point=entry_point,
            timeout=timeout,
        )

    results: list[ExecResult] = []
    listnode_positions = _typed_arg_positions(problem, "ListNode")
    treenode_positions = _typed_arg_positions(problem, "TreeNode")
    for tc in problem.test_cases:
        if problem.fn_name:
            wrapper_lines = [
                "\nimport json as __j",
                f"__args = __j.loads({tc.input!r})",
                "if not isinstance(__args, list):",
                "    __args = [__args]",
            ]
            if listnode_positions:
                wrapper_lines.extend(
                    [
                        "def __to_listnode(x):",
                        "    if x is None:",
                        "        return None",
                        "    if not isinstance(x, list):",
                        "        return x",
                        "    __dummy = ListNode(0)",
                        "    __cur = __dummy",
                        "    for __v in x:",
                        "        __cur.next = ListNode(__v)",
                        "        __cur = __cur.next",
                        "    return __dummy.next",
                        "def __from_listnode(node):",
                        "    __out = []",
                        "    __guard = 0",
                        "    while node is not None and __guard < 10000:",
                        "        __out.append(node.val)",
                        "        node = node.next",
                        "        __guard += 1",
                        "    return __out",
                        f"__ln_positions = {listnode_positions!r}",
                        "for __i in __ln_positions:",
                        "    if 0 <= __i < len(__args):",
                        "        __args[__i] = __to_listnode(__args[__i])",
                    ]
                )
            if treenode_positions:
                wrapper_lines.extend(
                    [
                        "def __to_treenode(vals):",
                        "    if vals is None:",
                        "        return None",
                        "    if not isinstance(vals, list):",
                        "        return vals",
                        "    if not vals:",
                        "        return None",
                        "    __nodes = [TreeNode(v) if v is not None else None for v in vals]",
                        "    __kids = __nodes[::-1]",
                        "    __root = __kids.pop()",
                        "    for __node in __nodes:",
                        "        if __node is not None:",
                        "            if __kids:",
                        "                __node.left = __kids.pop()",
                        "            if __kids:",
                        "                __node.right = __kids.pop()",
                        "    return __root",
                        "def __from_treenode(root):",
                        "    if root is None:",
                        "        return []",
                        "    __out = []",
                        "    __q = [root]",
                        "    __idx = 0",
                        "    while __idx < len(__q):",
                        "        __node = __q[__idx]",
                        "        __idx += 1",
                        "        if __node is None:",
                        "            __out.append(None)",
                        "            continue",
                        "        __out.append(__node.val)",
                        "        __q.append(__node.left)",
                        "        __q.append(__node.right)",
                        "    while __out and __out[-1] is None:",
                        "        __out.pop()",
                        "    return __out",
                        f"__tn_positions = {treenode_positions!r}",
                        "for __i in __tn_positions:",
                        "    if 0 <= __i < len(__args):",
                        "        __args[__i] = __to_treenode(__args[__i])",
                    ]
                )
            wrapper_lines.extend(
                [
                    "try:",
                    f"    __result = {problem.fn_name}(*__args)",
                    "except NameError:",
                    f"    __result = Solution().{problem.fn_name}(*__args)",
                ]
            )
            if listnode_positions:
                wrapper_lines.extend(
                    [
                        "if hasattr(__result, 'val') and hasattr(__result, 'next'):",
                        "    __result = __from_listnode(__result)",
                    ]
                )
            if treenode_positions:
                wrapper_lines.extend(
                    [
                        "if hasattr(__result, 'val') and hasattr(__result, 'left') and hasattr(__result, 'right'):",
                        "    __result = __from_treenode(__result)",
                    ]
                )
            wrapper_lines.append("print(__j.dumps(__result, sort_keys=True))")
            wrapper = "\n".join(wrapper_lines) + "\n"
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


def _execute_with_harness(
    code: str,
    harness: str,
    entry_point: str,
    timeout: float,
) -> FeedbackBundle:
    """Execute candidate against original LeetCode-style test harness."""
    marker = "__SECA_HARNESS_PASS__"
    wrapper = (
        "\nimport traceback as __tb\n"
        "try:\n"
        f"    __candidate = {entry_point}\n"
        "    check(__candidate)\n"
        f"    print('{marker}')\n"
        "except Exception:\n"
        "    __tb.print_exc()\n"
        "    raise\n"
    )
    snippet = code + "\n" + harness + "\n" + wrapper
    r = _run_snippet(snippet, stdin="", timeout=timeout)
    passed = (not r.timed_out) and (not r.stderr) and (marker in r.stdout)
    r.passed = passed
    summary = (
        "Passed harness checks."
        if passed
        else "Failed harness checks."
    )
    if r.timed_out:
        summary += f" Timed out ({timeout}s)."
    elif r.stderr:
        summary += f" Stderr: {r.stderr[:300]}"
    return FeedbackBundle(
        results=[r],
        summary=summary,
        pass_rate=1.0 if passed else 0.0,
        extracted_code=code,
        first_failure_stderr="" if passed else r.stderr,
        first_failure_stdout="" if passed else r.stdout,
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

# LeetCode-style completions often rely on typing aliases (List, Optional, etc.)
# and platform-provided helper classes (e.g., ListNode) without importing them.
# Inject compatibility definitions so such code executes in our sandbox.
_EXEC_PRELUDE = (
    "from typing import *\n"
    "inf = float('inf')\n"
    "class ListNode:\n"
    "    def __init__(self, val=0, next=None):\n"
    "        self.val = val\n"
    "        self.next = next\n"
    "class TreeNode:\n"
    "    def __init__(self, val=0, left=None, right=None):\n"
    "        self.val = val\n"
    "        self.left = left\n"
    "        self.right = right\n"
    "def list_node(vals):\n"
    "    if vals is None:\n"
    "        return None\n"
    "    if not isinstance(vals, list):\n"
    "        return vals\n"
    "    dummy = ListNode(0)\n"
    "    cur = dummy\n"
    "    for v in vals:\n"
    "        cur.next = ListNode(v)\n"
    "        cur = cur.next\n"
    "    return dummy.next\n"
    "def is_same_list(a, b):\n"
    "    guard = 0\n"
    "    while a is not None and b is not None and guard < 100000:\n"
    "        if a.val != b.val:\n"
    "            return False\n"
    "        a = a.next\n"
    "        b = b.next\n"
    "        guard += 1\n"
    "    return a is None and b is None\n"
    "def tree_node(vals):\n"
    "    if vals is None:\n"
    "        return None\n"
    "    if not isinstance(vals, list):\n"
    "        return vals\n"
    "    if not vals:\n"
    "        return None\n"
    "    nodes = [TreeNode(v) if v is not None else None for v in vals]\n"
    "    kids = nodes[::-1]\n"
    "    root = kids.pop()\n"
    "    for node in nodes:\n"
    "        if node is not None:\n"
    "            if kids:\n"
    "                node.left = kids.pop()\n"
    "            if kids:\n"
    "                node.right = kids.pop()\n"
    "    return root\n"
    "def is_same_tree(a, b):\n"
    "    qa, qb = [a], [b]\n"
    "    while qa and qb:\n"
    "        na = qa.pop(0)\n"
    "        nb = qb.pop(0)\n"
    "        if na is None and nb is None:\n"
    "            continue\n"
    "        if na is None or nb is None:\n"
    "            return False\n"
    "        if na.val != nb.val:\n"
    "            return False\n"
    "        qa.extend([na.left, na.right])\n"
    "        qb.extend([nb.left, nb.right])\n"
    "    return len(qa) == len(qb)\n"
)


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
            f.write(_EXEC_PRELUDE)
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
