"""APPS loader — local JSONL first, HuggingFace fallback."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from seca.data.problem import CodeProblem, TestCase

# Some APPS problems have test cases with huge integers (9000+ digits).
# Python 3.11+ limits int→str conversion by default; lift it.
sys.set_int_max_str_digits(0)

LOCAL_DIR = Path(__file__).resolve().parents[2] / "data_filtered"

# Every split the HF repo actually contains.
_ALL_SPLITS = ("train", "test")

# Runtime helpers/APIs that our lightweight executor does not emulate yet.
_UNSUPPORTED_LEETCODE_API_PATTERNS = (
    r"\bisBadVersion\s*\(",
    r"\bguess\s*\(",
    r"\bknows\s*\(",
    r"\bread4\s*\(",
    r"\brand7\s*\(",
    r"\bHtmlParser\b",
    r"\bMountainArray\b",
    r"\bBinaryMatrix\b",
    r"\bSea\b",
    r"\bArrayReader\b",
    r"\bCustomFunction\b",
    r"\bNestedInteger\b",
)


def load_apps(
    split: str = "test",
    difficulty: str | None = None,
    max_problems: int | None = None,
    data_file: str | None = None,
) -> list[CodeProblem]:
    rows = _load_rows(split, difficulty, data_file=data_file)
    if max_problems:
        rows = rows[:max_problems]

    problems: list[CodeProblem] = []
    for row in rows:
        sols = _safe_json(row.get("solutions", "[]"))
        gold = sols[0] if isinstance(sols, list) and sols else ""
        io_raw = _safe_json(row.get("input_output", "{}"))
        tests = _build_tests(io_raw)
        fn_name = _get_fn_name(io_raw)
        # skip problems missing gold or tests — unusable for distillation
        if not gold.strip() or not tests:
            continue
        if _should_skip_for_runtime_compat(row=row, io_raw=io_raw):
            continue
        meta: dict = {}
        if row.get("entry_point"):
            meta["leetcode_entry_point"] = row.get("entry_point")
        if row.get("test"):
            meta["leetcode_test_harness"] = row.get("test")
        if row.get("task_id"):
            meta["task_id"] = row.get("task_id")
        problems.append(CodeProblem(
            problem_id=str(row.get("problem_id", row.get("id", ""))),
            prompt=row.get("question", ""),
            gold_solution=gold,
            test_cases=tests,
            starter_code=row.get("starter_code", ""),
            difficulty=row.get("difficulty", ""),
            source="apps",
            fn_name=fn_name,
            meta=meta,
        ))
    return problems


def _should_skip_for_runtime_compat(row: dict, io_raw: dict | str) -> bool:
    """Skip rows that require unsupported online-judge runtime helpers.

    We currently support plain functions, ListNode and TreeNode conversions.
    """
    starter_code = row.get("starter_code", "") or ""
    has_harness = bool(row.get("test")) and bool(row.get("entry_point"))

    # Skip `Node`-typed tasks (distinct from ListNode/TreeNode) for now.
    # These include random-pointer/tree graph node variants not yet emulated.
    if re.search(r"\bNode\b", starter_code) and (
        "ListNode" not in starter_code and "TreeNode" not in starter_code
    ):
        return True

    # Skip API-backed interactive/judge helper tasks.
    for pat in _UNSUPPORTED_LEETCODE_API_PATTERNS:
        if re.search(pat, starter_code):
            return True

    # Converted datasets sometimes encode unimplemented cases as Error: ...
    # Keep these if harness is present (harness path can still execute correctly).
    if isinstance(io_raw, dict):
        outputs = io_raw.get("outputs", [])
        if any(isinstance(o, str) and o.startswith("Error:") for o in outputs) and not has_harness:
            return True

    return False


def _load_rows(
    split: str, difficulty: str | None, data_file: str | None = None
) -> list[dict]:
    if data_file:
        rows = _load_rows_from_file(data_file)
        if difficulty:
            rows = [r for r in rows if r.get("difficulty") == difficulty]
        return rows

    # "all" → merge every available split
    if split == "all":
        rows: list[dict] = []
        for s in _ALL_SPLITS:
            rows.extend(_load_rows_single(s))
        if difficulty:
            rows = [r for r in rows if r.get("difficulty") == difficulty]
        return rows

    rows = _load_rows_single(split)
    if difficulty:
        rows = [r for r in rows if r.get("difficulty") == difficulty]
    return rows


def _load_rows_single(split: str) -> list[dict]:
    """Load rows for one concrete split (local cache → HF download)."""
    local = LOCAL_DIR / f"apps_{split}.jsonl"
    if local.exists():
        return [json.loads(l) for l in local.read_text().splitlines() if l.strip()]

    # download from HuggingFace Hub
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="codeparrot/apps",
        filename=f"{split}.jsonl",
        repo_type="dataset",
    )
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    # cache locally for next time
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        import shutil
        shutil.copy(path, local)
    return rows


def _load_rows_from_file(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _safe_json(raw) -> any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _get_fn_name(io: dict | str) -> str:
    """Extract fn_name from input_output dict (empty for stdin/stdout problems)."""
    if isinstance(io, str):
        io = _safe_json(io)
    if not isinstance(io, dict):
        return ""
    return io.get("fn_name", "")


def _build_tests(io: dict | str) -> list[TestCase]:
    if isinstance(io, str):
        io = _safe_json(io)
    if not isinstance(io, dict):
        return []
    inputs = io.get("inputs", [])
    outputs = io.get("outputs", [])
    fn_name = io.get("fn_name", "")
    if fn_name:
        # Function-call problem: store args & expected return as JSON.
        # APPS wraps single return values in a list, e.g. ['ODD'] → unwrap.
        tests = []
        for i, o in zip(inputs, outputs):
            if isinstance(o, list) and len(o) == 1:
                o = o[0]
            tests.append(TestCase(
                input=json.dumps(i),
                expected_output=json.dumps(o, sort_keys=True) if not isinstance(o, str) else json.dumps(o),
            ))
        return tests
    return [
        TestCase(input=str(i), expected_output=str(o))
        for i, o in zip(inputs, outputs)
    ]
