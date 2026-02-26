"""APPS loader — local JSONL first, HuggingFace fallback."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from seca.data.problem import CodeProblem, TestCase

# Some APPS problems have test cases with huge integers (9000+ digits).
# Python 3.11+ limits int→str conversion by default; lift it.
sys.set_int_max_str_digits(0)

LOCAL_DIR = Path(__file__).resolve().parents[2] / "data"

# Every split the HF repo actually contains.
_ALL_SPLITS = ("train", "test")


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
        io_raw = _safe_json(row.get("input_output", "{}"))
        tests = _build_tests(io_raw)
        fn_name = _get_fn_name(io_raw)
        # skip problems missing gold or tests — unusable for distillation
        if not gold.strip() or not tests:
            continue
        problems.append(CodeProblem(
            problem_id=str(row.get("problem_id", row.get("id", ""))),
            prompt=row.get("question", ""),
            gold_solution=gold,
            test_cases=tests,
            starter_code=row.get("starter_code", ""),
            difficulty=row.get("difficulty", ""),
            source="apps",
            fn_name=fn_name,
        ))
    return problems


def _load_rows(split: str, difficulty: str | None) -> list[dict]:
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
