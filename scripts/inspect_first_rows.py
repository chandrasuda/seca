#!/usr/bin/env python3
"""Inspect first row of train & test, and run gold solution to diagnose failures."""
import json, sys
from pathlib import Path

sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seca.data.apps import _safe_json, _build_tests, _get_fn_name
from seca.data.problem import CodeProblem
from seca.sandbox.executor import execute_code


def inspect(path: Path):
    if not path.exists():
        print(f"  ⚠  {path} not found")
        return
    with open(path) as f:
        row = json.loads(f.readline())

    print(f"\n{'='*60}")
    print(f"  FILE: {path}")
    print(f"{'='*60}")
    print(f"  Keys:        {list(row.keys())}")
    print(f"  problem_id:  {row.get('problem_id')}")
    print(f"  difficulty:  {row.get('difficulty')}")
    print(f"  question:    {str(row.get('question',''))[:200]}...")
    print(f"  starter_code:{repr(row.get('starter_code',''))[:100]}")

    sols = _safe_json(row.get("solutions", "[]"))
    print(f"  # solutions: {len(sols) if isinstance(sols, list) else 'NOT A LIST'}")

    gold = sols[0] if isinstance(sols, list) and sols else ""
    if gold:
        print(f"  gold[0] (300 chars): {gold[:300]}")
    else:
        print(f"  gold: EMPTY — this would be marked as fail")

    io = _safe_json(row.get("input_output", "{}"))
    tests = _build_tests(io)
    fn_name = _get_fn_name(io)
    print(f"  fn_name:     {fn_name or '(stdin/stdout)'}")
    print(f"  # test cases: {len(tests)}")
    if tests:
        print(f"    input[0]:    {repr(tests[0].input)[:150]}")
        print(f"    expected[0]: {repr(tests[0].expected_output)[:150]}")

    if gold.strip() and tests:
        print(f"\n  Running gold solution against tests (timeout=10s)...")
        prob = CodeProblem(
            problem_id=str(row.get("problem_id", row.get("id", ""))),
            prompt="", gold_solution=gold, test_cases=tests,
            fn_name=fn_name,
        )
        fb = execute_code(gold, prob, timeout=10.0)
        print(f"  pass_rate: {fb.pass_rate}")
        print(f"  all_passed: {fb.all_passed}")
        for i, r in enumerate(fb.results):
            status = "PASS" if r.passed else ("TIMEOUT" if r.timed_out else "FAIL")
            print(f"    test {i}: {status}")
            if not r.passed and r.stderr:
                print(f"      stderr: {r.stderr[:200]}")
            if not r.passed and not r.timed_out and not r.stderr:
                print(f"      expected: {repr(tests[i].expected_output)[:100]}")
                print(f"      got:      {repr(r.stdout)[:100]}")
    else:
        print(f"\n  ⚠  No gold or no tests — auto-fail in filter")


data = ROOT / "data"
inspect(data / "apps_train.jsonl")
inspect(data / "apps_test.jsonl")
