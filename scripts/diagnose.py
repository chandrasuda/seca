#!/usr/bin/env python3
"""Diagnose why gold solutions fail — 1000 concurrent workers, detailed breakdown."""
from __future__ import annotations
import json, sys, time, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.set_int_max_str_digits(0)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seca.data.apps import _safe_json, _build_tests, _get_fn_name
from seca.data.problem import CodeProblem
from seca.sandbox.executor import execute_code

TIMEOUT = 10.0


def check_one(idx, row):
    sols = _safe_json(row.get("solutions", "[]"))
    gold = sols[0] if isinstance(sols, list) and sols else ""
    io_raw = _safe_json(row.get("input_output", "{}"))
    tests = _build_tests(io_raw)
    fn = _get_fn_name(io_raw)

    if not gold.strip():
        return idx, "no_gold", "", fn
    if not tests:
        return idx, "no_tests", "", fn

    prob = CodeProblem(
        problem_id=str(row.get("problem_id", "")),
        prompt="", gold_solution=gold,
        test_cases=tests[:5], fn_name=fn,
    )
    fb = execute_code(gold, prob, timeout=TIMEOUT)

    if fb.all_passed:
        return idx, "pass", "", fn

    # Categorize
    for i, r in enumerate(fb.results):
        if r.timed_out:
            return idx, "timeout", f"test={i}", fn
        if r.stderr:
            err = r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "?"
            return idx, "runtime_error", err, fn
        if not r.passed:
            exp = prob.test_cases[i].expected_output[:80]
            got = r.stdout.strip()[:80]
            return idx, "wrong_output", f"exp={exp!r} got={got!r}", fn

    return idx, "unknown", "", fn


def main():
    src = ROOT / "data" / "apps_train.jsonl"
    rows = []
    for line in src.read_text().splitlines():
        try:
            rows.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            pass

    total = len(rows)
    print(f"Diagnosing {total} train problems  |  1000 workers  |  {TIMEOUT}s timeout\n")

    reasons = collections.Counter()
    details = []  # (idx, reason, detail, fn)
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=1000) as pool:
        futs = {pool.submit(check_one, i, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            idx, reason, detail, fn = fut.result()
            reasons[reason] += 1
            done += 1
            if reason not in ("pass", "no_gold", "no_tests"):
                details.append((idx, reason, detail, fn))
            if done % 200 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else 0
                p = reasons["pass"]
                print(f"  [{done:>5}/{total}]  pass={p}  {rate:.0f} prob/s  ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s  ({total/elapsed:.0f} prob/s)\n")

    print("=== Breakdown ===")
    for k, v in reasons.most_common():
        print(f"  {k:20s} {v:>5}  ({v/total*100:5.1f}%)")

    print(f"\n=== Sample failures (first 30) ===")
    # Sort by reason for readability
    details.sort(key=lambda x: x[1])
    for idx, reason, detail, fn in details[:30]:
        fn_tag = f"fn={fn}" if fn else "stdin"
        print(f"  [{idx:>4}] {reason:15s} {fn_tag:30s} {detail[:120]}")

    # Group runtime errors by error type
    err_types = collections.Counter()
    for _, reason, detail, _ in details:
        if reason == "runtime_error":
            # grab the exception class
            parts = detail.split(":")
            err_types[parts[0].strip()] += 1
    if err_types:
        print(f"\n=== Runtime error types ===")
        for k, v in err_types.most_common(15):
            print(f"  {k:50s} {v:>5}")

    # Group wrong_output by fn vs stdin
    wrong_fn = sum(1 for _, r, _, fn in details if r == "wrong_output" and fn)
    wrong_io = sum(1 for _, r, _, fn in details if r == "wrong_output" and not fn)
    if wrong_fn or wrong_io:
        print(f"\n=== Wrong output: fn_name={wrong_fn}  stdin/stdout={wrong_io} ===")


if __name__ == "__main__":
    main()
