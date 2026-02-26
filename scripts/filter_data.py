#!/usr/bin/env python3
"""Filter APPS data — keep only problems whose gold solution passes all tests.

Reads from:    data/apps_train.jsonl, data/apps_test.jsonl
Writes to:     data_filtered/apps_train.jsonl, data_filtered/apps_test.jsonl

Uses a ThreadPoolExecutor (default 500 workers) so that hundreds of
subprocess-based test executions run in parallel.

Usage:
    python scripts/filter_data.py                        # both splits
    python scripts/filter_data.py --split train          # single split
    python scripts/filter_data.py --workers 200          # fewer workers
    python scripts/filter_data.py --timeout 15           # longer per-test timeout
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import collections

# APPS has huge integers; Python 3.11+ limits int→str conversion by default.
sys.set_int_max_str_digits(0)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seca.data.apps import _safe_json, _build_tests, _get_fn_name
from seca.data.problem import CodeProblem, TestCase
from seca.sandbox.executor import execute_code

SRC_DIR = ROOT / "data"
DST_DIR = ROOT / "data_filtered"
SPLITS = ("train", "test")


# ── per-problem worker function ─────────────────────────────────────────────

def _check_one(idx: int, row: dict, timeout: float) -> tuple[int, dict, bool, str, str]:
    """Return (index, row, passed, reason, detail)."""
    sols = _safe_json(row.get("solutions", "[]"))
    gold = sols[0] if isinstance(sols, list) and sols else ""
    io_raw = _safe_json(row.get("input_output", "{}"))
    tests = _build_tests(io_raw)
    fn_name = _get_fn_name(io_raw)

    if not gold.strip():
        return idx, row, False, "no_gold", ""
    if not tests:
        return idx, row, False, "no_tests", ""

    prob = CodeProblem(
        problem_id=str(row.get("problem_id", row.get("id", ""))),
        prompt="",
        gold_solution=gold,
        test_cases=tests,
        fn_name=fn_name,
    )
    fb = execute_code(gold, prob, timeout=timeout)
    if fb.all_passed:
        return idx, row, True, "pass", ""

    for i, r in enumerate(fb.results):
        if r.timed_out:
            return idx, row, False, "timeout", f"test={i}"
        if (r.stderr or "").strip():
            err = r.stderr.strip().splitlines()[-1][:200]
            return idx, row, False, "runtime_error", err
        if not r.passed:
            exp = prob.test_cases[i].expected_output[:80]
            got = r.stdout.strip()[:80]
            return idx, row, False, "wrong_output", f"exp={exp!r} got={got!r}"

    return idx, row, False, "unknown", ""


# ── filter one split ────────────────────────────────────────────────────────

def filter_split(split: str, workers: int, timeout: float) -> None:
    src = SRC_DIR / f"apps_{split}.jsonl"
    if not src.exists():
        print(f"  ⚠  {src} not found — skipping")
        return

    rows = []
    skipped_parse = 0
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            skipped_parse += 1
    total = len(rows)
    print(f"\n{'='*60}")
    print(f"  Split: {split}   Source: {src}")
    if skipped_parse:
        print(f"  ⚠  Skipped {skipped_parse} rows with bad JSON")
    print(f"  {total} problems  |  {workers} workers  |  {timeout}s timeout")
    print(f"{'='*60}")

    kept: list[tuple[int, dict]] = []
    done = 0
    stats = collections.Counter()
    failure_samples: dict[str, list[str]] = collections.defaultdict(list)
    t0 = time.time()

    DST_DIR.mkdir(parents=True, exist_ok=True)
    dst = DST_DIR / f"apps_{split}.jsonl"

    def _save_checkpoint():
        """Sort by original index and write all kept rows so far."""
        snap = sorted(kept, key=lambda x: x[0])
        with open(dst, "w") as f:
            for _, r in snap:
                f.write(json.dumps(r) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_check_one, i, r, timeout): i
            for i, r in enumerate(rows)
        }
        for fut in as_completed(futures):
            idx, row, ok, reason, detail = fut.result()
            done += 1

            if ok:
                kept.append((idx, row))
                stats["pass"] += 1
            else:
                stats[reason] += 1
                if reason in ("runtime_error", "wrong_output", "timeout", "unknown") and len(failure_samples[reason]) < 10:
                    pid = row.get("problem_id", row.get("id", "?"))
                    failure_samples[reason].append(f"pid={pid} idx={idx} {detail}".strip())

            fail_non_timeout = stats["runtime_error"] + stats["wrong_output"] + stats["unknown"] + stats["no_gold"] + stats["no_tests"]

            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else 0
                print(
                    f"  [{done:>5}/{total}]  "
                    f"pass={stats['pass']}  fail={fail_non_timeout}  timeout={stats['timeout']}  "
                    f"(runtime={stats['runtime_error']} wrong={stats['wrong_output']} no_gold={stats['no_gold']} no_tests={stats['no_tests']})  "
                    f"{rate:.1f} prob/s  ETA {eta:.0f}s",
                    flush=True,
                )
                # checkpoint every 100 problems
                _save_checkpoint()

    # final save
    _save_checkpoint()

    elapsed = time.time() - t0
    mb = dst.stat().st_size / 1024 / 1024
    fail_non_timeout = stats["runtime_error"] + stats["wrong_output"] + stats["unknown"] + stats["no_gold"] + stats["no_tests"]

    print(f"\n  ── {split} summary ──")
    print(f"  Total:            {total}")
    print(f"  ✓ pass:           {stats['pass']}")
    print(f"  ✗ fail:           {fail_non_timeout}")
    print(f"  ⏱ timeout:        {stats['timeout']}")
    print(f"  · runtime_error:  {stats['runtime_error']}")
    print(f"  · wrong_output:   {stats['wrong_output']}")
    print(f"  · no_gold:        {stats['no_gold']}")
    print(f"  · no_tests:       {stats['no_tests']}")
    print(f"  · unknown:        {stats['unknown']}")
    print(f"  Saved → {dst}  ({mb:.1f} MB, {elapsed:.1f}s)")

    log_path = DST_DIR / f"apps_{split}_filter_failures.log"
    with open(log_path, "w") as lf:
        lf.write(f"split={split} total={total} pass={stats['pass']} fail={fail_non_timeout} timeout={stats['timeout']}\n")
        lf.write(f"runtime_error={stats['runtime_error']} wrong_output={stats['wrong_output']} no_gold={stats['no_gold']} no_tests={stats['no_tests']} unknown={stats['unknown']} bad_json={skipped_parse}\n\n")
        for reason in ("timeout", "runtime_error", "wrong_output", "unknown"):
            lf.write(f"[{reason}]\n")
            for s in failure_samples.get(reason, []):
                lf.write(f"  {s}\n")
            lf.write("\n")
    print(f"  Failure log → {log_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Filter APPS to gold-passing problems")
    ap.add_argument("--split", default="all", help="'train', 'test', or 'all'")
    default_workers = min(64, (os.cpu_count() or 8) * 2)
    ap.add_argument("--workers", type=int, default=default_workers, help=f"Concurrent workers (default {default_workers})")
    ap.add_argument("--timeout", type=float, default=30.0, help="Per-test timeout in seconds")
    args = ap.parse_args()

    splits = list(SPLITS) if args.split == "all" else [args.split]
    for s in splits:
        filter_split(s, workers=args.workers, timeout=args.timeout)

    print(f"\n{'='*60}")
    print("All done ✓")


if __name__ == "__main__":
    main()
