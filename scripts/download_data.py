#!/usr/bin/env python3
"""Download ALL APPS splits from HuggingFace, validate gold solutions, keep passing ones.

Produces:
    data/apps_train.jsonl
    data/apps_test.jsonl

Usage:
    python scripts/download_data.py                # both splits
    python scripts/download_data.py --split test   # single split
    python scripts/download_data.py --workers 64   # more concurrency
    python scripts/download_data.py --timeout 30   # per-problem timeout
"""
from __future__ import annotations
import argparse, collections, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.set_int_max_str_digits(0)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seca.data.problem import CodeProblem, TestCase
from seca.sandbox.executor import execute_code
from seca.data.apps import _safe_json, _build_tests, _get_fn_name, LOCAL_DIR

_ALL_SPLITS = ("train", "test")


def _download_raw(split: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="codeparrot/apps",
        filename=f"{split}.jsonl",
        repo_type="dataset",
    )
    rows = []
    skipped = 0
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    if skipped:
        print(f"  ⚠  Skipped {skipped} rows with bad JSON")
    return rows


def _check_one(idx: int, row: dict, timeout: float) -> tuple[int, dict | None, str, str]:
    """Return (idx, row_or_None, reason, detail)."""
    sols = _safe_json(row.get("solutions", "[]"))
    gold = sols[0] if isinstance(sols, list) and sols else ""
    io_raw = _safe_json(row.get("input_output", "{}"))
    tests = _build_tests(io_raw)
    fn = _get_fn_name(io_raw)

    if not gold.strip():
        return idx, None, "no_gold", ""
    if not tests:
        return idx, None, "no_tests", ""

    prob = CodeProblem(
        problem_id=str(row.get("problem_id", row.get("id", ""))),
        prompt="", gold_solution=gold, test_cases=tests, fn_name=fn,
    )
    fb = execute_code(gold, prob, timeout=timeout)
    if fb.all_passed:
        return idx, row, "pass", ""

    for i, r in enumerate(fb.results):
        if r.timed_out:
            return idx, None, "timeout", f"test={i}"
        if (r.stderr or "").strip():
            err = r.stderr.strip().splitlines()[-1][:200]
            return idx, None, "runtime_error", err
        if not r.passed:
            exp = prob.test_cases[i].expected_output[:80]
            got = r.stdout.strip()[:80]
            return idx, None, "wrong_output", f"exp={exp!r} got={got!r}"

    return idx, None, "unknown", ""


def _download_split(split: str, workers: int, timeout: float) -> None:
    print(f"\n{'='*60}")
    print(f"  Downloading codeparrot/apps split={split} …")
    rows = _download_raw(split)
    total = len(rows)
    print(f"  {total} problems from HuggingFace")
    print(f"  {workers} workers  |  {timeout}s timeout")
    print(f"{'='*60}")

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOCAL_DIR / f"apps_{split}.jsonl"

    kept: list[tuple[int, dict]] = []
    stats = collections.Counter()
    failure_samples: dict[str, list[str]] = collections.defaultdict(list)
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_check_one, i, r, timeout): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            idx, row_out, reason, detail = fut.result()
            done += 1
            stats[reason] += 1

            if row_out is not None:
                kept.append((idx, row_out))
            elif reason in ("runtime_error", "wrong_output", "timeout", "unknown"):
                if len(failure_samples[reason]) < 10:
                    pid = rows[idx].get("problem_id", rows[idx].get("id", "?"))
                    failure_samples[reason].append(f"pid={pid} idx={idx} {detail}".strip())

            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else 0
                fail = stats["runtime_error"] + stats["wrong_output"] + stats["unknown"] + stats["no_gold"] + stats["no_tests"]
                print(
                    f"  [{done:>5}/{total}]  "
                    f"pass={stats['pass']}  fail={fail}  timeout={stats['timeout']}  "
                    f"(runtime={stats['runtime_error']} wrong={stats['wrong_output']} "
                    f"no_gold={stats['no_gold']} no_tests={stats['no_tests']})  "
                    f"{rate:.1f} prob/s  ETA {eta:.0f}s",
                    flush=True,
                )

    # Write kept rows in original order
    kept.sort(key=lambda x: x[0])
    with open(out_path, "w") as f:
        for _, row in kept:
            f.write(json.dumps({
                "problem_id": row.get("problem_id", ""),
                "question": row.get("question", ""),
                "solutions": row.get("solutions", "[]"),
                "input_output": row.get("input_output", "{}"),
                "difficulty": row.get("difficulty", ""),
                "starter_code": row.get("starter_code", ""),
            }) + "\n")

    elapsed = time.time() - t0
    mb = out_path.stat().st_size / 1024 / 1024
    fail = stats["runtime_error"] + stats["wrong_output"] + stats["unknown"] + stats["no_gold"] + stats["no_tests"]

    print(f"\n  ── {split} summary ──")
    print(f"  Total from HF:      {total}")
    print(f"  ✓ pass:             {stats['pass']}")
    print(f"  ✗ fail:             {fail}")
    print(f"  ⏱ timeout:          {stats['timeout']}")
    print(f"  · runtime_error:    {stats['runtime_error']}")
    print(f"  · wrong_output:     {stats['wrong_output']}")
    print(f"  · no_gold:          {stats['no_gold']}")
    print(f"  · no_tests:         {stats['no_tests']}")
    print(f"  · unknown:          {stats['unknown']}")
    print(f"  Saved → {out_path}  ({mb:.1f} MB, {elapsed:.1f}s)")

    # Write failure log
    log_path = LOCAL_DIR / f"apps_{split}_download_failures.log"
    with open(log_path, "w") as lf:
        lf.write(f"split={split} total={total} pass={stats['pass']} fail={fail} timeout={stats['timeout']}\n")
        lf.write(f"runtime_error={stats['runtime_error']} wrong_output={stats['wrong_output']} no_gold={stats['no_gold']} no_tests={stats['no_tests']} unknown={stats['unknown']}\n\n")
        for reason in ("timeout", "runtime_error", "wrong_output", "unknown"):
            lf.write(f"[{reason}]\n")
            for s in failure_samples.get(reason, []):
                lf.write(f"  {s}\n")
            lf.write("\n")
    print(f"  Failure log → {log_path}")


def main():
    default_workers = min(64, (os.cpu_count() or 8) * 2)
    p = argparse.ArgumentParser(description="Download & filter APPS dataset")
    p.add_argument("--split", default="all", help="'train', 'test', or 'all'")
    p.add_argument("--workers", type=int, default=default_workers, help=f"Concurrent workers (default {default_workers})")
    p.add_argument("--timeout", type=float, default=30.0, help="Per-problem timeout (default 30s)")
    args = p.parse_args()

    splits = list(_ALL_SPLITS) if args.split == "all" else [args.split]
    for split in splits:
        _download_split(split, workers=args.workers, timeout=args.timeout)

    print(f"\n{'='*60}")
    print("All done ✓")


if __name__ == "__main__":
    main()
