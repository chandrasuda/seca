#!/usr/bin/env python3
"""Validate dataset quality and inspect execution feedback.

Run this BEFORE training to catch data issues early.
Checks:
  1. How many problems have gold solutions?
  2. How many have test cases?
  3. Do gold solutions actually pass their own tests?
  4. What does execution feedback look like for the base model?
"""
from __future__ import annotations

import argparse
import logging
import random

import yaml

from seca.data.loader import load_problems
from seca.sandbox.executor import execute_code


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset quality")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n", type=int, default=20, help="Problems to inspect")
    parser.add_argument("--check-gold", action="store_true", help="Run gold solutions against tests")
    parser.add_argument("--show-feedback", action="store_true", help="Show sample feedback strings")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    problems = load_problems(cfg["data"])
    n = min(args.n, len(problems))

    # ── basic stats ──
    has_gold = sum(1 for p in problems if p.gold_solution.strip())
    has_tests = sum(1 for p in problems if p.test_cases)
    has_both = sum(1 for p in problems if p.gold_solution.strip() and p.test_cases)

    print(f"\n══ Dataset: {cfg['data']['dataset']} ══")
    print(f"  Total problems:      {len(problems)}")
    print(f"  With gold solution:  {has_gold}")
    print(f"  With test cases:     {has_tests}")
    print(f"  With BOTH (usable):  {has_both}")
    print(f"  Missing gold:        {len(problems) - has_gold}")
    print(f"  Missing tests:       {len(problems) - has_tests}")

    if not has_both:
        print("\n⚠  No problems have both gold solutions and test cases!")
        return

    # ── check gold solutions ──
    if args.check_gold:
        usable = [p for p in problems if p.gold_solution.strip() and p.test_cases]
        sample = random.sample(usable, min(n, len(usable)))
        gold_pass = 0
        gold_fail = 0

        print(f"\n── Checking {len(sample)} gold solutions against tests ──")
        for p in sample:
            fb = execute_code(p.gold_solution, p)
            if fb.all_passed:
                gold_pass += 1
            else:
                gold_fail += 1
                print(f"  ✗ {p.problem_id}: gold FAILS its own tests")
                print(f"    {fb.summary[:200]}")

        print(f"\n  Gold pass rate: {gold_pass}/{len(sample)}")
        if gold_fail:
            print(f"  ⚠  {gold_fail} gold solutions fail — filter these before training!")

    # ── show sample feedback ──
    if args.show_feedback:
        usable = [p for p in problems if p.test_cases]
        sample = random.sample(usable, min(3, len(usable)))

        print(f"\n── Sample execution feedback (running bad code) ──")
        for p in sample:
            bad_code = "print('wrong answer')"
            fb = execute_code(bad_code, p)
            print(f"\n  Problem: {p.problem_id}")
            print(f"  Feedback ({fb.pass_rate*100:.0f}% pass):")
            for line in fb.summary.split("\n")[:8]:
                print(f"    {line}")


if __name__ == "__main__":
    main()
