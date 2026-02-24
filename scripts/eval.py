#!/usr/bin/env python3
"""Standalone evaluation: load a run log and compute summary metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from seca.eval.metrics import forgetting


def main():
    parser = argparse.ArgumentParser(description="Analyse a SECA run log")
    parser.add_argument("log", type=str, help="Path to run_log.jsonl")
    args = parser.parse_args()

    entries = [json.loads(l) for l in Path(args.log).read_text().splitlines() if l.strip()]

    # ── per-type performance ──
    by_type: dict[str, list[float]] = {}
    for e in entries:
        t = e.get("etype", e.get("baseline", "?"))
        by_type.setdefault(t, []).append(e["new_perf"])

    print("── New-episode performance ──")
    for t, scores in by_type.items():
        print(f"  {t:12s}  mean={sum(scores)/len(scores):.4f}  n={len(scores)}")

    # ── retention ──
    anchor_scores = [e["anchor_perf"] for e in entries if "anchor_perf" in e]
    f = forgetting(anchor_scores)
    print(f"\n── Retention ──")
    print(f"  max_drop = {f['max_drop']:.4f}")
    print(f"  mean_auc = {f['auc']:.4f}")

    # ── routing stats (SECA / heuristic only) ──
    actions = [e.get("action") for e in entries if "action" in e]
    if actions:
        n_up = sum(1 for a in actions if a == "UPDATE")
        n_st = sum(1 for a in actions if a == "STORE")
        print(f"\n── Routing ──")
        print(f"  UPDATE = {n_up}  STORE = {n_st}  total = {len(actions)}")


if __name__ == "__main__":
    main()
