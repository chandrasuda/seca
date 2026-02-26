#!/usr/bin/env python3
"""Augment apps-format LeetCode JSONL with original harness fields.

Input:
  - leetcode_train_apps_fmt.jsonl / leetcode_test_apps_fmt.jsonl
  - leetcode_train.jsonl / leetcode_test.jsonl

Output:
  - leetcode_train_apps_harness_fmt.jsonl
  - leetcode_test_apps_harness_fmt.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    # Keep ensure_ascii=True so U+2028/U+2029 are escaped and never split rows.
    path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n")


def augment(apps_fmt_path: Path, raw_path: Path, out_path: Path) -> tuple[int, int]:
    apps_rows = _read_jsonl(apps_fmt_path)
    raw_rows = _read_jsonl(raw_path)
    raw_by_qid = {str(r.get("question_id")): r for r in raw_rows}

    attached = 0
    out_rows: list[dict] = []
    for row in apps_rows:
        qid = str(row.get("id", row.get("problem_id", "")))
        raw = raw_by_qid.get(qid)
        if raw:
            row["entry_point"] = raw.get("entry_point", "")
            row["test"] = raw.get("test", "")
            row["task_id"] = raw.get("task_id", "")
            attached += 1
        out_rows.append(row)

    _write_jsonl(out_path, out_rows)
    return len(out_rows), attached


def main() -> None:
    ap = argparse.ArgumentParser(description="Add harness fields to apps-format LeetCode JSONL")
    ap.add_argument("--leetcode-dir", default="leetcode")
    args = ap.parse_args()

    d = Path(args.leetcode_dir)
    train_n, train_attached = augment(
        d / "leetcode_train_apps_fmt.jsonl",
        d / "leetcode_train.jsonl",
        d / "leetcode_train_apps_harness_fmt.jsonl",
    )
    test_n, test_attached = augment(
        d / "leetcode_test_apps_fmt.jsonl",
        d / "leetcode_test.jsonl",
        d / "leetcode_test_apps_harness_fmt.jsonl",
    )
    print(
        f"train: {train_n} rows ({train_attached} with harness), "
        f"test: {test_n} rows ({test_attached} with harness)"
    )


if __name__ == "__main__":
    main()
