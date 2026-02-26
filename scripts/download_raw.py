#!/usr/bin/env python3
"""Download raw APPS splits from HuggingFace — NO gold-checking, just save the JSONL.

This is fast (just a download + normalize keys). Use filter_data.py afterwards
to keep only problems whose gold solution passes.

Produces:
    data/apps_train.jsonl
    data/apps_test.jsonl

Usage:
    python scripts/download_raw.py                # both splits
    python scripts/download_raw.py --split train  # single split
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
_ALL_SPLITS = ("train", "test")


def download_split(split: str) -> None:
    from huggingface_hub import hf_hub_download

    print(f"\n{'='*60}")
    print(f"Downloading codeparrot/apps  split={split} …")

    path = hf_hub_download(
        repo_id="codeparrot/apps",
        filename=f"{split}.jsonl",
        repo_type="dataset",
    )

    # Read raw rows
    raw_rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            raw_rows.append(json.loads(line))

    print(f"  {len(raw_rows)} problems downloaded")

    # Normalize keys: HF test split uses 'id' instead of 'problem_id'
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"apps_{split}.jsonl"

    with open(out_path, "w") as f:
        for i, row in enumerate(raw_rows):
            # Normalize to a consistent schema
            out = {
                "problem_id": str(row.get("problem_id", row.get("id", i))),
                "question": row.get("question", ""),
                "solutions": row.get("solutions", "[]"),
                "input_output": row.get("input_output", "{}"),
                "difficulty": row.get("difficulty", ""),
                "starter_code": row.get("starter_code", ""),
            }
            f.write(json.dumps(out) + "\n")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Saved {len(raw_rows)} rows → {out_path}  ({size_mb:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description="Download raw APPS data from HuggingFace")
    p.add_argument("--split", default="all", help="'train', 'test', or 'all' (default: all)")
    args = p.parse_args()

    splits = list(_ALL_SPLITS) if args.split == "all" else [args.split]
    for s in splits:
        download_split(s)

    print(f"\n{'='*60}")
    print("Raw download complete ✓")
    print("Next: python scripts/filter_data.py")


if __name__ == "__main__":
    main()
