#!/usr/bin/env python3
"""Evaluate a checkpoint on code-generation problems (Pass@k)."""
from __future__ import annotations
import argparse, json, logging, random
from pathlib import Path
import yaml, torch, numpy as np
from seca.models.base import BaseModel
from seca.eval.metrics import evaluate_problems
from seca.data.loader import load_problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed: cfg["seed"] = args.seed
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    model = BaseModel(**cfg["model"])
    if args.checkpoint:
        model.model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    problems = load_problems(cfg["data"])
    ec = cfg["eval"]
    results = evaluate_problems(model, problems, n_samples=ec.get("n_samples", 10),
                                k_values=ec.get("k_values", [1, 5, 10]),
                                temperature=ec.get("temperature", 0.8))

    print("\n══ Results ══")
    for k, v in results.items():
        print(f"  {k:25s}  {v:.4f}" if isinstance(v, float) else f"  {k:25s}  {v}")
    out = Path(ec["log_dir"]) / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, out.open("w"), indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
