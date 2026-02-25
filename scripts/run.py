#!/usr/bin/env python3
"""Train Omni-Teacher (or baseline) on code-generation problems."""
from __future__ import annotations
import argparse, logging, random
import yaml, torch, numpy as np
from seca.data.loader import load_problems
from seca.train.trainer import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--mode", choices=["sft", "sdft", "sdpo", "omni", "grpo"], default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None: cfg["seed"] = args.seed
    if args.mode is not None: cfg["training"]["mode"] = args.mode
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    problems = load_problems(cfg["data"])
    logging.info(f"Loaded {len(problems)} problems from {cfg['data']['dataset']}")
    Trainer(cfg).train(problems)
    print(f"\n✓ Done ({cfg['training']['mode']}).  Logs → {cfg['eval']['log_dir']}")


if __name__ == "__main__":
    main()
