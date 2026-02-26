#!/usr/bin/env python3
"""Ablation sweep over Omni-Teacher α/β/γ → logs/ablation_results.csv."""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from vllm import SamplingParams

from seca.data.loader import load_problems
from seca.eval.metrics import evaluate_problems
from seca.train.trainer import Trainer

ALPHAS = [0.2, 0.4, 0.6]
BETAS = [0.2, 0.4, 0.6]
GAMMAS = [0.1, 0.2, 0.3]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    with open(args.config) as f:
        base = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    seed = base.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    problems = load_problems(base["data"])
    eval_problems = problems[: min(50, len(problems))]
    inf_cfg = base.get("inference", {})
    ec = base.get("eval", {})
    sampling_params = SamplingParams(
        temperature=ec.get("temperature", 0.6),
        top_p=ec.get("top_p", inf_cfg.get("top_p", 0.95)),
        max_tokens=inf_cfg.get("max_new_tokens", 512),
    )

    results = []
    total = len(ALPHAS) * len(BETAS) * len(GAMMAS)
    for i, (a, b, g) in enumerate(itertools.product(ALPHAS, BETAS, GAMMAS)):
        logging.info("Ablation %d/%d: α=%.2f β=%.2f γ=%.2f", i + 1, total, a, b, g)
        cfg = copy.deepcopy(base)
        cfg["training"]["mode"] = "omni"
        cfg["omni"]["alpha_sdft"] = a
        cfg["omni"]["beta_sdpo"] = b
        cfg["omni"]["gamma_nll"] = g
        cfg["eval"]["log_dir"] = f"logs/ablation/a{a}_b{b}_g{g}/"
        trainer = Trainer(cfg)
        hist = trainer.train(problems, eval_problems=eval_problems)
        ev = evaluate_problems(
            trainer.model,
            eval_problems,
            n_samples=5,
            k_values=[1, 5],
            sampling_params=sampling_params,
        )
        results.append({
            "alpha": a,
            "beta": b,
            "gamma": g,
            **ev,
            "final_loss": hist[-1]["loss"] if hist else 0.0,
        })

    out = Path("logs/ablation_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
