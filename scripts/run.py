#!/usr/bin/env python3
"""Entry point: run SECA or a baseline on a continual stream."""
from __future__ import annotations

import argparse
import logging
import random

import yaml
import torch
import numpy as np

from seca.data.episode import build_stream
from seca.data.skills import load_skill_episodes
from seca.data.knowledge import load_knowledge_episodes
from seca.train.continual import SECALoop
from seca.train.baselines import run_baseline
from seca.models.base import BaseModel


def main():
    parser = argparse.ArgumentParser(description="SECA runner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["seca", "sft_all", "sdft_all", "rag_all", "heuristic"],
                        default="seca")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed

    _set_seed(cfg["seed"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    # ── load data ──
    skills = load_skill_episodes(cfg["data"]["skills_path"],
                                 max_episodes=cfg["data"]["stream_length"])
    knowledge = load_knowledge_episodes(cfg["data"]["knowledge_path"],
                                        max_episodes=cfg["data"]["stream_length"])
    stream = build_stream(skills, knowledge,
                          length=cfg["data"]["stream_length"],
                          skill_ratio=cfg["data"]["skill_ratio"],
                          seed=cfg["seed"])

    # anchor suite: first few of each type
    anchor = skills[:10] + knowledge[:10]

    # ── run ──
    if args.mode == "seca":
        loop = SECALoop(cfg)
        loop.set_anchor_suite(anchor)
        history = loop.run(stream)
    else:
        model = BaseModel(**cfg["model"])
        history = run_baseline(args.mode, model, stream, anchor, cfg)

    print(f"\n✓ Completed {len(history)} episodes.  Logs → {cfg['eval']['log_dir']}")


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
