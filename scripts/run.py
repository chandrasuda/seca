#!/usr/bin/env python3
"""Train Omni-Teacher (or baseline) on code-generation problems.

Uses vLLM for inference and HuggingFace for training. Target: Qwen3-1.7B.

Usage:
  python scripts/run.py --mode omni
  python scripts/run.py --mode sdpo --model Qwen/Qwen3-1.7B --epochs 5
  python scripts/run.py --resume checkpoints/omni_epoch2.pt
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

# Must be set before vLLM is imported so EngineCore runs in-process,
# which is required for sync_vllm_weights to reach the model runner directly.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np
import torch
import yaml

from seca.data.loader import load_problems
from seca.train.trainer import Trainer


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base."""
    out = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train Omni-Teacher (SDFT + SDPO) on code generation"
    )
    p.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    p.add_argument(
        "--mode",
        choices=["sft", "sdft", "sdpo", "omni", "grpo"],
        default=None,
        help="Training mode",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    # Model
    p.add_argument("--model", type=str, default=None, help="Model name (e.g. Qwen/Qwen3-1.7B)")
    # Training
    p.add_argument("--lr", type=float, default=None, help="Learning rate")
    p.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    p.add_argument("--num-samples", type=int, default=None, help="Rollouts per problem (G)")
    p.add_argument("--mini-batch-size", type=int, default=None, help="Gradient accumulation chunk size")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint output dir")
    p.add_argument("--log-dir", type=str, default=None, help="Log output dir")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    # vLLM / inference
    p.add_argument(
        "--gpu-memory-util",
        type=float,
        default=None,
        help="vLLM GPU memory fraction (0.3–0.5)",
    )
    p.add_argument("--max-new-tokens", type=int, default=None, help="Max tokens per rollout")
    # Data
    p.add_argument("--dataset", type=str, default=None, help="apps | livecodebench | kernelbench")
    p.add_argument("--max-problems", type=int, default=None, help="Max problems to load")
    p.add_argument("--difficulty", type=str, default=None, help="APPS difficulty")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    overrides: dict = {}
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.mode is not None:
        cfg["training"]["mode"] = args.mode
    if args.model is not None:
        overrides.setdefault("model", {})["name"] = args.model
    if args.lr is not None:
        overrides.setdefault("training", {})["lr"] = args.lr
    if args.epochs is not None:
        overrides.setdefault("training", {})["num_epochs"] = args.epochs
    if args.num_samples is not None:
        overrides.setdefault("training", {})["num_samples"] = args.num_samples
    if args.mini_batch_size is not None:
        overrides.setdefault("training", {})["mini_batch_size"] = args.mini_batch_size
    if args.checkpoint_dir is not None:
        overrides.setdefault("training", {})["checkpoint_dir"] = args.checkpoint_dir
    if args.log_dir is not None:
        overrides.setdefault("eval", {})["log_dir"] = args.log_dir
    if args.gpu_memory_util is not None:
        overrides.setdefault("vllm", {})["gpu_memory_utilization"] = args.gpu_memory_util
    if args.max_new_tokens is not None:
        overrides.setdefault("inference", {})["max_new_tokens"] = args.max_new_tokens
    if args.dataset is not None:
        overrides.setdefault("data", {})["dataset"] = args.dataset
    if args.max_problems is not None:
        overrides.setdefault("data", {})["max_problems"] = args.max_problems
    if args.difficulty is not None:
        overrides.setdefault("data", {})["difficulty"] = args.difficulty

    cfg = _deep_update(cfg, overrides)

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    log = logging.getLogger(__name__)

    problems = load_problems(cfg["data"])
    log.info(
        "Loaded %d problems from %s (dataset=%s)",
        len(problems),
        cfg["data"].get("dataset", "?"),
        cfg["data"].get("max_problems", "all"),
    )
    vllm_cfg = cfg.get("vllm", {})
    inf_cfg = cfg.get("inference", {})
    # PyYAML parses `5e-7` (no decimal point) as a string. Normalise to float here
    # and write back so Trainer always receives a proper numeric value.
    lr_val = float(cfg["training"]["lr"])
    cfg["training"]["lr"] = lr_val
    log.info(
        "Training: mode=%s model=%s lr=%.0e epochs=%d K=%d mini_batch=%d",
        cfg["training"]["mode"],
        cfg["model"]["name"],
        lr_val,
        cfg["training"]["num_epochs"],
        cfg["training"]["num_samples"],
        cfg["training"].get("mini_batch_size", 4),
    )
    log.info(
        "vLLM: gpu_mem=%.2f  inference: temp=%.2f top_p=%.2f max_tokens=%d",
        vllm_cfg.get("gpu_memory_utilization", 0.4),
        inf_cfg.get("temperature", 1.0),
        inf_cfg.get("top_p", 0.95),
        inf_cfg.get("max_new_tokens", 512),
    )

    trainer = Trainer(cfg)

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            trainer.model.model.load_state_dict(state, strict=False)
            log.info("Resumed from %s", ckpt_path)
        else:
            log.warning("Resume path not found: %s", ckpt_path)

    trainer.train(problems)
    log.info("Done. Logs → %s  Checkpoints → %s", cfg["eval"]["log_dir"], cfg["training"]["checkpoint_dir"])


if __name__ == "__main__":
    main()
