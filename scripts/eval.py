#!/usr/bin/env python3
"""Evaluate a checkpoint on code-generation problems (Pass@k).

Uses vLLM for fast generation. Supports Qwen3-1.7B and other HF causal LMs.

Usage:
  python scripts/eval.py --checkpoint checkpoints/omni_epoch2.pt
  python scripts/eval.py --model Qwen/Qwen3-1.7B --n-samples 20
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from vllm import SamplingParams

from seca.data.loader import load_problems
from seca.eval.metrics import evaluate_problems
from seca.models.base import BaseModel


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
    p = argparse.ArgumentParser(description="Evaluate checkpoint on code generation")
    p.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    p.add_argument("--checkpoint", default=None, help="Checkpoint path (.pt)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--model", type=str, default=None, help="Model name override")
    p.add_argument("--gpu-memory-util", type=float, default=None, help="vLLM GPU memory fraction")
    p.add_argument("--n-samples", type=int, default=None, help="Samples per problem for Pass@k")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    p.add_argument("--dataset", type=str, default=None, help="Dataset name")
    p.add_argument("--max-problems", type=int, default=None, help="Max problems to evaluate")
    p.add_argument("--log-dir", type=str, default=None, help="Log output dir")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    overrides: dict = {}
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.model is not None:
        overrides.setdefault("model", {})["name"] = args.model
    if args.gpu_memory_util is not None:
        overrides.setdefault("vllm", {})["gpu_memory_utilization"] = args.gpu_memory_util
    if args.dataset is not None:
        overrides.setdefault("data", {})["dataset"] = args.dataset
    if args.max_problems is not None:
        overrides.setdefault("data", {})["max_problems"] = args.max_problems
    if args.log_dir is not None:
        overrides.setdefault("eval", {})["log_dir"] = args.log_dir

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

    model_cfg = dict(cfg["model"])
    model_cfg["vllm_cfg"] = cfg.get("vllm", {})
    model = BaseModel(**model_cfg)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            model.model.load_state_dict(state, strict=False)
            log.info("Loaded checkpoint: %s", ckpt_path)
        else:
            log.warning("Checkpoint not found: %s", ckpt_path)

    problems = load_problems(cfg["data"])
    log.info("Evaluating on %d problems (dataset=%s)", len(problems), cfg["data"]["dataset"])

    ec = cfg["eval"]
    inf_cfg = cfg.get("inference", {})
    n_samples = args.n_samples if args.n_samples is not None else ec.get("n_samples", 10)
    temperature = args.temperature if args.temperature is not None else ec.get("temperature", 0.6)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=ec.get("top_p", inf_cfg.get("top_p", 0.95)),
        max_tokens=inf_cfg.get("max_new_tokens", 512),
    )

    results = evaluate_problems(
        model,
        problems,
        n_samples=n_samples,
        k_values=ec.get("k_values", [1, 5, 10]),
        sampling_params=sampling_params,
    )

    print("\n══ Results ══")
    for k, v in results.items():
        print(f"  {k:25s}  {v:.4f}" if isinstance(v, float) else f"  {k:25s}  {v}")

    log_dir = Path(ec["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved → %s", out)


if __name__ == "__main__":
    main()
