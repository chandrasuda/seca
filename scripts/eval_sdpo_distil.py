#!/usr/bin/env python3
"""Evaluate an SDPO checkpoint (HF format) on APPS (Pass@k).

Both train_sdpo_distil.py (SDPO) and train_sdft_distil.py (SDFT) save
checkpoints in standard HuggingFace format, so this eval script works for
either.  It loads the model via vLLM for fast batch generation, then runs
the APPS sandbox executor to compute Pass@k.

Usage:
  # Evaluate latest checkpoint directory
  python scripts/eval_sdpo_distil.py \
      --checkpoint checkpoints/sdpo_distil

  # Evaluate a specific step checkpoint
  python scripts/eval_sdpo_distil.py \
      --checkpoint checkpoints/sdpo_distil/checkpoint-400 \
      --n-samples 20 --max-problems 200

  # Quick smoke test
  python scripts/eval_sdpo_distil.py \
      --checkpoint checkpoints/sdpo_distil/checkpoint-100 \
      --max-problems 10 --n-samples 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from seca.data.apps import load_apps
from seca.sandbox.executor import execute_code
from seca.eval.metrics import pass_at_k
from seca.utils.tokenizer import make_no_thinking_tokenizer

log = logging.getLogger(__name__)


def _find_best_checkpoint(checkpoint_dir: str) -> str:
    """If checkpoint_dir has sub-dirs like checkpoint-NNN, return the latest."""
    p = Path(checkpoint_dir)
    if (p / "config.json").exists():
        return str(p)  # already a leaf checkpoint
    candidates = sorted(
        [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if not candidates:
        return str(p)  # fall back (vLLM will report error if invalid)
    latest = candidates[-1]
    log.info("Auto-selected latest checkpoint: %s", latest)
    return str(latest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate SDPO checkpoint (HF format) on APPS Pass@k"
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to HF checkpoint directory (train_sdpo_distil.py output)",
    )
    # Data
    p.add_argument("--split", default="test",
                   help="APPS split: train | test | all (default: test)")
    p.add_argument("--difficulty", default=None,
                   help="APPS difficulty filter: introductory | interview | competition")
    p.add_argument("--max-problems", type=int, default=None,
                   help="Cap on number of problems (None = full split)")
    # Sampling
    p.add_argument("--n-samples", type=int, default=10,
                   help="Completions per problem for Pass@k estimation")
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Sampling temperature (paper uses 0.6 for eval)")
    p.add_argument("--top-p", type=float, default=0.95,
                   help="Nucleus sampling top-p")
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="Max tokens per completion during eval")
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10],
                   help="k values for Pass@k (default: 1 5 10)")
    # Infra
    p.add_argument("--gpu-memory-util", type=float, default=0.9,
                   help="vLLM GPU memory fraction (high OK: no training during eval)")
    p.add_argument("--max-model-len", type=int, default=4096,
                   help="vLLM max context length")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="Sandbox execution timeout per test case (seconds)")
    # Output
    p.add_argument("--output-dir", default="logs/",
                   help="Directory to save results JSON")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    checkpoint_path = _find_best_checkpoint(args.checkpoint)
    log.info("Loading checkpoint via vLLM: %s", checkpoint_path)

    llm = LLM(
        model=checkpoint_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_util,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    make_no_thinking_tokenizer(tokenizer)  # disable <think> for Qwen3

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    log.info(
        "Loading APPS (split=%s difficulty=%s max=%s)...",
        args.split, args.difficulty, args.max_problems,
    )
    problems = load_apps(
        split=args.split,
        difficulty=args.difficulty,
        max_problems=args.max_problems,
    )
    log.info("Evaluating on %d problems", len(problems))

    all_pass_at: dict[int, list[float]] = {k: [] for k in args.k_values}
    total_pass_rate = 0.0

    for i, problem in enumerate(problems):
        prompt_text = problem.format_prompt()

        # Apply chat template (matches training-time formatting)
        messages = [{"role": "user", "content": prompt_text}]
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            formatted = prompt_text

        prompts = [formatted] * args.n_samples
        outputs = llm.generate(prompts, sampling_params)
        completions = [out.outputs[0].text for out in outputs]

        # Execute and score
        n_correct = 0
        pass_rates: list[float] = []
        for code in completions:
            fb = execute_code(code, problem, timeout=args.timeout)
            if fb.all_passed:
                n_correct += 1
            pass_rates.append(fb.pass_rate)

        total_pass_rate += sum(pass_rates) / max(len(pass_rates), 1)

        for k in args.k_values:
            all_pass_at[k].append(
                pass_at_k(args.n_samples, n_correct, min(k, args.n_samples))
            )

        if (i + 1) % 10 == 0 or (i + 1) == len(problems):
            p1 = sum(all_pass_at[1]) / max(len(all_pass_at[1]), 1)
            log.info(
                "Progress %d/%d | running pass@1=%.3f | n_correct_this=%d/%d",
                i + 1, len(problems), p1, n_correct, args.n_samples,
            )

    n = len(problems)
    results: dict = {
        f"pass@{k}": sum(v) / n if n else 0.0 for k, v in all_pass_at.items()
    }
    results["mean_test_pass_rate"] = total_pass_rate / n if n else 0.0
    results["n_problems"] = n
    results["checkpoint"] = checkpoint_path
    results["split"] = args.split
    results["difficulty"] = args.difficulty
    results["n_samples"] = args.n_samples
    results["temperature"] = args.temperature

    print("\n══ SDPO Evaluation Results ══")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key:30s}  {val:.4f}")
        else:
            print(f"  {key:30s}  {val}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(checkpoint_path).name
    out_file = out_dir / f"eval_sdpo_{ckpt_name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved → %s", out_file)


if __name__ == "__main__":
    main()
