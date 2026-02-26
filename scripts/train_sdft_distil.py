#!/usr/bin/env python3
"""SDFT training on APPS using the Self-Distillation submodule (Shenfeld et al., 2026).

Faithfully implements Algorithm 1 from arxiv:2601.19897:
  - Student generates on-policy from the problem prompt only
  - Teacher = EMA-updated copy of student, conditioned on prompt + gold solution
  - Loss = KL(student || teacher) via analytic per-token estimator
  - EMA: φ ← alpha * θ + (1 - alpha) * φ  (alpha=0.01, every step)

Dataset: APPS (codeparrot/apps) — filtered local copy in data_filtered/
Model:   Defaults to Qwen/Qwen3-1.7B (fits A100 40 GB with vLLM sleep mode)

Usage (local smoke test):
  python scripts/train_sdft_distil.py --max-problems 5 --epochs 1

Usage (Modal A100):
  modal run scripts/modal_distil_sdft.py --task training --max-problems 100 --epochs 1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the Self-Distillation submodule importable (distil_trainer, distil_config)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SD_PATH = REPO_ROOT / "Self-Distillation"
if str(SD_PATH) not in sys.path:
    sys.path.insert(0, str(SD_PATH))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from distil_config import DistilConfig
from distil_trainer import DistilTrainer
from seca.data.apps import load_apps
from seca.utils.tokenizer import make_no_thinking_tokenizer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Teacher prompt template — verbatim from Shenfeld et al. 2026, Section 3
# (main.py in the submodule uses the same structure for tool-use tasks)
# ---------------------------------------------------------------------------
_TEACHER_TEMPLATE = """{question}

This is an example for a response to the question:
<start_code>
{gold_solution}
<end_code>

Now answer with a complete Python solution of your own. Your response must contain ONLY executable code enclosed between <start_code> and <end_code> tokens."""


def build_hf_dataset(problems, seed: int = 42) -> Dataset:
    """Convert APPS CodeProblem list → HF Dataset with prompt / teacher_prompt columns.

    prompt         Student context: problem statement only (on-policy rollout input)
    teacher_prompt Teacher context: problem + gold solution (demonstration-conditioned)

    Both are stored in conversational format (list[dict]) so DistilTrainer can
    apply the model's chat template automatically via maybe_apply_chat_template.
    """
    rows = []
    for p in problems:
        student_content = p.format_prompt()
        teacher_content = _TEACHER_TEMPLATE.format(
            question=student_content,
            gold_solution=p.gold_solution.strip(),
        )
        rows.append(
            {
                "prompt": [{"role": "user", "content": student_content}],
                "teacher_prompt": [{"role": "user", "content": teacher_content}],
            }
        )

    dataset = Dataset.from_list(rows)
    dataset = dataset.shuffle(seed=seed)
    log.info(
        "Dataset ready: %d examples (prompt + teacher_prompt columns)", len(dataset)
    )
    return dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDFT training via the Self-Distillation submodule on APPS"
    )
    # Model
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="HF model name / local path (default: Qwen/Qwen3-1.7B)",
    )
    # Data
    p.add_argument(
        "--split",
        default="train",
        help="APPS split: train | test | all (default: train)",
    )
    p.add_argument(
        "--difficulty",
        default=None,
        help="APPS difficulty filter: introductory | interview | competition",
    )
    p.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Cap on number of APPS problems to use (None = all with gold + tests)",
    )
    # Training (matched to SDFT paper's skill-learning settings)
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs (paper uses 1-2 for skill learning)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (paper sweeps {5e-6, 1e-5, 5e-5}; 1e-5 is paper default)",
    )
    p.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Rollouts per prompt G (paper uses 8; more diversity = better signal)",
    )
    p.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Max prompt tokens (default: None = no truncation)",
    )
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=2048,
        help="Max completion tokens (paper: 2048 for skill learning)",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.01,
        help="EMA teacher mixup coefficient α (paper: 0.01)",
    )
    # vLLM
    p.add_argument(
        "--vllm-gpu-memory",
        type=float,
        default=0.35,
        help="vLLM GPU memory fraction (default 0.35; sleep mode frees memory during backward)",
    )
    # Output
    p.add_argument(
        "--output-dir",
        default="checkpoints/sdft_distil",
        help="Output directory for HF-format checkpoints",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B even if WANDB_API_KEY is set",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    # ------------------------------------------------------------------
    # Load and format APPS data
    # ------------------------------------------------------------------
    log.info(
        "Loading APPS (split=%s difficulty=%s max=%s)...",
        args.split,
        args.difficulty,
        args.max_problems,
    )
    problems = load_apps(
        split=args.split,
        difficulty=args.difficulty,
        max_problems=args.max_problems,
    )
    log.info("Loaded %d APPS problems with gold solutions + test cases", len(problems))

    dataset = build_hf_dataset(problems, seed=args.seed)

    # ------------------------------------------------------------------
    # Load model + teacher (identical initialisation; EMA diverges over training)
    # ------------------------------------------------------------------
    log.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    log.info("Loading teacher (ref_model, same weights): %s", args.model)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    make_no_thinking_tokenizer(tokenizer)  # disable <think> for Qwen3

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    has_wandb_key = bool(os.environ.get("WANDB_API_KEY", "").strip())
    report_to = "none" if (args.no_wandb or not has_wandb_key) else "wandb"
    if report_to == "none":
        log.info("W&B disabled (set WANDB_API_KEY to enable)")

    # ------------------------------------------------------------------
    # DistilConfig — matched to paper's skill-learning hyperparameters
    # ------------------------------------------------------------------
    # generation_batch_size = per_device_batch_size * num_processes * steps_per_generation
    # = 1 * 1 * num_generations = num_generations
    # generation_batch_size % num_generations = 0  ✓
    # → 1 unique prompt per generation batch, G completions per prompt
    config = DistilConfig(
        # ----- Identity / output -----
        output_dir=args.output_dir,
        run_name="sdft-apps",
        report_to=report_to,
        # ----- Precision -----
        bf16=True,
        fp16=False,
        # ----- LR schedule (paper: cosine with 10-step warmup) -----
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        # ----- Batch / gradient accumulation -----
        # gradient_accumulation_steps == num_generations so that one optimizer
        # step consumes all G completions from one prompt.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_generations,
        # ----- Generation -----
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=1.0,  # paper samples with T=1.0
        top_p=1.0,
        # ----- Training objective -----
        # alpha=1.0 → reverse KL = KL(student || teacher) as in the paper:
        #   ℒ(θ) = D_KL(π_θ(·|x) ∥ π_φ(·|x,c))
        alpha=1.0,
        # beta=0.0 → no additional KL penalty against a frozen reference
        beta=0.0,
        # ----- EMA teacher (paper: φ ← α·θ + (1-α)·φ, α=0.01, every step) -----
        sync_ref_model=True,
        ref_model_mixup_alpha=args.ema_alpha,  # α=0.01
        ref_model_sync_steps=1,               # update every gradient step
        # ----- vLLM colocated for on-policy generation -----
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory,
        vllm_enable_sleep_mode=True,          # frees vLLM memory during backward
        vllm_importance_sampling_correction=True,
        # ----- Stability -----
        # skip first 3 completion tokens from loss (reduces artifact copying)
        num_loss_tokens_to_skip=3,
        mask_truncated_completions=False,     # keep loss signal when many completions hit max_tokens
        # ----- Epochs / checkpointing -----
        num_train_epochs=args.epochs,
        save_steps=75,
        save_total_limit=3,
        logging_steps=1,
        seed=args.seed,
        # ----- Misc -----
        remove_unused_columns=False,
        disable_dropout=False,
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    total_steps = len(dataset) * args.epochs
    log.info(
        "Starting SDFT training: %d problems | %d epochs | G=%d rollouts | LR=%.1e | save_every=%d steps",
        len(problems),
        args.epochs,
        args.num_generations,
        args.lr,
        config.save_steps,
    )
    log.info("Total steps: %d | Checkpoints: %s", total_steps, args.output_dir)

    output = trainer.train()

    log.info(
        "Training complete. Runtime=%.1fs | samples/s=%.2f | final_loss=%.4f",
        output.metrics.get("train_runtime", 0),
        output.metrics.get("train_samples_per_second", 0),
        output.metrics.get("train_loss", 0),
    )
    log.info("Checkpoint saved → %s", args.output_dir)


if __name__ == "__main__":
    main()
