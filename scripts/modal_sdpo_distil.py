#!/usr/bin/env python3
"""Modal runner for SDPO training via standalone train_sdpo_distil.py.

Trains on APPS using the SDPO methodology from Shenfeld et al. 2026
(arxiv:2601.20802) — faithful standalone implementation without the verl
distributed framework.

Key differences from modal_distil_sdft.py (SDFT runner):
  - Uses train_sdpo_distil.py (on-policy rollouts + sandbox execution + KL distil)
  - Teacher = trust-region(ref_model, student_model, mix_coef=0.01) by default
  - Per-rollout teacher reprompts built from passing solutions in the rollout group
  - No TRL / DistilTrainer dependency; pure PyTorch training loop
  - Checkpoints saved to seca-sdpo-distil-outputs volume (separate from SDFT)

Usage:
  # Smoke test (5 problems, 1 epoch)
  modal run scripts/modal_sdpo_distil.py --task training --max-problems 5 --epochs 1

  # Full training run (all train-split problems, 1 epoch)
  modal run scripts/modal_sdpo_distil.py --task training --epochs 1

  # Training with difficulty filter
  modal run scripts/modal_sdpo_distil.py --task training \\
      --difficulty introductory --max-problems 500 --epochs 2

  # Evaluate a checkpoint saved to the volume
  modal run scripts/modal_sdpo_distil.py --task eval \\
      --checkpoint checkpoints/sdpo_distil/checkpoint-400

  # List saved checkpoints / logs
  modal run scripts/modal_sdpo_distil.py --task list-outputs
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Modal image
# Pure PyTorch training loop — no TRL or DistilTrainer required.
# vLLM is only needed for evaluation (eval_sdpo_distil.py).
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # PyTorch with CUDA 12.4 — must precede vLLM install
    .run_commands(
        "pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu124"
    )
    # Core training stack
    .pip_install(
        "transformers>=4.47,<4.60",
        "accelerate>=1.2",
        "datasets>=2.18",
        "peft>=0.14",
    )
    # vLLM — needed only for eval_sdpo_distil.py (fast batch generation)
    .pip_install("vllm>=0.8,<0.17")
    # Utilities
    .pip_install(
        "wandb",
        "pyyaml>=6.0",
        "numpy>=1.26",
        "scipy>=1.10",
        "rich",
    )
    # Mount the full project (includes seca/ package and data_filtered/)
    .add_local_dir(PROJECT_ROOT, remote_path="/workspace")
)

app = modal.App("seca-sdpo-distil", image=image)

# Separate volume from the SDFT runs so checkpoints don't collide
outputs_volume = modal.Volume.from_name("seca-sdpo-distil-outputs", create_if_missing=True)
OUTPUTS_MOUNT = "/mnt/outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_streaming(cmd: list[str], env: dict) -> str:
    """Stream subprocess stdout+stderr to Modal console in real-time."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/workspace",
        env=env,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    out = "".join(lines)
    if proc.returncode != 0:
        raise RuntimeError(f"Process failed (exit {proc.returncode})\n{out[-4000:]}")
    return out


def _modal_env() -> dict:
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    # Reduce memory fragmentation under mixed HF + sandbox execution
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    return env


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    timeout=21600,  # 6 hours — SDPO is slower than SDFT due to sandbox execution
    volumes={OUTPUTS_MOUNT: outputs_volume},
)
def run_sdpo_training(
    model: str = "Qwen/Qwen3-1.7B",
    split: str = "train",
    difficulty: str | None = None,
    max_problems: int | None = None,
    epochs: int = 1,
    lr: float = 1e-5,
    num_generations: int = 4,
    max_prompt_length: int = 1024,
    max_completion_length: int = 2048,
    max_teacher_prompt_length: int = 4096,
    teacher_regularization: str = "trust_region",
    teacher_update_rate: float = 0.01,
    distillation_topk: int = 20,
    alpha: float = 1.0,
    is_clip: float = 2.0,
    include_env_feedback: bool = False,
    exec_timeout: float = 10.0,
    seed: int = 42,
) -> str:
    """Run SDPO training on A100.

    Checkpoints are saved in HuggingFace format inside the
    seca-sdpo-distil-outputs volume at /mnt/outputs/checkpoints/sdpo_distil/.

    Args:
        model:                    HF model name (default: Qwen/Qwen3-1.7B)
        split:                    APPS split (train | test | all)
        difficulty:               Filter by difficulty (introductory | interview | competition)
        max_problems:             Cap on problems (None = all)
        epochs:                   Training epochs
        lr:                       Learning rate (paper: 1e-5 for rich-feedback)
        num_generations:          Rollouts per prompt G (paper: 4 for LCBv6)
        max_prompt_length:        Max prompt tokens
        max_completion_length:    Max completion tokens (paper: 2048)
        max_teacher_prompt_length Max tokens for reprompted teacher context
        teacher_regularization:   trust_region (best) or ema
        teacher_update_rate:      mix_coef for trust-region OR ema rate (paper: 0.01)
        distillation_topk:        K for top-K distillation (paper: K=20 rich feedback)
        alpha:                    KL direction: 1.0=reverse KL (paper default)
        is_clip:                  IS ratio clip threshold (paper: 2.0)
        include_env_feedback:     Include execution feedback in teacher (paper default: False)
        exec_timeout:             Sandbox timeout per test case
        seed:                     Random seed
    """
    import os
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    ckpt_dir = f"{OUTPUTS_MOUNT}/checkpoints/sdpo_distil"
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/train_sdpo_distil.py",
        "--model", model,
        "--split", split,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--num-generations", str(num_generations),
        "--max-prompt-length", str(max_prompt_length),
        "--max-completion-length", str(max_completion_length),
        "--max-teacher-prompt-length", str(max_teacher_prompt_length),
        "--teacher-regularization", teacher_regularization,
        "--teacher-update-rate", str(teacher_update_rate),
        "--distillation-topk", str(distillation_topk),
        "--alpha", str(alpha),
        "--is-clip", str(is_clip),
        "--exec-timeout", str(exec_timeout),
        "--output-dir", ckpt_dir,
        "--seed", str(seed),
    ]
    if difficulty is not None:
        cmd.extend(["--difficulty", difficulty])
    if max_problems is not None:
        cmd.extend(["--max-problems", str(max_problems)])
    if include_env_feedback:
        cmd.append("--include-env-feedback")

    env = _modal_env()
    if "WANDB_API_KEY" not in env:
        env["WANDB_API_KEY"] = ""  # empty key → falls back to "none" reporter

    out = _run_streaming(cmd, env=env)
    outputs_volume.commit()
    return out


# ---------------------------------------------------------------------------
# Eval function (runs on saved checkpoint inside the volume)
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    timeout=7200,  # 2 hours for full APPS test evaluation
    volumes={OUTPUTS_MOUNT: outputs_volume},
)
def run_sdpo_eval(
    checkpoint: str = "checkpoints/sdpo_distil",
    split: str = "test",
    difficulty: str | None = None,
    max_problems: int | None = None,
    n_samples: int = 10,
    temperature: float = 0.6,
    max_new_tokens: int = 2048,
    k_values: list[int] | None = None,
) -> str:
    """Evaluate an SDPO checkpoint (HF format) from the persistent volume.

    Args:
        checkpoint:     Relative path inside the volume (e.g. checkpoints/sdpo_distil)
        split:          APPS split to evaluate on (default: test)
        difficulty:     Difficulty filter (optional)
        max_problems:   Cap on evaluation problems
        n_samples:      Completions per problem for Pass@k
        temperature:    Sampling temperature (paper: 0.6 for eval)
        max_new_tokens: Max tokens per completion
        k_values:       k values for Pass@k (default: [1, 5, 10])
    """
    import os
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    log_dir = f"{OUTPUTS_MOUNT}/logs"
    os.makedirs(log_dir, exist_ok=True)

    ckpt_path = f"{OUTPUTS_MOUNT}/{checkpoint}"
    _k_values = k_values or [1, 5, 10]

    cmd = [
        sys.executable,
        "scripts/eval_sdpo_distil.py",
        "--checkpoint", ckpt_path,
        "--split", split,
        "--n-samples", str(n_samples),
        "--temperature", str(temperature),
        "--max-new-tokens", str(max_new_tokens),
        "--output-dir", log_dir,
        "--k-values", *[str(k) for k in _k_values],
    ]
    if difficulty is not None:
        cmd.extend(["--difficulty", difficulty])
    if max_problems is not None:
        cmd.extend(["--max-problems", str(max_problems)])

    out = _run_streaming(cmd, env=_modal_env())
    outputs_volume.commit()
    return out


# ---------------------------------------------------------------------------
# Volume listing helper
# ---------------------------------------------------------------------------

@app.function(volumes={OUTPUTS_MOUNT: outputs_volume})
def list_outputs() -> dict:
    """List checkpoints and logs in the seca-sdpo-distil-outputs volume."""
    import os
    result: dict = {"checkpoints": [], "logs": []}
    ckpt_dir = f"{OUTPUTS_MOUNT}/checkpoints"
    log_dir = f"{OUTPUTS_MOUNT}/logs"
    if os.path.isdir(ckpt_dir):
        for entry in sorted(os.listdir(ckpt_dir)):
            full = os.path.join(ckpt_dir, entry)
            if os.path.isdir(full):
                sub = sorted(os.listdir(full))
                result["checkpoints"].append({entry: sub})
            else:
                result["checkpoints"].append(entry)
    if os.path.isdir(log_dir):
        result["logs"] = sorted(os.listdir(log_dir))
    print("Checkpoints:", result["checkpoints"])
    print("Logs:", result["logs"])
    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    task: str = "training",          # training | eval | list-outputs
    model: str = "Qwen/Qwen3-1.7B",
    split: str = "train",
    difficulty: str | None = None,
    max_problems: int | None = None,
    epochs: int = 1,
    lr: float = 1e-5,
    num_generations: int = 4,
    max_prompt_length: int = 1024,
    max_completion_length: int = 2048,
    max_teacher_prompt_length: int = 4096,
    teacher_regularization: str = "trust_region",
    teacher_update_rate: float = 0.01,
    distillation_topk: int = 20,
    alpha: float = 1.0,
    is_clip: float = 2.0,
    include_env_feedback: bool = False,
    exec_timeout: float = 10.0,
    # eval-only args
    checkpoint: str = "checkpoints/sdpo_distil",
    eval_split: str = "test",
    n_samples: int = 10,
    temperature: float = 0.6,
    seed: int = 42,
) -> None:
    """CLI entrypoint for the seca-sdpo-distil Modal app.

    Examples:
      # Quick smoke test (5 problems, 1 epoch)
      modal run scripts/modal_sdpo_distil.py --task training --max-problems 5 --epochs 1

      # Full training (all APPS train split)
      modal run scripts/modal_sdpo_distil.py --task training --epochs 1

      # Training with environment feedback enabled
      modal run scripts/modal_sdpo_distil.py --task training --include-env-feedback

      # Evaluate saved checkpoint
      modal run scripts/modal_sdpo_distil.py --task eval \\
          --checkpoint checkpoints/sdpo_distil/checkpoint-400

      # List saved checkpoints/logs
      modal run scripts/modal_sdpo_distil.py --task list-outputs
    """
    if task == "training":
        run_sdpo_training.remote(
            model=model,
            split=split,
            difficulty=difficulty,
            max_problems=max_problems,
            epochs=epochs,
            lr=lr,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            max_teacher_prompt_length=max_teacher_prompt_length,
            teacher_regularization=teacher_regularization,
            teacher_update_rate=teacher_update_rate,
            distillation_topk=distillation_topk,
            alpha=alpha,
            is_clip=is_clip,
            include_env_feedback=include_env_feedback,
            exec_timeout=exec_timeout,
            seed=seed,
        )
    elif task == "eval":
        run_sdpo_eval.remote(
            checkpoint=checkpoint,
            split=eval_split,
            difficulty=difficulty,
            max_problems=max_problems,
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_completion_length,
        )
    elif task == "list-outputs":
        list_outputs.remote()
    else:
        raise ValueError(
            f"Unknown task {task!r}. Choose from: training | eval | list-outputs"
        )
