#!/usr/bin/env python3
"""Modal runner for SDFT training via the Self-Distillation submodule.

Trains on APPS using DistilTrainer (Self-Distillation/distil_trainer.py), which
implements the exact methodology from Shenfeld et al. 2026 (arxiv:2601.19897).

Key differences from the old seca-training app (modal_run.py):
  - Uses DistilTrainer, not the custom seca Trainer
  - Installs the exact dependency versions required by the submodule (trl, vllm, etc.)
  - Saves checkpoints in HF format (full model dir), evaluated by eval_sdft_distil.py
  - Evaluation is a separate function so you can re-run it without re-training

Usage:
  # Smoke test (5 problems, 1 epoch)
  modal run scripts/modal_distil_sdft.py --task training --max-problems 5 --epochs 1

  # Full training run (all train-split problems, 1 epoch)
  modal run scripts/modal_distil_sdft.py --task training --epochs 1

  # Training with difficulty filter
  modal run scripts/modal_distil_sdft.py --task training \
      --difficulty introductory --max-problems 500 --epochs 2

  # Evaluate a checkpoint that was already saved to the volume
  modal run scripts/modal_distil_sdft.py --task eval \
      --checkpoint checkpoints/sdft_distil/checkpoint-200

  # List saved checkpoints / logs
  modal run scripts/modal_distil_sdft.py --task list-outputs
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Modal image
# TRL >= 0.18: colocate mode for single-GPU training.
# TRL >= 0.28: VLLMClient moved to trl.generation.vllm_client (handled in distil_trainer.py).
# TRL supports vLLM 0.10.2, 0.11.x, 0.12.0 — pin vLLM 0.12.0 for compatibility.
# Self-Distillation submodule uses vllm_mode="colocate" (single-GPU friendly).
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # vLLM first — it pins compatible torch/CUDA versions automatically.
    # TRL's supported vLLM range includes 0.12.0.
    .pip_install("vllm==0.12.0")
    # Core training stack — no upper bound on transformers so TRL can pull what it needs
    .pip_install(
        "transformers>=4.47",
        "accelerate>=1.2",
        "datasets>=2.18",
        "peft>=0.14",
    )
    # TRL 0.24.0 — matches Self-Distillation requirements; avoid API drift (prepare_peft_model, etc.)
    .pip_install("trl==0.24.0")
    # Utilities
    .pip_install(
        "wandb",
        "pyyaml>=6.0",
        "numpy>=1.26",
        "scipy>=1.10",
        "rich",
    )
    # Mount the full project (includes Self-Distillation/ submodule and seca/ package)
    .add_local_dir(PROJECT_ROOT, remote_path="/workspace")
)

app = modal.App("seca-sdft-distil", image=image)

# Separate volume from the old seca-training runs so checkpoints don't collide
outputs_volume = modal.Volume.from_name("seca-sdft-distil-outputs", create_if_missing=True)
OUTPUTS_MOUNT = "/mnt/outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_streaming(cmd: list[str], env: dict) -> str:
    """Stream subprocess stdout+stderr to Modal console in real-time."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge for a single ordered stream
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
    # Required so vLLM EngineCore runs in-process for weight sync
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["PYTHONPATH"] = "/workspace"
    # Reduce memory fragmentation under vLLM + HuggingFace coexistence
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    return env


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    timeout=14400,  # 4 hours (APPS train split ≈ 4.3k usable problems)
    volumes={OUTPUTS_MOUNT: outputs_volume},
)
def run_sdft_training(
    model: str = "Qwen/Qwen3-1.7B",
    split: str = "train",
    difficulty: str | None = None,
    max_problems: int | None = None,
    data_file: str | None = None,
    epochs: int = 1,
    lr: float = 1e-5,
    num_generations: int = 8,
    max_prompt_length: int | None = None,
    max_completion_length: int = 2048,
    ema_alpha: float = 0.01,
    vllm_gpu_memory: float = 0.35,
    seed: int = 42,
) -> str:
    """Run SDFT training on A100.

    Checkpoints are saved as HuggingFace model directories inside
    the seca-sdft-distil-outputs volume at /mnt/outputs/checkpoints/sdft_distil/.

    Args:
        model:                  HF model name or path (default Qwen/Qwen3-1.7B)
        split:                  APPS split (train | test | all)
        difficulty:             Filter by difficulty (introductory | interview | competition)
        max_problems:           Cap on problems (None = all with gold + tests)
        epochs:                 Training epochs
        lr:                     Learning rate (paper: 1e-5 for skill learning)
        num_generations:        Rollouts per prompt G (paper: 8)
        max_prompt_length:      Max prompt tokens
        max_completion_length:  Max completion tokens (paper: 2048)
        ema_alpha:              EMA teacher mixup α (paper: 0.01)
        vllm_gpu_memory:        vLLM GPU fraction (sleep mode reclaims this during backward)
        seed:                   Random seed
    """
    import os
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    ckpt_dir = f"{OUTPUTS_MOUNT}/checkpoints/sdft_distil"
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/train_sdft_distil.py",
        "--model", model,
        "--split", split,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--num-generations", str(num_generations),
        "--max-completion-length", str(max_completion_length),
        "--ema-alpha", str(ema_alpha),
        "--vllm-gpu-memory", str(vllm_gpu_memory),
        "--output-dir", ckpt_dir,
        "--seed", str(seed),
    ]
    resolved_data_file = data_file
    if resolved_data_file is None:
        resolved_data_file = (
            "/workspace/leetcode/leetcode_train_apps_harness_fmt.jsonl"
            if split == "train"
            else "/workspace/leetcode/leetcode_test_apps_harness_fmt.jsonl"
        )
    cmd.extend(["--data-file", resolved_data_file])
    if difficulty is not None:
        cmd.extend(["--difficulty", difficulty])
    if max_problems is not None:
        cmd.extend(["--max-problems", str(max_problems)])
    if max_prompt_length is not None:
        cmd.extend(["--max-prompt-length", str(max_prompt_length)])

    env = _modal_env()
    # Pass W&B credentials if configured as Modal secrets
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
def run_sdft_eval(
    checkpoint: str = "checkpoints/sdft_distil",
    split: str = "test",
    difficulty: str | None = None,
    max_problems: int | None = None,
    n_samples: int = 10,
    temperature: float = 0.6,
    max_new_tokens: int = 2048,
    k_values: list[int] | None = None,
) -> str:
    """Evaluate a SDFT checkpoint (HF format) from the persistent volume.

    Args:
        checkpoint:     Relative path inside the volume (e.g. checkpoints/sdft_distil)
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
        "scripts/eval_sdft_distil.py",
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
    """List checkpoints and logs in the seca-sdft-distil-outputs volume."""
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
    data_file: str | None = None,
    epochs: int = 1,
    lr: float = 1e-5,
    num_generations: int = 8,
    max_prompt_length: int | None = None,
    max_completion_length: int = 2048,
    ema_alpha: float = 0.01,
    vllm_gpu_memory: float = 0.35,
    # eval-only args
    checkpoint: str = "checkpoints/sdft_distil",
    eval_split: str = "test",
    n_samples: int = 10,
    temperature: float = 0.6,
    seed: int = 42,
) -> None:
    """CLI entrypoint for the seca-sdft-distil Modal app.

    Examples:
      # Quick smoke test (5 problems, 1 epoch)
      modal run scripts/modal_distil_sdft.py --task training --max-problems 5 --epochs 1

      # Full training (all APPS train split)
      modal run scripts/modal_distil_sdft.py --task training --epochs 1

      # Evaluate saved checkpoint
      modal run scripts/modal_distil_sdft.py --task eval \
          --checkpoint checkpoints/sdft_distil/checkpoint-200

      # List saved checkpoints/logs
      modal run scripts/modal_distil_sdft.py --task list-outputs
    """
    if task == "training":
        run_sdft_training.remote(
            model=model,
            split=split,
            difficulty=difficulty,
            max_problems=max_problems,
            data_file=data_file,
            epochs=epochs,
            lr=lr,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            ema_alpha=ema_alpha,
            vllm_gpu_memory=vllm_gpu_memory,
            seed=seed,
        )
    elif task == "eval":
        run_sdft_eval.remote(
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
