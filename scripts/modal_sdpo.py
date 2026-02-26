#!/usr/bin/env python3
"""Modal runner for SDPO training — separate app from SDFT (seca-training).

SDPO nuances vs SDFT:
  - No gold solution in the teacher context; uses live execution feedback instead.
  - Each rollout is sandbox-executed (subprocess per test case) → more CPU time
    per step. With K=16 completions and ~5 tests each, expect ~80 subprocesses
    per problem. The training timeout is set to 4 h accordingly.
  - Four teacher-context cases (A/B/C/D) based on pass/fail outcome per completion.
  - Cases A/B (passed completions) contribute minimal KL — the main signal comes
    from Cases C (failed but group has a passing demo) and D (all failed).
  - EMA teacher is updated after every problem step, same as SDFT.
  - vLLM weight sync happens after every optimizer.step().

Separate volume (seca-sdpo-outputs) keeps SDPO checkpoints apart from SDFT.

Usage:
  modal run scripts/modal_sdpo.py                               # diagnostics
  modal run scripts/modal_sdpo.py --task training               # full training run
  modal run scripts/modal_sdpo.py --task training \\
      --max-problems 10 --epochs 2 --num-samples 8
  modal run scripts/modal_sdpo.py --task list-outputs           # list checkpoints
  modal volume get seca-sdpo-outputs checkpoints/ ./local_ckpt/ # download
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Same image spec as seca-training; separate app allows independent deployment.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        'pip install "torch>=2.1,<2.5" --extra-index-url https://download.pytorch.org/whl/cu124'
    )
    .pip_install(
        "transformers>=4.40",
        "datasets>=2.18",
        "vllm>=0.8,<0.17",
        "accelerate>=1.2",
        "trl>=0.16",
        "pyyaml>=6.0",
        "numpy>=1.26",
        "torch-c-dlpack-ext",   # silences TVM EnvTensorAllocator warning
    )
    .add_local_dir(PROJECT_ROOT, remote_path="/workspace")
)

app = modal.App("seca-sdpo", image=image)

# Separate persistent volume from seca-training so SDFT and SDPO
# checkpoints and logs never collide.
outputs_volume = modal.Volume.from_name("seca-sdpo-outputs", create_if_missing=True)
OUTPUTS_MOUNT = "/mnt/outputs"


def _run_streaming(cmd: list[str], env: dict) -> str:
    """Stream subprocess stdout+stderr in real time; raise on non-zero exit."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge so we get one ordered stream
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
    # Must be 0: vLLM EngineCore must run in-process so direct weight sync works.
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["PYTHONPATH"] = "/workspace"
    # Reduces vLLM+HF memory fragmentation on A100.
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    return env


@app.function(
    gpu="A100",
    timeout=3600,
)
def run_sdpo_diagnostics(config_path: str = "configs/default.yaml") -> str:
    """Run 8 SDPO-specific diagnostic checks on A100 with live log output.

    Checks (no model needed):
      1. Executor pass/fail detection — gold passes, wrong code fails
      2. Teacher context construction — Cases A/B/C/D are distinct and correct

    Checks (model required):
      3. SDPO KL non-zero            — Case D KL in [0.01, 5.0]
      4. Feedback changes KL         — Case C (demo) ≠ Case D (no demo)
      5. Gradients flow              — total norm in [0.01, 200.0]
      6. EMA teacher updates         — delta norm ~1e-6 to 1e-4
      7. vLLM weight sync            — no crash after SDPO gradient step
      8. Loss decreases              — second-half mean < first-half (20 steps)
    """
    import os
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    return _run_streaming(
        [sys.executable, "scripts/run_sdpo_diagnostics.py", "--config", config_path],
        env=_modal_env(),
    )


@app.function(
    gpu="A100",
    # 4 hours: SDPO runs sandbox subprocesses for every rollout.
    # Rough estimate: K=16 completions × 5 tests × 10s timeout × 100 problems ≈ 2.2 h.
    # Buffer for model forward passes, EMA updates, and vLLM sync overhead.
    timeout=14400,
    volumes={OUTPUTS_MOUNT: outputs_volume},
)
def run_sdpo_training(
    config_path: str = "configs/default.yaml",
    max_problems: int | None = None,
    epochs: int | None = None,
    num_samples: int | None = None,       # G rollouts per problem (default 16 in config)
    mini_batch_size: int | None = None,   # gradient-accum chunk (default 1)
    gpu_memory_util: float | None = None, # vLLM GPU fraction (default 0.3)
    max_new_tokens: int | None = None,    # max tokens per rollout (default 256)
) -> str:
    """Run SDPO training on A100. Checkpoints and logs saved to seca-sdpo-outputs.

    The training loop (scripts/run.py --mode sdpo) does for each problem:
      1. Generate K completions via vLLM (fast inference).
      2. Execute each completion in the sandbox to obtain FeedbackBundles.
      3. For each completion, build the SDPO teacher context (one of 4 cases):
           A: passed, different demo available  → teacher sees alt correct solution
           B: passed, only solution             → teacher sees own solution (KL ≈ 0)
           C: failed, group has a passing demo  → teacher sees demo + feedback
           D: all failed                        → teacher sees feedback only
      4. Compute KL(student || EMA-teacher) over completion tokens only.
      5. Backprop, clip gradients, optimizer step.
      6. EMA update: ϕ ← 0.995·ϕ + 0.005·θ
      7. Sync updated HF weights back into vLLM for next rollout.
    """
    import os
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    ckpt_dir = f"{OUTPUTS_MOUNT}/checkpoints"
    log_dir = f"{OUTPUTS_MOUNT}/logs"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        sys.executable, "scripts/run.py",
        "--mode", "sdpo",
        "--config", config_path,
        "--checkpoint-dir", ckpt_dir,
        "--log-dir", log_dir,
    ]
    if max_problems is not None:
        cmd.extend(["--max-problems", str(max_problems)])
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if num_samples is not None:
        cmd.extend(["--num-samples", str(num_samples)])
    if mini_batch_size is not None:
        cmd.extend(["--mini-batch-size", str(mini_batch_size)])
    if gpu_memory_util is not None:
        cmd.extend(["--gpu-memory-util", str(gpu_memory_util)])
    if max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])

    out = _run_streaming(cmd, env=_modal_env())
    outputs_volume.commit()
    return out


@app.function(volumes={OUTPUTS_MOUNT: outputs_volume})
def list_sdpo_outputs() -> dict:
    """List SDPO checkpoints and logs in the persistent volume."""
    import os
    result = {"checkpoints": [], "logs": []}
    ckpt_dir = f"{OUTPUTS_MOUNT}/checkpoints"
    log_dir = f"{OUTPUTS_MOUNT}/logs"
    if os.path.isdir(ckpt_dir):
        result["checkpoints"] = sorted(os.listdir(ckpt_dir))
    if os.path.isdir(log_dir):
        result["logs"] = sorted(os.listdir(log_dir))
    print("Checkpoints:", result["checkpoints"])
    print("Logs:", result["logs"])
    return result


@app.local_entrypoint()
def main(
    task: str = "diagnostics",           # diagnostics | training | list-outputs
    config: str = "configs/default.yaml",
    max_problems: int | None = None,
    epochs: int | None = None,
    num_samples: int | None = None,
    mini_batch_size: int | None = None,
    gpu_memory_util: float | None = None,
    max_new_tokens: int | None = None,
) -> None:
    """CLI entrypoint.

    Examples:
      modal run scripts/modal_sdpo.py
      modal run scripts/modal_sdpo.py --task training --max-problems 5 --epochs 1
      modal run scripts/modal_sdpo.py --task training --num-samples 8 --mini-batch-size 2
      modal run scripts/modal_sdpo.py --task list-outputs
    """
    if task == "diagnostics":
        run_sdpo_diagnostics.remote(config_path=config)
    elif task == "list-outputs":
        list_sdpo_outputs.remote()
    elif task == "training":
        run_sdpo_training.remote(
            config_path=config,
            max_problems=max_problems,
            epochs=epochs,
            num_samples=num_samples,
            mini_batch_size=mini_batch_size,
            gpu_memory_util=gpu_memory_util,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(
            f"Unknown task: {task!r}. Choose from 'diagnostics', 'training', 'list-outputs'."
        )
