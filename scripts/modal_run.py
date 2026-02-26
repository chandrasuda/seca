#!/usr/bin/env python3
"""Run SECA diagnostics and training on Modal with A100 GPU.

Usage:
  modal run scripts/modal_run.py::run_diagnostics
  modal run scripts/modal_run.py::run_training --mode sdft --max-problems 5 --epochs 1
  modal run scripts/modal_run.py --task list-outputs   # list checkpoints in seca-outputs volume
  modal volume get seca-outputs checkpoints/sdft_epoch0.pt ./local.pt  # download checkpoint
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Image with PyTorch (CUDA), vLLM, and seca dependencies
# Modal GPU containers have CUDA; install PyTorch from PyTorch index for CUDA build
# add_local_dir bundles the project into the image at container startup (copy=False)
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
        "torch-c-dlpack-ext",  # silences TVM EnvTensorAllocator warning
    )
    .add_local_dir(PROJECT_ROOT, remote_path="/workspace")
)

app = modal.App("seca-training", image=image)

# Persistent volume for checkpoints and logs (survives container restarts)
outputs_volume = modal.Volume.from_name("seca-outputs", create_if_missing=True)
OUTPUTS_MOUNT = "/mnt/outputs"


def _run_streaming(cmd: list[str], env: dict) -> str:
    """Stream subprocess stdout+stderr to Modal in real-time; raise on non-zero exit."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge so we get one ordered stream
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
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["PYTHONPATH"] = "/workspace"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    return env


@app.function(
    gpu="A100",
    timeout=3600,
)
def run_diagnostics(config_path: str = "configs/default.yaml") -> str:
    """Run SDFT diagnostic checks on A100 with live log output."""
    import os
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    return _run_streaming(
        [sys.executable, "scripts/run_diagnostics.py", "--config", config_path],
        env=_modal_env(),
    )


@app.function(
    gpu="A100",
    timeout=7200,
    volumes={OUTPUTS_MOUNT: outputs_volume},
)
def run_training(
    mode: str = "sdft",
    config_path: str = "configs/default.yaml",
    max_problems: int | None = None,
    epochs: int | None = None,
    num_samples: int | None = None,
    mini_batch_size: int | None = None,
) -> str:
    """Run SECA training on A100. Checkpoints and logs are saved to the seca-outputs volume."""
    import os
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")

    ckpt_dir = f"{OUTPUTS_MOUNT}/checkpoints"
    log_dir = f"{OUTPUTS_MOUNT}/logs"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        sys.executable, "scripts/run.py", "--mode", mode, "--config", config_path,
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

    out = _run_streaming(cmd, env=_modal_env())
    outputs_volume.commit()
    return out


@app.function(volumes={OUTPUTS_MOUNT: outputs_volume})
def list_outputs() -> dict:
    """List checkpoints and logs in the persistent volume."""
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
    task: str = "diagnostics",  # diagnostics | training | list-outputs
    config: str = "configs/default.yaml",
    mode: str = "sdft",
    max_problems: int | None = None,
    epochs: int | None = None,
    num_samples: int | None = None,
    mini_batch_size: int | None = None,
) -> None:
    """CLI entrypoint. Usage:
    modal run scripts/modal_run.py  # runs diagnostics
    modal run scripts/modal_run.py --task training --mode sdft --max-problems 5 --epochs 1
    """
    if task == "diagnostics":
        run_diagnostics.remote(config_path=config)
    elif task == "list-outputs":
        list_outputs.remote()
    elif task == "training":
        run_training.remote(
            mode=mode,
            config_path=config,
            max_problems=max_problems,
            epochs=epochs,
            num_samples=num_samples,
            mini_batch_size=mini_batch_size,
        )
    else:
        raise ValueError(f"Unknown task: {task}. Use 'diagnostics', 'training', or 'list-outputs'.")
