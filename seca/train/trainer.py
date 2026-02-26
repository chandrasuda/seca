"""Training loop: SFT, SDFT, SDPO, Omni, GRPO — vLLM inference, HF training.

SDFT/SDPO use EMA teacher, AdamW, gradient clipping, and per-step weight sync to vLLM.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from vllm import SamplingParams

from seca.data.problem import CodeProblem
from seca.eval.metrics import evaluate_problems
from seca.models.base import BaseModel
from seca.models.omni import OmniTeacher
from seca.models.sdft import SDFTOperator
from seca.models.sdpo import SDPOOperator
from seca.sandbox.executor import execute_code

log = logging.getLogger(__name__)


def _ema_update(ema_model: BaseModel, model: BaseModel, alpha: float = 0.01) -> None:
    """Update EMA teacher: ϕ ← (1-α)·ϕ + α·θ. No gradients."""
    with torch.no_grad():
        for p_ema, p_model in zip(ema_model.model.parameters(), model.model.parameters()):
            p_ema.data.mul_(1.0 - alpha).add_(p_model.data, alpha=alpha)


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = cfg["training"]["mode"]

        # Build model with vLLM + HF dual-engine
        model_cfg = dict(cfg["model"])
        model_cfg["vllm_cfg"] = cfg.get("vllm", {})
        self.model = BaseModel(**model_cfg)

        # Sampling params for vLLM rollout generation
        inf_cfg = cfg.get("inference", {})
        self.sampling_params = SamplingParams(
            temperature=inf_cfg.get("temperature", 1.0),
            top_p=inf_cfg.get("top_p", 0.95),
            max_tokens=inf_cfg.get("max_new_tokens", 512),
        )

        train_cfg = cfg["training"]
        lr = train_cfg.get("lr", 5e-7)
        weight_decay = train_cfg.get("weight_decay", 0.01)
        self.optimizer = AdamW(
            self.model.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.K = train_cfg.get("num_samples", 16)
        self.mini_batch_size = train_cfg.get("mini_batch_size", 4)
        self.num_epochs = train_cfg.get("num_epochs", 3)
        self.ema_alpha = train_cfg.get("ema_alpha", 0.005)

        self.log_dir = Path(cfg["eval"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.eval_every = train_cfg.get("eval_every_epoch", 1)

        ops = {"sdft": SDFTOperator, "sdpo": SDPOOperator, "omni": OmniTeacher}
        self.op = ops[self.mode](cfg.get(self.mode, {})) if self.mode in ops else None

        log.info(
            "Trainer init: mode=%s, K=%d, mini_batch=%d, lr=%.0e, ema_alpha=%.4f",
            self.mode,
            self.K,
            self.mini_batch_size,
            lr,
            self.ema_alpha,
        )

    def train(
        self,
        problems: list[CodeProblem],
        eval_problems: list[CodeProblem] | None = None,
    ) -> list[dict]:
        history: list[dict] = []
        eval_problems = eval_problems or problems[: min(20, len(problems))]

        needs_teacher = self.mode in ("sdft", "sdpo", "omni")
        ema_teacher = self.model.snapshot() if needs_teacher else None
        if needs_teacher:
            log.info("EMA teacher created (alpha=%.4f)", self.ema_alpha)

        for epoch in range(self.num_epochs):
            t0 = time.time()
            log.info("Epoch %d/%d  mode=%s  problems=%d", epoch + 1, self.num_epochs, self.mode, len(problems))

            epoch_metrics = self._train_epoch(problems, epoch, ema_teacher)
            epoch_metrics["wall_time_s"] = round(time.time() - t0, 1)

            if (epoch + 1) % self.eval_every == 0:
                eval_cfg = self.cfg.get("eval", {})
                inf_cfg = self.cfg.get("inference", {})
                n_eval = min(eval_cfg.get("n_samples", 5), 5)
                ev = evaluate_problems(
                    self.model,
                    eval_problems,
                    n_samples=n_eval,
                    k_values=[1],
                    temperature=eval_cfg.get("temperature", 0.6),
                    sampling_params=SamplingParams(
                        temperature=eval_cfg.get("temperature", 0.6),
                        top_p=eval_cfg.get("top_p", 0.95),
                        max_tokens=inf_cfg.get("max_new_tokens", 512),
                    ),
                )
                epoch_metrics["eval_pass@1"] = ev.get("pass@1", 0.0)
                log.info("  eval pass@1=%.4f (n_samples=%d)", epoch_metrics["eval_pass@1"], n_eval)

            self._save_checkpoint(epoch)

            history.append(epoch_metrics)
            log.info(
                "  loss=%.4f  wall=%.1fs",
                epoch_metrics["loss"],
                epoch_metrics["wall_time_s"],
            )

        self._save_log(history)
        return history

    def _train_epoch(
        self,
        problems: list[CodeProblem],
        epoch: int,
        ema_teacher: BaseModel | None,
    ) -> dict:
        total_loss = 0.0
        all_metrics: dict[str, float] = {}
        needs_teacher = self.mode in ("sdft", "sdpo", "omni")

        for i, problem in enumerate(problems):
            step_start = time.time()
            prompt = problem.format_prompt()
            prompts = [prompt] * self.K

            if self.mode == "sft":
                loss, metrics = self._sft_step(problem)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
            elif self.mode == "grpo":
                completions = self.model.generate(prompts, sampling_params=self.sampling_params)
                loss, metrics = self._grpo_step(problem, completions)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
            else:
                # SDFT / SDPO / Omni: vLLM generate → sandbox execute → KL loss
                completions = self.model.generate(prompts, sampling_params=self.sampling_params)
                feedback_bundles = [execute_code(c, problem) for c in completions]
                n_pass = sum(1 for fb in feedback_bundles if fb.all_passed)
                log.debug("  problem %d: %d/%d passed", i + 1, n_pass, self.K)

                # Gradient accumulation over mini-batches
                num_chunks = (len(completions) + self.mini_batch_size - 1) // self.mini_batch_size
                self.optimizer.zero_grad()
                step_loss_sum = 0.0
                problem_metrics: dict[str, float] = {}
                for chunk_start in range(0, len(completions), self.mini_batch_size):
                    chunk_end = min(chunk_start + self.mini_batch_size, len(completions))
                    chunk_completions = completions[chunk_start:chunk_end]
                    chunk_feedbacks = feedback_bundles[chunk_start:chunk_end]
                    loss, metrics = self._distill_step(
                        ema_teacher, problem, chunk_completions, chunk_feedbacks
                    )
                    (loss / num_chunks).backward()
                    step_loss_sum += loss.item()
                    for k, v in metrics.items():
                        problem_metrics[k] = problem_metrics.get(k, 0.0) + v

                for k in problem_metrics:
                    problem_metrics[k] /= num_chunks
                for k, v in problem_metrics.items():
                    all_metrics[k] = all_metrics.get(k, 0.0) + v

                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                loss_val = step_loss_sum / num_chunks
                loss = torch.tensor(loss_val, device=self.model.device)
                metrics = problem_metrics

            if needs_teacher and ema_teacher is not None:
                _ema_update(ema_teacher, self.model, alpha=self.ema_alpha)

            if self.mode in ("sdft", "sdpo", "omni"):
                self.model.sync_vllm_weights()
                log.debug("  synced HF weights to vLLM")

            total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - step_start
                step_loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                log.info(
                    "  step %d/%d  loss=%.4f  step_time=%.1fs  avg_loss=%.4f",
                    i + 1,
                    len(problems),
                    step_loss_val,
                    elapsed,
                    total_loss / (i + 1),
                )

        n = len(problems)
        return {
            "epoch": epoch,
            "mode": self.mode,
            "loss": total_loss / n,
            **{k: v / n for k, v in all_metrics.items()},
        }

    def _sft_step(self, problem: CodeProblem) -> tuple[torch.Tensor, dict]:
        text = f"{problem.format_prompt()}\n{problem.gold_solution}"
        enc = self.model.encode([text])
        loss = self.model.model(**enc, labels=enc["input_ids"]).loss
        return loss, {"nll": loss.item()}

    def _grpo_step(
        self, problem: CodeProblem, completions: list[str]
    ) -> tuple[torch.Tensor, dict]:
        rewards = []
        for c in completions:
            fb = execute_code(c, problem)
            rewards.append(fb.pass_rate)

        r = torch.tensor(rewards, device=self.model.device, dtype=torch.float32)
        if r.std() > 0:
            r = (r - r.mean()) / (r.std() + 1e-8)

        total_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        for comp, reward in zip(completions, r):
            text = f"{problem.format_prompt()}\n{comp}"
            enc = self.model.encode([text])
            nll = self.model.model(**enc, labels=enc["input_ids"]).loss
            total_loss = total_loss + (-reward * nll)

        return total_loss / len(completions), {
            "grpo_mean_reward": sum(rewards) / len(rewards),
        }

    def _distill_step(
        self,
        teacher: BaseModel | None,
        problem: CodeProblem,
        completions: list[str],
        feedback_bundles: list,
    ) -> tuple[torch.Tensor, dict]:
        probs = [problem] * len(completions)
        if self.mode == "sdft":
            return self.op.loss(self.model, teacher, probs, completions)
        return self.op.loss(self.model, teacher, probs, completions, feedback_bundles)

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.ckpt_dir / f"{self.mode}_epoch{epoch}.pt"
        torch.save(self.model.model.state_dict(), path)
        log.info("  checkpoint → %s", path)

    def _save_log(self, history: list[dict]) -> None:
        out = self.log_dir / f"train_{self.mode}.jsonl"
        with open(out, "w") as f:
            for entry in history:
                f.write(json.dumps(entry) + "\n")
        log.info("Training log → %s", out)
