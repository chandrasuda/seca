"""Training loop: SFT, SDFT, SDPO, Omni, GRPO — with checkpoints + eval.

SDFT/SDPO use EMA teacher (α=0.01), AdamW (lr=1e-6, weight_decay=0.01),
gradient clipping 1.0, and per-step EMA update.
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
import torch
from torch.optim import AdamW
from seca.data.problem import CodeProblem
from seca.models.base import BaseModel
from seca.models.sdft import SDFTOperator
from seca.models.sdpo import SDPOOperator
from seca.models.omni import OmniTeacher
from seca.sandbox.executor import execute_code
from seca.eval.metrics import evaluate_problems

log = logging.getLogger(__name__)


def _ema_update(ema_model: BaseModel, model: BaseModel, alpha: float = 0.01):
    """Update EMA teacher: ϕ ← (1-α)·ϕ + α·θ. No gradients."""
    with torch.no_grad():
        for p_ema, p_model in zip(ema_model.model.parameters(), model.model.parameters()):
            p_ema.data.mul_(1.0 - alpha).add_(p_model.data, alpha=alpha)


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = cfg["training"]["mode"]
        self.model = BaseModel(**cfg["model"])

        # SDPO paper Table 12: lr=1e-6, weight_decay=0.01
        train_cfg = cfg["training"]
        lr = train_cfg.get("lr", 1e-6)
        weight_decay = train_cfg.get("weight_decay", 0.01)
        self.optimizer = AdamW(
            self.model.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.K = train_cfg.get("num_samples", 4)
        self.num_epochs = train_cfg.get("num_epochs", 3)
        self.ema_alpha = train_cfg.get("ema_alpha", 0.01)

        self.log_dir = Path(cfg["eval"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.eval_every = train_cfg.get("eval_every_epoch", 1)

        ops = {"sdft": SDFTOperator, "sdpo": SDPOOperator, "omni": OmniTeacher}
        self.op = ops[self.mode](cfg.get(self.mode, {})) if self.mode in ops else None

    def train(self, problems: list[CodeProblem],
              eval_problems: list[CodeProblem] | None = None) -> list[dict]:
        history: list[dict] = []
        eval_problems = eval_problems or problems[:20]

        needs_teacher = self.mode in ("sdft", "sdpo", "omni")
        ema_teacher = self.model.snapshot() if needs_teacher else None

        for epoch in range(self.num_epochs):
            t0 = time.time()
            log.info(f"Epoch {epoch+1}/{self.num_epochs}  mode={self.mode}")

            epoch_metrics = self._train_epoch(problems, epoch, ema_teacher)
            epoch_metrics["wall_time_s"] = round(time.time() - t0, 1)

            # per-epoch eval
            if (epoch + 1) % self.eval_every == 0:
                eval_cfg = self.cfg.get("eval", {})
                ev = evaluate_problems(
                    self.model, eval_problems,
                    n_samples=min(eval_cfg.get("n_samples", 5), 5),
                    k_values=[1],
                    temperature=eval_cfg.get("temperature", 0.8),
                )
                epoch_metrics["eval_pass@1"] = ev.get("pass@1", 0.0)
                log.info(f"  eval pass@1={ev.get('pass@1', 0):.4f}")

            # save checkpoint
            self._save_checkpoint(epoch)

            history.append(epoch_metrics)
            log.info(f"  loss={epoch_metrics['loss']:.4f}  wall={epoch_metrics['wall_time_s']}s")

        self._save_log(history)
        return history

    def _train_epoch(self, problems: list[CodeProblem], epoch: int,
                     ema_teacher: BaseModel | None) -> dict:
        total_loss = 0.0
        all_metrics: dict[str, float] = {}
        needs_teacher = self.mode in ("sdft", "sdpo", "omni")

        for i, problem in enumerate(problems):
            prompts = [problem.format_prompt()] * self.K

            if self.mode == "sft":
                loss, metrics = self._sft_step(problem)
            elif self.mode == "grpo":
                loss, metrics = self._grpo_step(problem, prompts)
            else:
                completions = self.model.generate(prompts, temperature=0.8)
                feedback_bundles = [execute_code(c, problem) for c in completions]
                loss, metrics = self._distill_step(
                    ema_teacher, problem, completions, feedback_bundles,
                )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(), self.max_grad_norm,
            )
            self.optimizer.step()

            # EMA update (separate from optimizer, no gradients)
            if needs_teacher and ema_teacher is not None:
                _ema_update(ema_teacher, self.model, alpha=self.ema_alpha)

            total_loss += loss.item()
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v

        n = len(problems)
        return {
            "epoch": epoch,
            "mode": self.mode,
            "loss": total_loss / n,
            **{k: v / n for k, v in all_metrics.items()},
        }

    def _sft_step(self, problem: CodeProblem):
        text = f"{problem.format_prompt()}\n{problem.gold_solution}"
        enc = self.model.encode([text])
        loss = self.model.model(**enc, labels=enc["input_ids"]).loss
        return loss, {"nll": loss.item()}

    def _grpo_step(self, problem: CodeProblem, prompts: list[str]):
        completions = self.model.generate(prompts, temperature=0.8)
        rewards = []
        for c in completions:
            fb = execute_code(c, problem)
            rewards.append(fb.pass_rate)

        r = torch.tensor(rewards, device=self.model.device)
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

    def _distill_step(self, teacher: BaseModel | None, problem: CodeProblem,
                     completions: list[str], feedback_bundles: list):
        probs = [problem] * len(completions)
        if self.mode == "sdft":
            return self.op.loss(self.model, teacher, probs, completions)
        return self.op.loss(self.model, teacher, probs, completions, feedback_bundles)

    def _save_checkpoint(self, epoch: int):
        path = self.ckpt_dir / f"{self.mode}_epoch{epoch}.pt"
        torch.save(self.model.model.state_dict(), path)
        log.info(f"  checkpoint → {path}")

    def _save_log(self, history):
        out = self.log_dir / f"train_{self.mode}.jsonl"
        with open(out, "w") as f:
            for entry in history:
                f.write(json.dumps(entry) + "\n")
        log.info(f"Training log → {out}")
