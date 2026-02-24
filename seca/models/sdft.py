"""Self-Distillation Fine-Tuning (SDFT) update operator.

Implements on-policy distillation from Shenfeld et al. (2026):
  1. Snapshot current model as teacher.
  2. Condition teacher on demonstration (prefix with demo).
  3. Student samples K completions on-policy.
  4. Compute forward-KL(teacher ∥ student) on the on-policy samples.
  5. Update student to minimise that loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from seca.models.base import BaseModel
from seca.data.episode import Episode


class SDFTOperator:
    """Applies one SDFT update step for a single episode."""

    def __init__(self, cfg: dict):
        self.K = cfg.get("num_on_policy_samples", 4)
        self.temp_s = cfg.get("temperature_student", 1.0)
        self.temp_t = cfg.get("temperature_teacher", 0.7)
        self.kl_weight = cfg.get("kl_weight", 0.1)
        self.lr = cfg.get("lr", 1e-5)
        self.epochs = cfg.get("epochs_per_episode", 1)
        self.max_grad_norm = cfg.get("max_grad_norm", 1.0)

    @torch.no_grad()
    def _teacher_logits(
        self, teacher: BaseModel, demo_prefix: str, student_completions: list[str],
    ) -> torch.Tensor:
        """Get teacher logits conditioned on the demonstration."""
        texts = [f"{demo_prefix}\n{c}" for c in student_completions]
        enc = teacher.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=teacher.max_len,
        ).to(teacher.device)
        return teacher.model(**enc).logits

    def update(self, model: BaseModel, episode: Episode) -> dict:
        """Run SDFT on *model* for one episode; returns metrics dict."""
        teacher = model.snapshot()

        # demo prefix for teacher conditioning
        demo_prefix = f"### Instruction\n{episode.prompt}\n### Response\n{episode.reference}"

        optimizer = AdamW(model.model.parameters(), lr=self.lr)

        total_loss = 0.0
        for _ in range(self.epochs):
            # 1. student on-policy samples
            completions = model.generate(
                [episode.prompt] * self.K,
                temperature=self.temp_s,
                do_sample=True,
            )

            # 2. build training sequences: prompt + completion
            seqs = [f"{episode.prompt}\n{c}" for c in completions]
            enc = model.tokenizer(
                seqs, return_tensors="pt", padding=True,
                truncation=True, max_length=model.max_len,
            ).to(model.device)

            # 3. student logits
            student_logits = model.model(**enc).logits / self.temp_s

            # 4. teacher logits (demo-conditioned)
            teacher_logits = self._teacher_logits(teacher, demo_prefix, completions)

            # align sequence lengths (teacher may be longer due to demo prefix)
            min_len = min(student_logits.size(1), teacher_logits.size(1))
            s_logits = student_logits[:, :min_len, :]
            t_logits = teacher_logits[:, -min_len:, :] / self.temp_t

            # 5. forward KL: KL(teacher ∥ student)
            t_probs = F.softmax(t_logits, dim=-1)
            s_log_probs = F.log_softmax(s_logits, dim=-1)
            kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")

            # 6. supervised NLL on gold reference
            ref_enc = model.tokenizer(
                [f"{episode.prompt}\n{episode.reference}"],
                return_tensors="pt", padding=True,
                truncation=True, max_length=model.max_len,
            ).to(model.device)
            nll = model.model(**ref_enc, labels=ref_enc["input_ids"]).loss

            loss = nll + self.kl_weight * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), self.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()

        return {"sdft_loss": total_loss / max(self.epochs, 1)}
