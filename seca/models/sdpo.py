"""SDPO — execution-feedback-conditioned self-distillation.

Teacher sees: prompt + student attempt + execution feedback.
"""
from __future__ import annotations
import torch, torch.nn.functional as F
from seca.models.base import BaseModel
from seca.data.problem import CodeProblem


class SDPOOperator:
    def __init__(self, cfg: dict):
        self.temp_s = cfg.get("temperature_student", 1.0)
        self.temp_t = cfg.get("temperature_teacher", 0.7)
        self.kl_weight = cfg.get("kl_weight", 0.5)

    @torch.no_grad()
    def _teacher_logits(self, teacher, problems, completions, feedbacks):
        texts = [
            f"### Problem\n{p.prompt}\n### Student Attempt\n{c}\n### Execution Feedback\n{f}"
            for p, c, f in zip(problems, completions, feedbacks)
        ]
        return teacher.model(**teacher.encode(texts)).logits

    def loss(self, model: BaseModel, teacher: BaseModel,
             problems: list[CodeProblem], completions: list[str],
             feedbacks: list[str]) -> tuple[torch.Tensor, dict]:
        s_enc = model.encode([f"{p.format_prompt()}\n{c}" for p, c in zip(problems, completions)])
        s_logits = model.model(**s_enc).logits / self.temp_s
        t_logits = self._teacher_logits(teacher, problems, completions, feedbacks)

        L = min(s_logits.size(1), t_logits.size(1))
        s_lp = F.log_softmax(s_logits[:, :L, :], dim=-1)
        t_lp = F.log_softmax(t_logits[:, -L:, :] / self.temp_t, dim=-1)
        kl = F.kl_div(t_lp, s_lp, log_target=True, reduction="batchmean")
        return self.kl_weight * kl, {"sdpo_kl": kl.item()}