"""SDFT — demo-conditioned self-distillation (Shenfeld et al., 2026).

Teacher sees: prompt + gold solution + student attempt.
KL is computed over completion positions only; teacher logits are detached.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from seca.models.base import BaseModel
from seca.data.problem import CodeProblem


def _kl_divergence_topk(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    topk: int = 100,
) -> torch.Tensor:
    """Sparse KL(P || Q) using top-K tokens under student; tail term for rest."""
    student_probs = student_log_probs.exp()

    # Get top-K indices under student at each position
    topk_probs, topk_indices = student_probs.topk(
        min(topk, student_log_probs.size(-1)), dim=-1
    )

    # Gather teacher probs at same indices
    teacher_probs_at_topk = torch.gather(
        teacher_log_probs.exp(), -1, topk_indices
    )

    # Tail: mass outside top-K
    p_tail = (1.0 - topk_probs.sum(dim=-1)).clamp(min=1e-10)
    q_tail = (1.0 - teacher_probs_at_topk.sum(dim=-1)).clamp(min=1e-10)

    # KL over top-K: P * (log P - log Q)
    s_lp_at = student_log_probs.gather(-1, topk_indices)
    t_lp_at = teacher_log_probs.gather(-1, topk_indices)
    kl_topk = (topk_probs * (s_lp_at - t_lp_at)).sum(dim=-1)

    # Tail term
    kl_tail = p_tail * (p_tail.log() - q_tail.log())

    kl_per_pos = kl_topk + kl_tail
    return kl_per_pos.mean()


class SDFTOperator:
    def __init__(self, cfg: dict):
        self.temp_s = cfg.get("temperature_student", 1.0)
        self.temp_t = cfg.get("temperature_teacher", 0.7)
        self.kl_weight = cfg.get("kl_weight", 0.5)
        self.topk = cfg.get("topk", 100)

    @torch.no_grad()
    def _teacher_logits(self, teacher: BaseModel, problems: list[CodeProblem],
                        completions: list[str]) -> torch.Tensor:
        # Teacher sees same prompt format as student + gold + student attempt
        texts = [
            f"{p.format_prompt()}\n### Optimal Solution\n{p.gold_solution}\n### Student Attempt\n{c}"
            for p, c in zip(problems, completions)
        ]
        return teacher.model(**teacher.encode(texts)).logits

    def loss(self, model: BaseModel, teacher: BaseModel,
             problems: list[CodeProblem], completions: list[str],
             ) -> tuple[torch.Tensor, dict]:
        losses = []
        for p, c in zip(problems, completions):
            prompt = p.format_prompt()

            s_enc = model.encode([f"{prompt}\n{c}"])
            s_logits = model.model(**s_enc).logits / self.temp_s
            t_logits = self._teacher_logits(teacher, [p], [c]) / self.temp_t

            # Completion positions: last len(completion) tokens
            prompt_enc = model.encode([prompt + "\n"])
            prompt_len = prompt_enc["input_ids"].shape[1]
            total_len = s_enc["input_ids"].shape[1]
            completion_len = total_len - prompt_len
            if completion_len <= 0:
                continue

            # Use last completion_len positions for both
            s_lp = F.log_softmax(s_logits[:, -completion_len:, :], dim=-1)
            t_lp = F.log_softmax(t_logits[:, -completion_len:, :], dim=-1)

            kl = _kl_divergence_topk(s_lp, t_lp, topk=self.topk)
            losses.append(kl)

        if not losses:
            return torch.tensor(0.0, device=model.device), {"sdft_kl": 0.0}

        total_kl = sum(losses) / len(losses)
        return self.kl_weight * total_kl, {"sdft_kl": total_kl.item()}
