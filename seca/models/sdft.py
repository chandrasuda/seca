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
    """Sparse KL(P_student || P_teacher) using top-K tokens under student; tail term for rest."""
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

    return (kl_topk + kl_tail).mean()


class SDFTOperator:
    def __init__(self, cfg: dict):
        self.temp_s = cfg.get("temperature_student", 1.0)
        self.temp_t = cfg.get("temperature_teacher", 1.0)
        self.kl_weight = cfg.get("kl_weight", 0.5)
        self.topk = cfg.get("topk", 100)

    def loss(
        self,
        model: BaseModel,
        teacher: BaseModel,
        problems: list[CodeProblem],
        completions: list[str],
    ) -> tuple[torch.Tensor, dict]:
        losses = []
        for p, c in zip(problems, completions):
            prompt = p.format_prompt()
            student_input = f"{prompt}\n{c}"
            teacher_prefix = (
                f"{prompt}\n### Optimal Solution\n{p.gold_solution}\n### Student Attempt\n"
            )
            teacher_input = teacher_prefix + c

            # Student forward (gradients flow here)
            s_enc = model.encode([student_input])
            s_logits = model.model(**s_enc).logits / self.temp_s

            # Teacher forward — keep everything inside no_grad so the temp division
            # does not create an unnecessary graph node
            with torch.no_grad():
                t_enc = teacher.encode([teacher_input])
                t_logits = teacher.model(**t_enc).logits / self.temp_t

            # Student completion length: tokens in c appended after the student prompt
            s_prompt_enc = model.encode([f"{prompt}\n"])
            s_prompt_len = s_prompt_enc["input_ids"].shape[1]
            s_completion_len = s_enc["input_ids"].shape[1] - s_prompt_len

            # Teacher completion length: computed independently to guard against BPE
            # boundary effects — the same text can tokenise differently depending on
            # what precedes it, so we never assume the student count equals the teacher count.
            t_prefix_enc = teacher.encode([teacher_prefix])
            t_prefix_len = t_prefix_enc["input_ids"].shape[1]
            t_completion_len = t_enc["input_ids"].shape[1] - t_prefix_len

            completion_len = min(s_completion_len, t_completion_len)
            if completion_len <= 0:
                continue

            s_lp = F.log_softmax(s_logits[:, -completion_len:, :], dim=-1)
            t_lp = F.log_softmax(t_logits[:, -completion_len:, :], dim=-1)

            kl = _kl_divergence_topk(s_lp, t_lp, topk=self.topk)
            losses.append(kl)

        if not losses:
            return torch.tensor(0.0, device=model.device), {"sdft_kl": 0.0}

        total_kl = sum(losses) / len(losses)
        return self.kl_weight * total_kl, {"sdft_kl": total_kl.item()}
