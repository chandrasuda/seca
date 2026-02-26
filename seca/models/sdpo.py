"""SDPO — execution-feedback-conditioned self-distillation.

Teacher sees: prompt + (optional correct solution) + feedback + instruction.
Assistant output y_i is at the end. Three cases based on pass/fail.
KL is computed over completion positions only; teacher logits are detached.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from seca.models.base import BaseModel
from seca.data.problem import CodeProblem
from seca.sandbox.executor import FeedbackBundle


def _kl_divergence_topk(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    topk: int = 20,
) -> torch.Tensor:
    """Sparse KL(P || Q) using top-K tokens under student; tail term for rest."""
    student_probs = student_log_probs.exp()

    topk_probs, topk_indices = student_probs.topk(
        min(topk, student_log_probs.size(-1)), dim=-1
    )

    teacher_probs_at_topk = torch.gather(
        teacher_log_probs.exp(), -1, topk_indices
    )

    p_tail = (1.0 - topk_probs.sum(dim=-1)).clamp(min=1e-10)
    q_tail = (1.0 - teacher_probs_at_topk.sum(dim=-1)).clamp(min=1e-10)

    s_lp_at = student_log_probs.gather(-1, topk_indices)
    t_lp_at = teacher_log_probs.gather(-1, topk_indices)
    kl_topk = (topk_probs * (s_lp_at - t_lp_at)).sum(dim=-1)
    kl_tail = p_tail * (p_tail.log() - q_tail.log())

    kl_per_pos = kl_topk + kl_tail
    return kl_per_pos.mean()


def _build_teacher_context(
    problem: CodeProblem,
    completion: str,
    feedback_summary: str,
    passed: bool,
    passing_completion: str | None,
) -> str:
    """Build teacher context per SDPO paper Table 2: three cases."""
    prompt = problem.format_prompt()

    # SDPO Table 2: User block with optional solution + feedback
    parts = [f"User:\n{prompt}\n\n"]
    if passed:
        parts.append(f"Correct solution (from your successful attempt):\n{completion}\n\n")
    elif passing_completion is not None:
        parts.append(f"Correct solution (from your successful attempt):\n{passing_completion}\n\n")
        parts.append(
            f"The following is feedback from your unsuccessful earlier attempt: {feedback_summary}\n\n"
        )
    else:
        parts.append(
            f"The following is feedback from your unsuccessful earlier attempt: {feedback_summary}\n\n"
        )
    parts.append("Correctly solve the original question.\n\nAssistant:\n")
    return "".join(parts)


class SDPOOperator:
    def __init__(self, cfg: dict):
        self.temp_s = cfg.get("temperature_student", 1.0)
        self.temp_t = cfg.get("temperature_teacher", 0.7)
        self.kl_weight = cfg.get("kl_weight", 0.5)
        self.topk = cfg.get("topk", 20)

    def loss(self, model: BaseModel, teacher: BaseModel,
             problems: list[CodeProblem], completions: list[str],
             feedback_bundles: list[FeedbackBundle],
             ) -> tuple[torch.Tensor, dict]:
        passed_indices = [i for i, fb in enumerate(feedback_bundles) if fb.all_passed]
        passing_completion = None
        if passed_indices:
            passing_completion = completions[passed_indices[0]]

        losses = []
        for i, (p, c, fb) in enumerate(zip(problems, completions, feedback_bundles)):
            prompt = p.format_prompt()
            student_input = f"{prompt}\n{c}"

            teacher_context = _build_teacher_context(
                problem=p,
                completion=c,
                feedback_summary=fb.summary,
                passed=fb.all_passed,
                passing_completion=passing_completion if not fb.all_passed else None,
            )
            teacher_input = teacher_context + c

            s_enc = model.encode([student_input])
            s_logits = model.model(**s_enc).logits / self.temp_s

            with torch.no_grad():
                t_enc = teacher.encode([teacher_input])
                t_logits = teacher.model(**t_enc).logits / self.temp_t

            # Completion positions: last len(completion) tokens
            prompt_enc = model.encode([prompt + "\n"])
            prompt_len = prompt_enc["input_ids"].shape[1]
            total_len = s_enc["input_ids"].shape[1]
            completion_len = total_len - prompt_len
            if completion_len <= 0:
                continue

            s_lp = F.log_softmax(s_logits[:, -completion_len:, :], dim=-1)
            t_lp = F.log_softmax(t_logits[:, -completion_len:, :], dim=-1)

            kl = _kl_divergence_topk(s_lp, t_lp, topk=self.topk)
            losses.append(kl)

        if not losses:
            return torch.tensor(0.0, device=model.device), {"sdpo_kl": 0.0}

        total_kl = sum(losses) / len(losses)
        return self.kl_weight * total_kl, {"sdpo_kl": total_kl.item()}
