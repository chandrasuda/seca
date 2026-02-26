"""SDPO — execution-feedback-conditioned self-distillation.

Teacher sees: prompt + (optional correct solution) + feedback + instruction.
Assistant output y_i is at the end. Three cases based on pass/fail.
KL is computed over completion positions only; teacher logits are detached.

Teacher context logic (mirrors SDPO paper Table 2 +
submodule's dont_reprompt_on_self_success=True default):

  Case A: y_i passed, another passing completion y_j exists
          → teacher sees {prompt} + solution=y_j                  (no feedback; y_i already succeeded)
  Case B: y_i passed, no other passing completion
          → teacher sees {prompt} + solution=y_i                  (own solution as demo; low KL signal)
  Case C: y_i failed, some y_j from the group passed
          → teacher sees {prompt} + solution=y_j + feedback_i
  Case D: all completions failed
          → teacher sees {prompt} + feedback_i                    (feedback only)

In cases A & B the teacher is very confident about y_i
(it knows the solution), so KL is low and gradient is small —
correct, since the model is already producing correct completions.
Cases C & D carry the main learning signal.
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
    """Sparse KL(P_student || P_teacher) using top-K tokens under student; tail term for rest."""
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

    return (kl_topk + kl_tail).mean()


def _build_teacher_context(
    problem: CodeProblem,
    completion: str,
    feedback_summary: str,
    passed: bool,
    demo_completion: str | None,
    feedback_truncate_chars: int | None = None,
    demo_truncate_chars: int | None = None,
) -> str:
    """Build teacher context per SDPO paper Table 2.

    demo_completion: the best available correct solution to show the teacher.
        - For passed samples: a DIFFERENT passing completion (preferred) or own.
        - For failed samples: any passing completion from the group, or None.

    feedback_truncate_chars: if set, truncate feedback_summary to this many chars.
    demo_truncate_chars: if set, truncate demo_completion to this many chars.
    """
    feedback = feedback_summary
    if feedback_truncate_chars is not None and len(feedback) > feedback_truncate_chars:
        feedback = feedback[:feedback_truncate_chars] + "\n  [...truncated]"
    demo = demo_completion
    if demo_truncate_chars is not None and demo is not None and len(demo) > demo_truncate_chars:
        demo = demo[:demo_truncate_chars] + "\n# [...truncated]"
    prompt = problem.format_prompt()
    parts = [f"User:\n{prompt}\n\n"]

    if passed:
        if demo is not None and demo != completion:
            # Case A: own attempt succeeded; show a different correct solution as the demo.
            # Teacher sees the alternative approach → richer signal than just own solution.
            parts.append(
                f"Correct solution (from another successful attempt):\n{demo}\n\n"
            )
        else:
            # Case B: own attempt succeeded and it is the only solution.
            # Teacher sees own solution — will be very confident; KL ≈ 0, minimal gradient.
            parts.append(
                f"Correct solution (from your successful attempt):\n{completion}\n\n"
            )
    elif demo is not None:
        # Case C: failed, but another attempt in the group passed.
        # Show that correct solution + feedback about this failing attempt.
        parts.append(
            f"Correct solution (from a successful attempt):\n{demo}\n\n"
        )
        parts.append(
            f"The following is feedback from your unsuccessful earlier attempt:\n{feedback}\n\n"
        )
    else:
        # Case D: all attempts failed — feedback only.
        parts.append(
            f"The following is feedback from your unsuccessful earlier attempt:\n{feedback}\n\n"
        )

    parts.append("Correctly solve the original question.\n\nAssistant:\n")
    return "".join(parts)


class SDPOOperator:
    def __init__(self, cfg: dict):
        self.temp_s = cfg.get("temperature_student", 1.0)
        self.temp_t = cfg.get("temperature_teacher", 1.0)  # paper uses no temperature scaling
        self.kl_weight = cfg.get("kl_weight", 0.5)
        self.topk = cfg.get("topk", 20)

    def loss(
        self,
        model: BaseModel,
        teacher: BaseModel,
        problems: list[CodeProblem],
        completions: list[str],
        feedback_bundles: list[FeedbackBundle],
    ) -> tuple[torch.Tensor, dict]:
        # Index every completion that passed all tests.
        passed_indices = [i for i, fb in enumerate(feedback_bundles) if fb.all_passed]

        losses = []
        for i, (p, c, fb) in enumerate(zip(problems, completions, feedback_bundles)):
            prompt = p.format_prompt()
            student_input = f"{prompt}\n{c}"

            # Per-sample demo selection (mirrors dont_reprompt_on_self_success=True):
            #   passed  → prefer a DIFFERENT passing completion; fall back to own
            #   failed  → use first passing completion in the group (any index)
            if fb.all_passed:
                other_passed = [j for j in passed_indices if j != i]
                demo = completions[other_passed[0]] if other_passed else None
            else:
                demo = completions[passed_indices[0]] if passed_indices else None

            # Ensure teacher context fits so completion tokens aren't truncated away.
            # Truncate feedback and demo iteratively until context + completion fits.
            min_completion_tokens = 64
            max_ctx_tokens = model.max_len - min_completion_tokens
            feedback_cap = len(fb.summary)
            demo_cap = len(demo) if demo else 0
            for _ in range(20):
                teacher_context = _build_teacher_context(
                    problem=p,
                    completion=c,
                    feedback_summary=fb.summary,
                    passed=fb.all_passed,
                    demo_completion=demo,
                    feedback_truncate_chars=feedback_cap if feedback_cap < len(fb.summary) else None,
                    demo_truncate_chars=demo_cap if demo_cap < len(demo) and demo else None,
                )
                ctx_enc = model.encode([teacher_context])
                n_ctx = ctx_enc["input_ids"].shape[1]
                if n_ctx <= max_ctx_tokens:
                    break
                # Truncate: prefer feedback (Cases C/D), then demo (Cases A/B/C)
                excess = n_ctx - max_ctx_tokens
                trim_tokens = excess + 100
                trim_chars = min(trim_tokens * 4, 2000)
                if fb.summary and feedback_cap > 200:
                    feedback_cap = max(200, feedback_cap - trim_chars)
                elif demo and demo_cap > 200:
                    demo_cap = max(200, demo_cap - trim_chars)
                else:
                    break

            teacher_input = teacher_context + c

            # Student forward (gradients flow here)
            s_enc = model.encode([student_input])
            s_logits = model.model(**s_enc).logits / self.temp_s

            # Teacher forward — fully inside no_grad; temp division included
            with torch.no_grad():
                t_enc = teacher.encode([teacher_input])
                t_logits = teacher.model(**t_enc).logits / self.temp_t

            # Student completion length
            s_prompt_enc = model.encode([prompt + "\n"])
            s_prompt_len = s_prompt_enc["input_ids"].shape[1]
            s_completion_len = s_enc["input_ids"].shape[1] - s_prompt_len

            # Teacher completion length — computed independently to guard against
            # BPE boundary effects (teacher prefix is much longer than student prompt)
            t_prefix_enc = teacher.encode([teacher_context])
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
            # Return loss with grad_fn so backward() doesn't raise. Case D with
            # long feedback can truncate completion; dummy has zero gradient.
            dummy = model.model.get_input_embeddings().weight[0, 0] * 0.0
            return dummy, {"sdpo_kl": 0.0}

        total_kl = sum(losses) / len(losses)
        return self.kl_weight * total_kl, {"sdpo_kl": total_kl.item()}
