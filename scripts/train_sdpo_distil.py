#!/usr/bin/env python3
"""SDPO training on APPS (Shenfeld et al. 2026, arxiv:2601.20802).

Standalone implementation faithful to the SDPO submodule without requiring
the verl distributed framework.  Uses HuggingFace for model loading + training
and HF generate() for on-policy rollouts.

Algorithm (Algorithm 1 from the paper):
  - G rollouts per problem (default 4) sampled on-policy from the student
  - Each rollout is executed in the APPS sandbox to get pass/fail + feedback
  - Teacher context = reprompted prompt (original problem + passing solution
    from the rollout group) — teacher never sees the student's own attempt
  - Loss = KL(student ∥ teacher) via top-K logit distillation (K=20)
  - IS ratio clipping (is_clip=2.0) for training stability
  - Teacher regularization = trust-region (best performing, +14.5% vs. none)

Four teacher cases per completion in a rollout group:
  A) Passed  + another passing completion exists  → teacher sees alt solution
  B) Passed  + no other passing completion        → skip (cant use self)
  C) Failed  + group has any passing completion   → teacher sees that solution
  D) Failed  + no passing completion in group     → skip (no signal)

  With --include-env-feedback:
  C) additionally includes execution feedback text
  D) uses feedback-only reprompt (no solution)

Usage (local smoke test):
  python scripts/train_sdpo_distil.py --max-problems 5 --epochs 1

Usage (Modal A100):
  modal run scripts/modal_sdpo_distil.py --task training --max-problems 100 --epochs 1
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from seca.data.apps import load_apps
from seca.sandbox.executor import execute_code
from seca.utils.tokenizer import make_no_thinking_tokenizer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exact reprompt templates from the SDPO submodule
# (SDPO/verl/workers/config/actor.py, lines 77-90)
# ---------------------------------------------------------------------------
_REPROMPT_TEMPLATE = (
    "{prompt}{solution}{feedback}\n\n"
    "Correctly solve the original question. Your response must contain ONLY executable code "
    "enclosed between <start_code> and <end_code> tokens.\n"
)
_SOLUTION_TEMPLATE = (
    "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
)
_FEEDBACK_TEMPLATE = (
    "\nThe following is feedback from your unsuccessful earlier attempt:\n\n"
    "{feedback_raw}\n\n"
)


# ---------------------------------------------------------------------------
# Teacher reprompt construction
# ---------------------------------------------------------------------------

def build_teacher_prompt(
    problem_text: str,
    completion: str,
    passed: bool,
    feedback: str,
    group_passing_completions: list[str],
    dont_reprompt_on_self_success: bool = True,
    include_environment_feedback: bool = False,
) -> Optional[str]:
    """Build teacher reprompt for one completion in a rollout group.

    Faithfully replicates _maybe_build_self_distillation_batch() from
    SDPO/verl/trainer/ppo/ray_trainer.py.

    Returns None when there is no distillation signal (cases B and D without
    environment feedback).
    """
    solution_section = ""
    feedback_section = ""

    if passed:
        # Exclude self when dont_reprompt_on_self_success=True
        if dont_reprompt_on_self_success:
            candidates = [c for c in group_passing_completions if c != completion]
        else:
            candidates = list(group_passing_completions)

        if not candidates:
            return None  # Case B: no alt solution available

        # Case A: teacher sees first alt passing solution
        solution_section = _SOLUTION_TEMPLATE.format(
            successful_previous_attempt=candidates[0]
        )
    else:
        # Failing completion
        if group_passing_completions:
            # Case C: teacher sees first passing solution from the group
            solution_section = _SOLUTION_TEMPLATE.format(
                successful_previous_attempt=group_passing_completions[0]
            )
            if include_environment_feedback and feedback:
                feedback_section = _FEEDBACK_TEMPLATE.format(feedback_raw=feedback)
        else:
            # Case D: no passing solution in the group
            if include_environment_feedback and feedback:
                feedback_section = _FEEDBACK_TEMPLATE.format(feedback_raw=feedback)
            else:
                return None  # No signal at all

    if not solution_section and not feedback_section:
        return None

    return _REPROMPT_TEMPLATE.format(
        prompt=problem_text,
        solution=solution_section,
        feedback=feedback_section,
    )


# ---------------------------------------------------------------------------
# Log-prob extraction helpers
# ---------------------------------------------------------------------------

def get_response_logprobs(
    logits: torch.Tensor,    # (1, seq_len, vocab)
    input_ids: torch.Tensor, # (1, seq_len)
    prompt_len: int,
    topk: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract per-token and top-K log probs for response tokens.

    For a sequence [prompt_tokens][response_tokens], logit at position i
    predicts token i+1 (autoregressive shift).

    Returns:
        token_lp:  (resp_len,)    log p of each actual response token
        topk_lp:   (resp_len, K)  log p of top-K tokens at each position
        topk_idx:  (resp_len, K)  token indices of top-K
    """
    seq_len = input_ids.shape[1]
    resp_len = seq_len - prompt_len
    if resp_len <= 0:
        raise ValueError(
            f"No response tokens: seq_len={seq_len}, prompt_len={prompt_len}"
        )

    # Logits that predict each response token (shifted by 1)
    resp_logits = logits[0, prompt_len - 1 : seq_len - 1, :].float()  # (resp_len, vocab)
    resp_log_probs = F.log_softmax(resp_logits, dim=-1)                # (resp_len, vocab)

    # Per-token log probs (used for IS correction)
    resp_tokens = input_ids[0, prompt_len:].unsqueeze(1)               # (resp_len, 1)
    token_lp = resp_log_probs.gather(1, resp_tokens).squeeze(1)        # (resp_len,)

    # Top-K log probs and their indices (student defines the K positions)
    K = min(topk, resp_log_probs.shape[-1])
    topk_lp, topk_idx = torch.topk(resp_log_probs, K, dim=-1)         # (resp_len, K)

    return token_lp, topk_lp, topk_idx


def get_teacher_topk_at_student_indices(
    teacher_logits: torch.Tensor,    # (1, seq_len, vocab)
    teacher_input_ids: torch.Tensor, # (1, seq_len) — teacher sequence
    teacher_prompt_len: int,
    student_topk_idx: torch.Tensor,  # (resp_len, K) — from student
) -> torch.Tensor:
    """Get teacher log probs at the student's top-K token positions.

    Teacher and student have different prompts but the same response tokens.
    student_topk_idx determines which K vocabulary positions to compare,
    ensuring both distributions are evaluated on the same support.
    """
    seq_len = teacher_input_ids.shape[1]
    resp_len = seq_len - teacher_prompt_len

    teacher_resp_logits = teacher_logits[0, teacher_prompt_len - 1 : seq_len - 1, :].float()
    teacher_log_probs = F.log_softmax(teacher_resp_logits, dim=-1)     # (resp_len, vocab)

    # Gather at student's top-K indices
    teacher_topk_lp = teacher_log_probs.gather(1, student_topk_idx)   # (resp_len, K)
    return teacher_topk_lp


# ---------------------------------------------------------------------------
# SDPO distillation loss
# (faithful translation of compute_self_distillation_loss from core_algos.py)
# ---------------------------------------------------------------------------

def _add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    """Append log-probability of tail (residual mass) to top-K log probs.

    log(1 - Σpᵢ) computed stably via logsumexp + expm1.
    """
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True).clamp(max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def compute_sdpo_loss(
    student_topk_lp: torch.Tensor,          # (resp_len, K)
    teacher_topk_lp: torch.Tensor,          # (resp_len, K) same K positions
    old_token_lp: Optional[torch.Tensor],   # (resp_len,) from rollout
    student_token_lp: torch.Tensor,         # (resp_len,) current student
    alpha: float = 1.0,
    is_clip: Optional[float] = 2.0,
    add_tail: bool = True,
) -> torch.Tensor:
    """SDPO top-K distillation loss with optional IS correction.

    alpha=1.0  → reverse KL: KL(student ∥ teacher)   [paper default]
    alpha=0.0  → forward  KL: KL(teacher ∥ student)
    0<alpha<1  → generalised JSD interpolation
    """
    # 1. Add tail bucket to top-K log probs (for numerical completeness)
    if add_tail:
        s = _add_tail_bucket(student_topk_lp)  # (resp_len, K+1)
        t = _add_tail_bucket(teacher_topk_lp)  # (resp_len, K+1)
    else:
        # Renormalise top-K to sum to 1
        s = student_topk_lp - torch.logsumexp(student_topk_lp, dim=-1, keepdim=True)
        t = teacher_topk_lp - torch.logsumexp(teacher_topk_lp, dim=-1, keepdim=True)

    # 2. KL divergence
    # F.kl_div(input=log_q, target=log_p, log_target=True) = KL(p‖q) = Σ p*(log_p - log_q)
    if alpha == 1.0:
        # KL(student ‖ teacher): q=teacher, p=student
        kl = F.kl_div(t, s, reduction="none", log_target=True)
    elif alpha == 0.0:
        # KL(teacher ‖ student): q=student, p=teacher
        kl = F.kl_div(s, t, reduction="none", log_target=True)
    else:
        # Generalised JSD (from core_algos.py lines 1147-1160)
        a = torch.tensor(alpha, dtype=s.dtype, device=s.device)
        mix = torch.logsumexp(
            torch.stack([s + torch.log(1 - a), t + torch.log(a)]), dim=0
        )
        kl_student = F.kl_div(mix, s, reduction="none", log_target=True)
        kl_teacher = F.kl_div(mix, t, reduction="none", log_target=True)
        kl = torch.lerp(kl_student, kl_teacher, a)

    per_token_loss = kl.sum(dim=-1)  # (resp_len,)

    # 3. IS ratio correction (clipped importance sampling, is_clip=2.0 from paper)
    if is_clip is not None and old_token_lp is not None:
        neg_approx_kl = (student_token_lp - old_token_lp).detach().clamp(-20.0, 20.0)
        ratio = torch.exp(neg_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    return per_token_loss.mean()


# ---------------------------------------------------------------------------
# EMA teacher update
# ---------------------------------------------------------------------------

def update_ema(ema_model: torch.nn.Module, student: torch.nn.Module, rate: float) -> None:
    """EMA update: φ ← (1 - rate) * φ + rate * θ  (applied after each optimizer step)."""
    with torch.no_grad():
        for ema_p, stu_p in zip(ema_model.parameters(), student.parameters()):
            ema_p.data.mul_(1.0 - rate).add_(stu_p.data.to(ema_p.device), alpha=rate)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDPO training (arxiv:2601.20802) on APPS"
    )
    # Model
    p.add_argument(
        "--model", default="Qwen/Qwen3-1.7B",
        help="HF model name / local path (default: Qwen/Qwen3-1.7B)",
    )
    # Data
    p.add_argument("--split", default="train",
                   help="APPS split: train | test | all (default: train)")
    p.add_argument("--difficulty", default=None,
                   help="APPS difficulty filter: introductory | interview | competition")
    p.add_argument("--max-problems", type=int, default=None,
                   help="Cap on number of APPS problems (None = all)")
    p.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Optional JSONL path in APPS-compatible format (overrides split source)",
    )
    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Learning rate (paper: 1e-5 for LCBv6 rich-feedback)")
    p.add_argument("--num-generations", type=int, default=4,
                   help="G rollouts per problem (paper: 4 for LCBv6)")
    p.add_argument("--max-prompt-length", type=int, default=None,
                   help="Max student prompt tokens (default: None = no truncation)")
    p.add_argument("--max-completion-length", type=int, default=2048,
                   help="Max completion tokens per rollout")
    p.add_argument("--max-teacher-prompt-length", type=int, default=4096,
                   help="Max teacher (reprompted) context tokens")
    p.add_argument("--warmup-steps", type=int, default=10,
                   help="LR warmup steps (paper: 10)")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # SDPO-specific
    p.add_argument(
        "--teacher-regularization", default="trust_region",
        choices=["trust_region", "ema"],
        help="Teacher regularization: trust_region=best (+14.5%%), ema=simpler",
    )
    p.add_argument(
        "--teacher-update-rate", type=float, default=0.01,
        help="mix_coef for trust-region OR ema rate (paper: 0.01)",
    )
    p.add_argument(
        "--distillation-topk", type=int, default=20,
        help="K for top-K logit distillation (paper: K=20 for rich feedback)",
    )
    p.add_argument(
        "--alpha", type=float, default=1.0,
        help="KL direction: 1.0=reverse KL [paper default for rich feedback]",
    )
    p.add_argument(
        "--is-clip", type=float, default=2.0,
        help="IS ratio clip threshold (paper: 2.0)",
    )
    p.add_argument(
        "--include-env-feedback", action="store_true", default=False,
        help="Include execution feedback in teacher prompt (paper default: False)",
    )
    p.add_argument(
        "--allow-self-reprompt", action="store_true", default=False,
        help="Allow reprompting on self success (paper default: False = dont_reprompt=True)",
    )
    p.add_argument("--exec-timeout", type=float, default=10.0,
                   help="Sandbox execution timeout per test case (seconds)")
    # Output
    p.add_argument("--output-dir", default="checkpoints/sdpo_distil",
                   help="Output directory for HF-format checkpoints")
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paper setting: dont_reprompt_on_self_success=True unless --allow-self-reprompt
    dont_reprompt = not args.allow_self_reprompt

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    log.info(
        "Loading APPS (split=%s difficulty=%s max=%s)...",
        args.split, args.difficulty, args.max_problems,
    )
    problems = load_apps(
        split=args.split,
        difficulty=args.difficulty,
        max_problems=args.max_problems,
        data_file=args.data_file,
    )
    log.info("Loaded %d APPS problems with gold solutions + test cases", len(problems))

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    log.info("Loading student model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    make_no_thinking_tokenizer(tokenizer)  # disable <think> for Qwen3

    # Reference model: frozen initial weights for trust-region teacher
    # Also serves as the EMA base for EMA mode
    log.info("Loading frozen ref model: %s", args.model)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    for param in ref_model.parameters():
        param.requires_grad_(False)
    ref_model.eval()

    # EMA teacher model (only needed for EMA regularization)
    ema_model: Optional[torch.nn.Module] = None
    if args.teacher_regularization == "ema":
        log.info("Creating EMA teacher model (copy of initial weights)")
        ema_model = copy.deepcopy(ref_model)  # same frozen initial weights
        ema_model.eval()

    # ------------------------------------------------------------------
    # Optimizer and cosine LR schedule
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    # One optimizer step per problem per epoch
    total_steps = args.epochs * len(problems)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    has_wandb_key = bool(os.environ.get("WANDB_API_KEY", "").strip())
    use_wandb = has_wandb_key and not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project="seca-sdpo", name="sdpo-apps", config=vars(args))
    else:
        log.info("W&B disabled (set WANDB_API_KEY to enable)")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    log.info(
        "Starting SDPO training: %d problems | %d epochs | G=%d | "
        "teacher=%s(rate=%.3f) | K=%d | alpha=%.1f | is_clip=%.1f",
        len(problems), args.epochs, args.num_generations,
        args.teacher_regularization, args.teacher_update_rate,
        args.distillation_topk, args.alpha, args.is_clip,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(args.epochs):
        log.info("=== Epoch %d/%d ===", epoch + 1, args.epochs)
        epoch_losses: list[float] = []
        epoch_pass_rates: list[float] = []
        epoch_tests_passed: list[int] = []
        epoch_tests_total: list[int] = []

        for prob_idx, problem in enumerate(problems):
            log.info("Processing problem %d/%d ...", prob_idx + 1, len(problems))

            # ---- 1. Format student prompt ----
            student_prompt_text = problem.format_prompt()
            student_messages = [{"role": "user", "content": student_prompt_text}]
            student_prompt_formatted = tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            student_tok_kwargs = {"return_tensors": "pt"}
            if args.max_prompt_length is not None:
                student_tok_kwargs.update(
                    {"truncation": True, "max_length": args.max_prompt_length}
                )
            student_prompt_ids = tokenizer(
                student_prompt_formatted, **student_tok_kwargs
            ).input_ids.to(device)
            student_prompt_len = student_prompt_ids.shape[1]

            # ---- 2. Rollout: generate G completions one at a time ----
            model.eval()
            gen_ids_list: list[torch.Tensor] = []
            with torch.no_grad():
                for g_idx in range(args.num_generations):
                    log.info("  Generating completion %d/%d ...", g_idx + 1, args.num_generations)
                    out = model.generate(
                        student_prompt_ids,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        max_new_tokens=args.max_completion_length,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    gen_ids_list.append(out[0, student_prompt_len:])

            completions = [
                tokenizer.decode(g, skip_special_tokens=True)
                for g in gen_ids_list
            ]
            completion_lens = [len(g) for g in gen_ids_list]
            log.info(
                "  Generated %d completions (tokens: %s), running sandbox ...",
                len(completions), completion_lens,
            )

            # ---- 3. Execute completions in APPS sandbox ----
            fb_list = []
            _LOG_SNIP = 600  # chars to show for raw/extracted code in logs

            for c_idx, c in enumerate(completions):
                log.info("  Sandbox %d/%d ...", c_idx + 1, len(completions))
                fb = execute_code(c, problem, timeout=args.exec_timeout)
                fb_list.append(fb)
                status = "PASS" if fb.all_passed else "FAIL"
                log.info("    %s (pass_rate=%.3f, %d tests)", status, fb.pass_rate, len(fb.results))

                if not fb.all_passed:
                    raw_preview = c[: _LOG_SNIP] + ("..." if len(c) > _LOG_SNIP else "")
                    ext_preview = (fb.extracted_code or "")[: _LOG_SNIP]
                    if len(fb.extracted_code or "") > _LOG_SNIP:
                        ext_preview += "..."
                    log.info("    [FAIL] Raw response (first %d chars):\n%s", _LOG_SNIP, raw_preview)
                    log.info("    [FAIL] Extracted code executed:\n%s", ext_preview or "(empty)")
                    if fb.first_failure_stderr:
                        log.info("    [FAIL] Stderr: %s", fb.first_failure_stderr[:800])
                    if fb.first_failure_stdout and not fb.first_failure_stderr:
                        log.info("    [FAIL] Stdout (wrong output): %s", fb.first_failure_stdout[:500])

            n_passed = sum(1 for fb in fb_list if fb.all_passed)
            mean_pass_rate = sum(fb.pass_rate for fb in fb_list) / len(fb_list)
            tests_passed = sum(
                sum(1 for r in fb.results if r.passed) for fb in fb_list
            )
            tests_total = sum(len(fb.results) for fb in fb_list)
            test_pass_rate = (tests_passed / tests_total) if tests_total else 0.0
            epoch_pass_rates.append(mean_pass_rate)
            epoch_tests_passed.append(tests_passed)
            epoch_tests_total.append(tests_total)
            log.info(
                "  Sandbox done: %d/%d completions fully passed, "
                "completion_mean_pass_rate=%.3f, test_pass_rate=%d/%d=%.3f",
                n_passed, len(fb_list), mean_pass_rate,
                tests_passed, tests_total, test_pass_rate,
            )

            # ---- 4. Get old log probs for IS correction ----
            model.eval()
            old_token_lps: list[Optional[torch.Tensor]] = []
            log.info("  Computing old log probs for IS correction ...")
            with torch.no_grad():
                for gen_ids in gen_ids_list:
                    full_ids = torch.cat(
                        [student_prompt_ids, gen_ids.unsqueeze(0)], dim=1
                    )
                    outputs = model(full_ids)
                    try:
                        old_lp, _, _ = get_response_logprobs(
                            outputs.logits, full_ids,
                            student_prompt_len, topk=args.distillation_topk,
                        )
                        old_token_lps.append(old_lp)
                    except ValueError:
                        old_token_lps.append(None)

            # ---- 5. Build teacher reprompts (SDPO reprompting mechanism) ----
            passing_completions = [
                c for c, fb in zip(completions, fb_list) if fb.all_passed
            ]
            teacher_prompts = [
                build_teacher_prompt(
                    problem_text=student_prompt_text,
                    completion=c,
                    passed=fb.all_passed,
                    feedback=fb.summary,
                    group_passing_completions=passing_completions,
                    dont_reprompt_on_self_success=dont_reprompt,
                    include_environment_feedback=args.include_env_feedback,
                )
                for c, fb in zip(completions, fb_list)
            ]
            n_valid = sum(1 for tp in teacher_prompts if tp is not None)

            if n_valid == 0:
                reason = (
                    "0 passing completions and include_env_feedback=False → "
                    "no distillation signal. Use --include-env-feedback to learn from execution feedback on failures."
                )
                log.info(
                    "  Skipping gradient step: %d valid teacher prompts. %s",
                    n_valid, reason,
                )
                global_step += 1
                epoch_losses.append(0.0)
                continue

            log.info(
                "  Building gradients: %d/%d completions have valid teacher prompts",
                n_valid, len(completions),
            )

            # ---- 6. Training: gradient accumulation over valid completions ----
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            n_accumulated = 0

            for comp_idx, (gen_ids, tp, old_lp) in enumerate(
                zip(gen_ids_list, teacher_prompts, old_token_lps)
            ):
                if tp is None or old_lp is None:
                    continue

                n_accumulated += 1
                log.info("  Backprop completion %d/%d (seq_len=%d) ...", comp_idx + 1, len(completions), gen_ids.shape[0])

                # --- 6a. Student forward on [student_prompt + completion] (with grad) ---
                s_full = torch.cat(
                    [student_prompt_ids, gen_ids.unsqueeze(0)], dim=1
                )
                s_out = model(s_full)
                try:
                    stu_token_lp, stu_topk_lp, stu_topk_idx = get_response_logprobs(
                        s_out.logits, s_full,
                        student_prompt_len, topk=args.distillation_topk,
                    )
                except ValueError:
                    continue

                # --- 6b. Teacher forward on [teacher_prompt + completion] (no grad) ---
                teacher_messages = [{"role": "user", "content": tp}]
                teacher_prompt_formatted = tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                teacher_prompt_ids = tokenizer(
                    teacher_prompt_formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_teacher_prompt_length,
                ).input_ids.to(device)
                teacher_prompt_len = teacher_prompt_ids.shape[1]
                t_full = torch.cat([teacher_prompt_ids, gen_ids.unsqueeze(0)], dim=1)

                with torch.no_grad():
                    if args.teacher_regularization == "trust_region":
                        # Trust-region teacher: logits = lerp(ref, student, mix_coef)
                        # Both are evaluated on teacher (reprompted) input.
                        ref_out = ref_model(t_full)
                        stu_on_teacher = model(t_full)
                        teacher_logits = torch.lerp(
                            ref_out.logits,
                            stu_on_teacher.logits,
                            args.teacher_update_rate,
                        )
                    else:  # EMA
                        ema_out = ema_model(t_full)
                        teacher_logits = ema_out.logits

                tea_topk_lp = get_teacher_topk_at_student_indices(
                    teacher_logits, t_full, teacher_prompt_len, stu_topk_idx
                )

                # --- 6c. Compute and accumulate distillation loss ---
                loss = compute_sdpo_loss(
                    student_topk_lp=stu_topk_lp,
                    teacher_topk_lp=tea_topk_lp,
                    old_token_lp=old_lp.to(device),
                    student_token_lp=stu_token_lp,
                    alpha=args.alpha,
                    is_clip=args.is_clip,
                )
                # Normalize by number of valid completions in this group
                (loss / n_valid).backward()
                total_loss += loss.item() / n_valid

            if total_loss > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # EMA teacher update after each optimizer step
                if args.teacher_regularization == "ema" and ema_model is not None:
                    update_ema(ema_model, model, args.teacher_update_rate)

                log.info(
                    "  Step complete: loss=%.4f (from %d completions), lr=%.2e",
                    total_loss, n_accumulated, scheduler.get_last_lr()[0],
                )
            else:
                log.info("  Step complete: no valid gradients (total_loss=0)")

            global_step += 1
            epoch_losses.append(total_loss)

            # Periodic logging
            if (prob_idx + 1) % 10 == 0 or (prob_idx + 1) == len(problems):
                recent = epoch_losses[-10:]
                avg_loss = sum(recent) / max(len(recent), 1)
                avg_pr = sum(epoch_pass_rates[-10:]) / max(len(epoch_pass_rates[-10:]), 1)
                recent_tests_passed = sum(epoch_tests_passed[-10:])
                recent_tests_total = sum(epoch_tests_total[-10:])
                avg_test_pr = (
                    recent_tests_passed / recent_tests_total
                    if recent_tests_total else 0.0
                )
                log.info(
                    "Epoch %d  %d/%d  loss=%.4f  completion_mean_pass_rate=%.3f  "
                    "test_pass_rate=%.3f (%d/%d tests)  "
                    "full_passed=%d/%d completions  valid_distil=%d/%d",
                    epoch + 1, prob_idx + 1, len(problems),
                    avg_loss, avg_pr,
                    avg_test_pr, recent_tests_passed, recent_tests_total,
                    n_passed, args.num_generations,
                    n_valid, args.num_generations,
                )
                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": total_loss,
                        "train/full_passed_completions": n_passed,
                        "train/completion_mean_pass_rate": mean_pass_rate,
                        "train/tests_passed": tests_passed,
                        "train/tests_total": tests_total,
                        "train/test_pass_rate": test_pass_rate,
                        "train/n_valid_distil": n_valid,
                        "train/lr": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                    })

            # Periodic checkpoint
            if global_step % args.save_steps == 0:
                ckpt_path = out_dir / f"checkpoint-{global_step}"
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                log.info("Checkpoint saved → %s", ckpt_path)

        log.info(
            "Epoch %d complete | avg_loss=%.4f | completion_mean_pass_rate=%.3f | "
            "test_pass_rate=%d/%d=%.3f",
            epoch + 1,
            sum(epoch_losses) / max(len(epoch_losses), 1),
            sum(epoch_pass_rates) / max(len(epoch_pass_rates), 1),
            sum(epoch_tests_passed),
            sum(epoch_tests_total),
            (sum(epoch_tests_passed) / sum(epoch_tests_total))
            if sum(epoch_tests_total) else 0.0,
        )

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info("Training complete. Final checkpoint saved → %s", out_dir)

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
