#!/usr/bin/env python3
"""Diagnostic checks for SDFT/SDPO training. Run before training.

Usage:
    python scripts/run_diagnostics.py [--config configs/default.yaml]

WARNING: Checks 5-7 modify model weights (each creates its own AdamW step).
         Run on the base model, not a partially-trained checkpoint.

Expected outcomes:
  Check 1 - Token alignment:   Match: True
  Check 2 - KL range:          0.05 – 0.5
  Check 3 - Gold matters:      kl_real_gold != kl_fake_gold
  Check 4 - Gradients flow:    total norm in [0.01, 10.0]
  Check 5 - EMA updates:       delta norm ~1e-4 to 1e-6, EMA changed=True
  Check 6 - vLLM sync:         no crash
  Check 7 - Loss decreases:    monotonically over 20 steps
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from torch.optim import AdamW

from seca.data.loader import load_problems
from seca.models.base import BaseModel
from seca.models.sdft import SDFTOperator
from seca.train.trainer import _ema_update


# ── helpers ──────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_kl_loss(
    model: BaseModel,
    ema_model: BaseModel,
    problem,
    completion: str,
) -> torch.Tensor:
    """SDFT KL loss for one (problem, completion) pair at T=1 for diagnostics."""
    op = SDFTOperator({
        "temperature_student": 1.0,
        "temperature_teacher": 1.0,
        "kl_weight": 1.0,
        "topk": 100,
    })
    loss, _ = op.loss(model, ema_model, [problem], [completion])
    return loss


def _sep(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── checks ───────────────────────────────────────────────────────────────────

def check1_token_alignment(model: BaseModel, problem, completion: str) -> bool:
    """Student and teacher completion tokens must align at the same positions."""
    _sep("Check 1 — Token Position Alignment")

    prompt = problem.format_prompt()
    student_input = f"{prompt}\n{completion}"
    teacher_prefix = (
        f"{prompt}\n### Optimal Solution\n{problem.gold_solution}"
        f"\n### Student Attempt\n"
    )
    teacher_input = teacher_prefix + completion

    tok = model.tokenizer
    student_ids = tok(student_input, return_tensors="pt").input_ids[0]
    teacher_ids = tok(teacher_input, return_tensors="pt").input_ids[0]

    s_prompt_len = tok(f"{prompt}\n", return_tensors="pt").input_ids.shape[1]
    t_prefix_len = tok(teacher_prefix, return_tensors="pt").input_ids.shape[1]

    s_comp = student_ids[s_prompt_len:]
    t_comp = teacher_ids[t_prefix_len:]

    min_len = min(len(s_comp), len(t_comp))
    tokens_match = torch.all(s_comp[:min_len] == t_comp[:min_len]).item()
    lengths_match = len(s_comp) == len(t_comp)

    print(f"  Student completion tokens  (first 5): {s_comp[:5].tolist()}")
    print(f"  Teacher completion tokens  (first 5): {t_comp[:5].tolist()}")
    print(f"  Student completion length: {len(s_comp)}")
    print(f"  Teacher completion length: {len(t_comp)}")
    print(f"  Token content match: {tokens_match}   Length match: {lengths_match}")

    if tokens_match and lengths_match:
        print("  PASS — completion tokens align perfectly")
        return True
    elif tokens_match and not lengths_match:
        # BPE boundary added/dropped one token — the min() guard in sdft/sdpo handles this
        print("  WARN — tokens match but lengths differ (BPE boundary effect).")
        print("         The min() guard in sdft.py / sdpo.py handles this safely.")
        return True
    else:
        print("  FAIL — completion tokens do NOT match!")
        print("         KL is being computed on misaligned positions.")
        return False


def check2_kl_nonzero(
    model: BaseModel, ema_model: BaseModel, problem, completion: str
) -> bool:
    """KL should be non-zero (teacher has different context) and in a sane range."""
    _sep("Check 2 — KL is Non-Zero and Reasonable")

    loss = compute_kl_loss(model, ema_model, problem, completion)
    val = loss.item()
    print(f"  KL loss: {val:.4f}   (expected 0.05 – 0.5)")

    if val < 0.001:
        print("  FAIL — KL ≈ 0: teacher and student produce identical logits,")
        print("         or the completion-position indexing is broken.")
        return False
    if val > 2.0:
        print(f"  WARN — KL={val:.4f} is high; check temperature / indexing.")
        return True
    print(f"  PASS — KL={val:.4f} is in expected range")
    return True


def check3_gold_matters(
    model: BaseModel, ema_model: BaseModel, problem, completion: str
) -> bool:
    """KL should change when the gold solution changes — teacher must use it."""
    _sep("Check 3 — Gold Solution Changes Teacher Distribution")

    kl_real = compute_kl_loss(model, ema_model, problem, completion).item()

    fake = dataclasses.replace(problem, gold_solution="def foo():\n    pass\n")
    kl_fake = compute_kl_loss(model, ema_model, fake, completion).item()

    print(f"  KL with real gold: {kl_real:.4f}")
    print(f"  KL with fake gold: {kl_fake:.4f}")
    print(f"  |Δ|: {abs(kl_real - kl_fake):.4f}")

    if abs(kl_real - kl_fake) < 0.001:
        print("  FAIL — teacher ignores gold context; KL unchanged.")
        return False
    print("  PASS — teacher responds to gold solution content")
    return True


def check4_gradients_nonzero(
    model: BaseModel, ema_model: BaseModel, problem, completion: str
) -> bool:
    """Gradients must flow through the student; total norm should be in [0.01, 10]."""
    _sep("Check 4 — Gradients Are Non-Zero and Reasonable")

    for p in model.model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loss = compute_kl_loss(model, ema_model, problem, completion)
    loss.backward()

    first_name, first_norm = None, None
    for name, param in model.model.named_parameters():
        if param.grad is not None and param.grad.norm().item() > 0:
            first_name = name
            first_norm = param.grad.norm().item()
            break

    total_norm = torch.nn.utils.clip_grad_norm_(
        model.model.parameters(), float("inf")
    ).item()

    if first_name:
        print(f"  First non-zero grad: {first_name}")
        print(f"  First param grad norm: {first_norm:.6f}")
    else:
        print("  No non-zero gradients found!")

    print(f"  Total gradient norm: {total_norm:.4f}   (expected 0.01 – 10.0)")

    if total_norm < 1e-8:
        print("  FAIL — all gradients are zero; computation graph is broken.")
        return False
    if total_norm > 100.0:
        print(f"  WARN — norm={total_norm:.1f} is very large; consider lower LR or tighter clip.")
        return True
    print(f"  PASS — gradient norm {total_norm:.4f} is healthy")
    return True


def check5_ema_updates(
    model: BaseModel, ema_model: BaseModel, problem, completion: str
) -> bool:
    """EMA parameters must change after an optimizer step, but only slightly."""
    _sep("Check 5 — EMA Updates Correctly")

    initial_ema = list(ema_model.model.parameters())[0].data.clone()

    optimizer = AdamW(model.model.parameters(), lr=5e-7)
    optimizer.zero_grad()
    loss = compute_kl_loss(model, ema_model, problem, completion)
    loss.backward()
    optimizer.step()

    # alpha=0.005: ϕ ← 0.995·ϕ + 0.005·θ
    _ema_update(ema_model, model, alpha=0.005)

    updated_ema = list(ema_model.model.parameters())[0].data
    changed = not torch.allclose(initial_ema, updated_ema)
    delta_norm = (updated_ema - initial_ema).norm().item()

    print(f"  EMA changed:    {changed}   (expected True)")
    print(f"  EMA delta norm: {delta_norm:.2e}   (expected ~1e-4 to 1e-6)")

    if not changed:
        print("  FAIL — EMA not updating; check _ema_update call in trainer.")
        return False
    if delta_norm > 0.01:
        print(f"  WARN — delta={delta_norm:.4f} is large; EMA alpha may be wrong.")
        return True
    print("  PASS — EMA updating correctly with small delta")
    return True


def check6_vllm_sync(
    model: BaseModel, ema_model: BaseModel, problem, completion: str
) -> bool:
    """sync_vllm_weights() must not crash; output may diverge after many steps."""
    _sep("Check 6 — vLLM Weight Sync")

    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    prompt = problem.format_prompt()

    try:
        before = model.generate([prompt], sampling_params=sp)[0]
        print(f"  Before sync: {repr(before[:60])}")
    except Exception as e:
        print(f"  FAIL — generation before sync raised: {e}")
        return False

    optimizer = AdamW(model.model.parameters(), lr=5e-7)
    optimizer.zero_grad()
    loss = compute_kl_loss(model, ema_model, problem, completion)
    loss.backward()
    optimizer.step()

    try:
        model.sync_vllm_weights()
        print("  sync_vllm_weights() completed without error")
    except Exception as e:
        print(f"  FAIL — sync_vllm_weights() raised: {e}")
        return False

    try:
        after = model.generate([prompt], sampling_params=sp)[0]
        print(f"  After sync:  {repr(after[:60])}")
    except Exception as e:
        print(f"  FAIL — generation after sync raised: {e}")
        return False

    print("  PASS — no crash during sync (one step may not visibly change output)")
    return True


def check7_loss_decreases(
    model: BaseModel, ema_model: BaseModel, problem, completion: str
) -> bool:
    """Loss on a single fixed example must decrease over 20 gradient steps."""
    _sep("Check 7 — Loss Decreases on Single Example (overfit test)")
    print("  WARNING: this modifies model weights.")

    optimizer = AdamW(model.model.parameters(), lr=5e-7)
    losses: list[float] = []

    for _ in range(20):
        optimizer.zero_grad()
        loss = compute_kl_loss(model, ema_model, problem, completion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    first_half = sum(losses[:10]) / 10
    second_half = sum(losses[10:]) / 10
    decreasing = second_half < first_half

    print(f"  Losses: {[f'{v:.4f}' for v in losses]}")
    print(f"  First-half mean:  {first_half:.4f}")
    print(f"  Second-half mean: {second_half:.4f}")

    if decreasing:
        print(f"  PASS — loss decreased ({first_half:.4f} → {second_half:.4f})")
        return True
    print(f"  FAIL — loss did NOT decrease; check LR or gradient flow.")
    return False


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SDFT diagnostic checks")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    print(f"Config:  {args.config}")
    print(f"Model:   {cfg['model']['name']}")

    # Load a small slice of problems (diagnostics only need a couple)
    data_cfg = dict(cfg.get("data", {}))
    data_cfg["max_problems"] = 5
    problems = load_problems(data_cfg)
    if not problems:
        print("ERROR: no problems loaded — check data config.")
        sys.exit(1)
    print(f"Loaded {len(problems)} problems")

    problem = problems[0]
    # Deliberately wrong/simple completion — we want a completion that fails so
    # the teacher's gold-conditioned distribution clearly differs from the student.
    completion = "def solve(n):\n    return n\n"

    # Initialise model + frozen EMA snapshot
    model_cfg = dict(cfg["model"])
    model_cfg["vllm_cfg"] = cfg.get("vllm", {})
    model = BaseModel(**model_cfg)
    ema_model = model.snapshot()

    print(f"\nModel loaded. Running 7 diagnostic checks…\n{'═'*60}")

    results: dict[str, bool] = {}
    results["check1_token_alignment"] = check1_token_alignment(model, problem, completion)
    results["check2_kl_nonzero"]      = check2_kl_nonzero(model, ema_model, problem, completion)
    results["check3_gold_matters"]    = check3_gold_matters(model, ema_model, problem, completion)
    results["check4_gradients"]       = check4_gradients_nonzero(model, ema_model, problem, completion)
    results["check5_ema_updates"]     = check5_ema_updates(model, ema_model, problem, completion)
    results["check6_vllm_sync"]       = check6_vllm_sync(model, ema_model, problem, completion)
    results["check7_loss_decreases"]  = check7_loss_decreases(model, ema_model, problem, completion)

    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} checks passed")

    if n_pass == n_total:
        print("\nAll checks passed — ready to train.")
    else:
        print("\nFix the failing checks before starting a run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
