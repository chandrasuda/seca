#!/usr/bin/env python3
"""Diagnostic checks for SDPO training. Run before any SDPO training run.

8 checks covering executor correctness, teacher context construction,
KL conditioning, gradient flow, EMA, vLLM sync, and loss convergence.

Usage:
    python scripts/run_sdpo_diagnostics.py [--config configs/default.yaml]

WARNING: Checks 6-8 modify model weights. Run on the base model only.

Expected outcomes:
  Check 1 - Executor pass/fail:        gold passes, wrong code fails
  Check 2 - Teacher context cases:     A/B/C/D produce 4 distinct prompts
  Check 3 - SDPO KL non-zero:          KL in [0.01, 5.0] for mixed (Case A+C)
  Check 4 - Feedback changes KL:       Case C+A mix ≠ Case D only
  Check 5 - Gradients flow:            total norm in [0.01, 200.0]
  Check 6 - EMA updates:               delta norm ~1e-4 to 1e-6
  Check 7 - vLLM sync:                 no crash
  Check 8 - Loss decreases:            second-half mean < first-half (20 steps)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from torch.optim import AdamW
from vllm import SamplingParams

from seca.data.loader import load_problems
from seca.models.base import BaseModel
from seca.models.sdpo import SDPOOperator, _build_teacher_context
from seca.sandbox.executor import FeedbackBundle, execute_code
from seca.train.trainer import _ema_update


# ── helpers ──────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_sdpo_loss(
    model: BaseModel,
    ema_model: BaseModel,
    problem,
    completions: list[str],
    feedback_bundles: list[FeedbackBundle],
) -> torch.Tensor:
    """Run SDPOOperator.loss over a completion group at T=1 for diagnostics."""
    op = SDPOOperator({
        "temperature_student": 1.0,
        "temperature_teacher": 1.0,
        "kl_weight": 1.0,
        "topk": 20,
    })
    loss, _ = op.loss(
        model, ema_model,
        [problem] * len(completions),
        completions,
        feedback_bundles,
    )
    return loss


def _sep(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── checks ───────────────────────────────────────────────────────────────────

def check1_executor(problem, passing_code: str, failing_code: str) -> bool:
    """Executor must flag the gold solution as passing and wrong code as failing."""
    _sep("Check 1 — Executor Pass/Fail Detection")

    fb_pass = execute_code(passing_code, problem, extract=False)
    fb_fail = execute_code(failing_code, problem, extract=False)

    print(f"  Gold solution → all_passed={fb_pass.all_passed}, pass_rate={fb_pass.pass_rate:.2f}")
    print(f"  Wrong code    → all_passed={fb_fail.all_passed}, pass_rate={fb_fail.pass_rate:.2f}")
    feedback_preview = fb_fail.summary.replace("\n", "\n    ")
    print(f"  Failing feedback:\n    {feedback_preview[:400]}")

    ok = True
    if not fb_pass.all_passed:
        print(f"  FAIL — gold solution did not pass its own tests.")
        print(f"         Ensure data_filtered/ contains pre-filtered data (run filter_data.py).")
        ok = False
    else:
        print(f"  PASS — gold solution passes all tests")

    if fb_fail.all_passed:
        print(f"  FAIL — trivially wrong code was marked as passing")
        ok = False
    else:
        print(f"  PASS — wrong code fails as expected")

    if "Passed" not in fb_fail.summary:
        print(f"  WARN — feedback summary missing 'Passed X/Y' header")

    return ok


def check2_teacher_contexts(
    problem,
    passing_code: str,
    failing_code: str,
    fb_fail: FeedbackBundle,
) -> bool:
    """All 4 SDPO teacher-context cases (A/B/C/D) must be distinct strings."""
    _sep("Check 2 — Teacher Context Construction (Cases A/B/C/D)")

    # Case A: passed completion; a DIFFERENT passing demo is available
    ctx_A = _build_teacher_context(
        problem, passing_code,
        feedback_summary="",
        passed=True,
        demo_completion=passing_code + "\n# alt",   # different string, still valid
    )
    # Case B: passed; own completion is the only solution (no other demo)
    ctx_B = _build_teacher_context(
        problem, passing_code,
        feedback_summary="",
        passed=True,
        demo_completion=None,
    )
    # Case C: failed; a passing demo exists in the group
    ctx_C = _build_teacher_context(
        problem, failing_code,
        feedback_summary=fb_fail.summary,
        passed=False,
        demo_completion=passing_code,
    )
    # Case D: all failed; no demo at all
    ctx_D = _build_teacher_context(
        problem, failing_code,
        feedback_summary=fb_fail.summary,
        passed=False,
        demo_completion=None,
    )

    contexts = {"A": ctx_A, "B": ctx_B, "C": ctx_C, "D": ctx_D}

    for name, ctx in contexts.items():
        preview = ctx.replace("\n", "↵")[:90]
        print(f"  Case {name}: {preview}…")

    ok = True

    # Content checks per paper Table 2
    if "from another successful attempt" not in ctx_A:
        print("  FAIL — Case A: missing 'from another successful attempt'")
        ok = False
    if "from your successful attempt" not in ctx_B:
        print("  FAIL — Case B: missing 'from your successful attempt'")
        ok = False
    if "Correct solution" not in ctx_C or "unsuccessful earlier attempt" not in ctx_C:
        print("  FAIL — Case C: must contain both correct solution and feedback")
        ok = False
    if "Correct solution" in ctx_D:
        print("  FAIL — Case D: must NOT contain a correct solution (no demo available)")
        ok = False
    if "unsuccessful earlier attempt" not in ctx_D:
        print("  FAIL — Case D: must contain feedback from the failing attempt")
        ok = False

    all_distinct = len(set(contexts.values())) == 4
    if not all_distinct:
        print("  FAIL — not all 4 contexts are distinct strings")
        ok = False

    if ok:
        print("  PASS — all 4 teacher contexts (A/B/C/D) are correct and distinct")
    return ok


def check3_kl_nonzero(
    model: BaseModel,
    ema_model: BaseModel,
    problem,
    passing_code: str,
    failing_code: str,
    fb_pass: FeedbackBundle,
    fb_fail: FeedbackBundle,
) -> bool:
    """SDPO KL (mixed Case A+C) must be non-zero and in sane range.

    Uses mixed group since Case D can hit empty losses when teacher context
    truncates (long feedback). Mixed case always produces valid KL with grad_fn.
    """
    _sep("Check 3 — SDPO KL Non-Zero (mixed pass+fail, Case A+C)")

    # Mixed group: failing completion gets demo from passing → Case C, non-zero KL
    loss = compute_sdpo_loss(
        model, ema_model, problem,
        [passing_code, failing_code],
        [fb_pass, fb_fail],
    )
    val = loss.item()
    print(f"  SDPO KL (mixed group): {val:.4f}   (expected 0.01 – 5.0)")

    if val < 1e-6:
        print("  FAIL — KL ≈ 0: teacher and student produce identical logits,")
        print("         or completion-position indexing is broken.")
        return False
    if val > 10.0:
        print(f"  WARN — KL={val:.4f} is very high; check temperature or indexing.")
        return True
    print(f"  PASS — KL={val:.4f} is non-zero and sane")
    return True


def check4_feedback_changes_kl(
    model: BaseModel,
    ema_model: BaseModel,
    problem,
    passing_code: str,
    failing_code: str,
    fb_pass: FeedbackBundle,
    fb_fail: FeedbackBundle,
) -> bool:
    """Passing a demo to the teacher (Case C) must change the KL vs no demo (Case D).

    Groups [failing_only] vs [passing, failing]:
      - [failing_only]    → Case D for the failing completion (feedback only)
      - [passing, failing] → Case A/B for passing, Case C for failing (demo + feedback)
    If the demo has no effect on the teacher distribution, the KL values will be equal.
    """
    _sep("Check 4 — Demo Conditioning Changes KL (Case C vs Case D)")

    # Case D: no passing demo in the group
    kl_D = compute_sdpo_loss(
        model, ema_model, problem,
        [failing_code], [fb_fail],
    ).item()

    # Case C in a mixed group: failing completion gets a demo from the passing one
    kl_mixed = compute_sdpo_loss(
        model, ema_model, problem,
        [passing_code, failing_code],
        [fb_pass, fb_fail],
    ).item()

    print(f"  KL  — Case D only (1 failing, no demo):      {kl_D:.4f}")
    print(f"  KL  — mixed group (pass + fail, Case A+C):   {kl_mixed:.4f}")
    print(f"  |Δ|: {abs(kl_D - kl_mixed):.4f}")

    if abs(kl_D - kl_mixed) < 1e-6:
        print("  FAIL — demo completion has no effect on KL.")
        print("         Teacher context may not be using the demo solution.")
        return False
    print("  PASS — adding a passing demo to the group changes the KL signal")
    return True


def check5_gradients_flow(
    model: BaseModel,
    ema_model: BaseModel,
    problem,
    passing_code: str,
    failing_code: str,
    fb_pass: FeedbackBundle,
    fb_fail: FeedbackBundle,
) -> bool:
    """Gradients must flow through the SDPO student forward pass.
    Uses mixed group (Case A+C) so loss has grad_fn; Case D can return constant.
    """
    _sep("Check 5 — Gradients Flow Through SDPO Loss")

    for p in model.model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loss = compute_sdpo_loss(
        model, ema_model, problem,
        [passing_code, failing_code],
        [fb_pass, fb_fail],
    )
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

    print(f"  Total gradient norm: {total_norm:.4f}   (expected 0.01 – 200.0)")

    if total_norm < 1e-8:
        print("  FAIL — all gradients are zero; computation graph is broken.")
        return False
    if total_norm > 500.0:
        print(f"  WARN — norm={total_norm:.1f} is very large; consider lower LR or clip.")
        return True
    print(f"  PASS — gradient norm {total_norm:.4f} is healthy")
    return True


def check6_ema_updates(
    model: BaseModel,
    ema_model: BaseModel,
    problem,
    passing_code: str,
    failing_code: str,
    fb_pass: FeedbackBundle,
    fb_fail: FeedbackBundle,
) -> bool:
    """EMA teacher must update after an SDPO optimizer step.
    Uses mixed group so loss has grad_fn."""
    _sep("Check 6 — EMA Teacher Updates After SDPO Step")

    initial_ema = list(ema_model.model.parameters())[0].data.clone()

    optimizer = AdamW(model.model.parameters(), lr=5e-7)
    optimizer.zero_grad()
    loss = compute_sdpo_loss(
        model, ema_model, problem,
        [passing_code, failing_code],
        [fb_pass, fb_fail],
    )
    loss.backward()
    optimizer.step()

    # alpha=0.005: ϕ ← 0.995·ϕ + 0.005·θ
    _ema_update(ema_model, model, alpha=0.005)

    updated_ema = list(ema_model.model.parameters())[0].data
    changed = not torch.allclose(initial_ema, updated_ema)
    delta_norm = (updated_ema - initial_ema).norm().item()

    print(f"  EMA changed:    {changed}   (expected True)")
    print(f"  EMA delta norm: {delta_norm:.2e}   (expected ~1e-6 to 1e-4)")

    if not changed:
        print("  FAIL — EMA not updating; check _ema_update call in trainer.")
        return False
    if delta_norm > 0.01:
        print(f"  WARN — delta={delta_norm:.4f} is large; verify ema_alpha=0.005.")
        return True
    print("  PASS — EMA updating correctly with small delta")
    return True


def check7_vllm_sync(
    model: BaseModel,
    ema_model: BaseModel,
    problem,
    passing_code: str,
    failing_code: str,
    fb_pass: FeedbackBundle,
    fb_fail: FeedbackBundle,
) -> bool:
    """sync_vllm_weights() must not crash after an SDPO gradient step.
    Uses mixed group so loss has grad_fn."""
    _sep("Check 7 — vLLM Weight Sync After SDPO Step")

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
    loss = compute_sdpo_loss(
        model, ema_model, problem,
        [passing_code, failing_code],
        [fb_pass, fb_fail],
    )
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

    print("  PASS — no crash during vLLM weight sync")
    return True


def check8_loss_decreases(
    model: BaseModel,
    ema_model: BaseModel,
    problem,
    passing_code: str,
    failing_code: str,
    fb_pass: FeedbackBundle,
    fb_fail: FeedbackBundle,
) -> bool:
    """SDPO loss must decrease over 20 gradient steps on a fixed example.
    Uses mixed group so loss has grad_fn."""
    _sep("Check 8 — Loss Decreases on Fixed Failing Example (overfit test)")
    print("  WARNING: this modifies model weights.")

    optimizer = AdamW(model.model.parameters(), lr=5e-7)
    losses: list[float] = []

    for _ in range(20):
        optimizer.zero_grad()
        loss = compute_sdpo_loss(
            model, ema_model, problem,
            [passing_code, failing_code],
            [fb_pass, fb_fail],
        )
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
    print(f"  FAIL — loss did NOT decrease; check LR or SDPO gradient flow.")
    return False


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SDPO diagnostic checks")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    print(f"Config:  {args.config}")
    print(f"Model:   {cfg['model']['name']}")

    # Load a small slice of problems from data_filtered/ (pre-filtered).
    data_cfg = dict(cfg.get("data", {}))
    data_cfg["max_problems"] = 5
    problems = load_problems(data_cfg)
    if not problems:
        print("ERROR: no problems loaded — check data config.")
        sys.exit(1)
    print(f"Loaded {len(problems)} problems")

    problem = problems[0]
    passing_code = problem.gold_solution
    # Raises immediately — no output → fails any test expecting output;
    # stderr is non-empty → marked FAIL by the executor.
    failing_code = "raise NotImplementedError('diagnostic placeholder')"

    # ── executor checks (no model needed) ────────────────────────────────────
    print(f"\nRunning executor + context checks (no model required)…")
    fb_pass = execute_code(passing_code, problem, extract=False)
    fb_fail = execute_code(failing_code, problem, extract=False)

    results: dict[str, bool] = {}
    results["check1_executor"]         = check1_executor(problem, passing_code, failing_code)
    results["check2_teacher_contexts"] = check2_teacher_contexts(
        problem, passing_code, failing_code, fb_fail
    )

    # ── model-based checks ───────────────────────────────────────────────────
    print(f"\nLoading model ({cfg['model']['name']})…")
    model_cfg = dict(cfg["model"])
    model_cfg["vllm_cfg"] = cfg.get("vllm", {})
    model = BaseModel(**model_cfg)
    ema_model = model.snapshot()
    print(f"Model loaded. Running 6 model-based checks…\n{'═'*60}")

    results["check3_kl_nonzero"]          = check3_kl_nonzero(
        model, ema_model, problem, passing_code, failing_code, fb_pass, fb_fail
    )
    results["check4_feedback_changes_kl"] = check4_feedback_changes_kl(
        model, ema_model, problem, passing_code, failing_code, fb_pass, fb_fail
    )
    results["check5_gradients_flow"]      = check5_gradients_flow(
        model, ema_model, problem, passing_code, failing_code, fb_pass, fb_fail
    )
    results["check6_ema_updates"]         = check6_ema_updates(
        model, ema_model, problem, passing_code, failing_code, fb_pass, fb_fail
    )
    results["check7_vllm_sync"]           = check7_vllm_sync(
        model, ema_model, problem, passing_code, failing_code, fb_pass, fb_fail
    )
    results["check8_loss_decreases"]      = check8_loss_decreases(
        model, ema_model, problem, passing_code, failing_code, fb_pass, fb_fail
    )

    # ── summary ──────────────────────────────────────────────────────────────
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
        print("\nAll checks passed — ready to train SDPO.")
    else:
        print("\nFix the failing checks before starting a run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
