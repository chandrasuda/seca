#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Run all 5 methods sequentially, then evaluate each checkpoint.
# Usage:  bash scripts/sweep_all.sh [CONFIG] [EXTRA_ARGS...]
# Example: bash scripts/sweep_all.sh configs/default.yaml --model Qwen/Qwen3-1.7B
# ══════════════════════════════════════════════════════════════
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
shift || true
EXTRA_ARGS=("$@")
MODES="sft sdft sdpo omni grpo"

echo "═══ Omni-Teacher: Full Method Sweep (vLLM + HF) ═══"
echo "Config: $CONFIG"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo ""

for MODE in $MODES; do
    echo "────────────────────────────────────"
    echo "  Training: $MODE"
    echo "────────────────────────────────────"
    python scripts/run.py --mode "$MODE" --config "$CONFIG" "${EXTRA_ARGS[@]}"
    echo ""
done

echo ""
echo "═══ Evaluating all methods ═══"
echo ""

for MODE in $MODES; do
    echo "────────────────────────────────────"
    echo "  Evaluating: $MODE"
    echo "────────────────────────────────────"

    CKPT=$(ls -t checkpoints/${MODE}_epoch*.pt 2>/dev/null | head -1 || true)
    if [ -n "$CKPT" ]; then
        python scripts/eval.py --config "$CONFIG" --checkpoint "$CKPT" "${EXTRA_ARGS[@]}"
    else
        echo "  (no checkpoint found, evaluating base model)"
        python scripts/eval.py --config "$CONFIG" "${EXTRA_ARGS[@]}"
    fi
    echo ""
done

echo "═══ Done. Results in logs/ ═══"
