#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Run all 5 methods sequentially, then evaluate each checkpoint.
# Usage:  bash scripts/sweep_all.sh [CONFIG]
# ══════════════════════════════════════════════════════════════
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
MODES="sft sdft sdpo omni grpo"

echo "═══ Omni-Teacher: Full Method Sweep ═══"
echo "Config: $CONFIG"
echo ""

for MODE in $MODES; do
    echo "────────────────────────────────────"
    echo "  Training: $MODE"
    echo "────────────────────────────────────"
    python scripts/run.py --mode "$MODE" --config "$CONFIG"
    echo ""
done

echo ""
echo "═══ Evaluating all methods ═══"
echo ""

for MODE in $MODES; do
    echo "────────────────────────────────────"
    echo "  Evaluating: $MODE"
    echo "────────────────────────────────────"

    # find the latest checkpoint for this mode
    CKPT=$(ls -t checkpoints/${MODE}_epoch*.pt 2>/dev/null | head -1 || true)
    if [ -n "$CKPT" ]; then
        python scripts/eval.py --config "$CONFIG" --checkpoint "$CKPT"
    else
        echo "  (no checkpoint found, evaluating base model)"
        python scripts/eval.py --config "$CONFIG"
    fi
    echo ""
done

echo "═══ Done. Results in logs/ ═══"
