#!/usr/bin/env python3
"""Generate paper figures from training logs → figures/."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install matplotlib"); exit(1)

METHODS = ["sft", "sdft", "sdpo", "omni", "grpo"]
COLORS = {"sft": "#888", "sdft": "#4a90d9", "sdpo": "#e67e22",
           "omni": "#27ae60", "grpo": "#8e44ad"}


def _load_log(mode):
    p = Path(f"logs/train_{mode}.jsonl")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _save(fig, name):
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ figures/{name}")


def _plot_curves(key, ylabel, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    for m in METHODS:
        entries = _load_log(m)
        vals = [(e["epoch"]+1, e[key]) for e in entries if key in e]
        if not vals: continue
        any_data = True
        ax.plot(*zip(*vals), "o-", label=m.upper(), color=COLORS.get(m))
    if any_data:
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=0.3)
        _save(fig, fname)
    else:
        plt.close(fig)


def main():
    _plot_curves("loss", "Loss", "training_curves.png")
    _plot_curves("eval_pass@1", "Pass@1", "pass_at_1_curves.png")

    # ablation heatmap
    abl = Path("logs/ablation_results.csv")
    if abl.exists():
        rows = list(csv.DictReader(abl.open()))
        alphas = sorted(set(float(r["alpha"]) for r in rows))
        betas = sorted(set(float(r["beta"]) for r in rows))
        grid = np.zeros((len(alphas), len(betas)))
        for r in rows:
            grid[alphas.index(float(r["alpha"])), betas.index(float(r["beta"]))] = float(r.get("pass@1", 0))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid, cmap="YlGn", origin="lower", aspect="auto")
        ax.set_xticks(range(len(betas)), [str(b) for b in betas])
        ax.set_yticks(range(len(alphas)), [str(a) for a in alphas])
        ax.set_xlabel("β (SDPO)"); ax.set_ylabel("α (SDFT)")
        fig.colorbar(im, label="Pass@1")
        for i in range(len(alphas)):
            for j in range(len(betas)):
                ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=9)
        _save(fig, "ablation_heatmap.png")

    print("Done.")


if __name__ == "__main__":
    main()
