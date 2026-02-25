# NEXT STEPS — Omni-Teacher (CS224N Project)

Everything below is a command you can copy-paste. No code to write — just run.

---

## Step 0: Install

```bash
cd /Users/csuda/Downloads/seca
pip install -e .
```

---

## Step 1: Validate Your Data (run BEFORE any training)

```bash
# Check APPS dataset — how many problems have gold + tests?
python scripts/inspect_data.py --config configs/default.yaml

# Verify gold solutions actually pass their own tests (catches bad data)
python scripts/inspect_data.py --check-gold --n 30

# See what execution feedback looks like (this is what the SDPO teacher sees)
python scripts/inspect_data.py --show-feedback

# Also validate LiveCodeBench
# (edit configs/default.yaml → data.dataset: "livecodebench" first, then:)
python scripts/inspect_data.py --check-gold --show-feedback
```

**What to look for:**
- If >10% of gold solutions fail their tests → filter those problems or switch datasets
- If the feedback summary is too terse (just "FAIL") → open `seca/sandbox/executor.py` and increase the stderr truncation limit

---

## Step 2: Smoke Test (5 problems, 1 epoch)

Edit `configs/default.yaml` temporarily:
```yaml
data:
  max_problems: 5
training:
  num_epochs: 1
```

Then:
```bash
python scripts/run.py --mode sft
python scripts/run.py --mode omni
```

Verify: no OOM, no crashes, checkpoints appear in `checkpoints/`, logs in `logs/`.

---

## Step 3: Full Experiment Sweep (all 5 methods)

Restore `configs/default.yaml` to full size:
```yaml
data:
  max_problems: 100
training:
  num_epochs: 3
```

Then one command runs everything:
```bash
bash scripts/sweep_all.sh
```

This trains SFT → SDFT → SDPO → Omni → GRPO sequentially, then evaluates each checkpoint. Results go to `logs/`.

Or run individual methods:
```bash
python scripts/run.py --mode sft
python scripts/run.py --mode sdft
python scripts/run.py --mode sdpo
python scripts/run.py --mode omni
python scripts/run.py --mode grpo
```

---

## Step 4: Ablation Sweep (α/β/γ grid)

```bash
python scripts/sweep_ablation.py --config configs/default.yaml
```

This runs a 3×3×3 grid over (α, β, γ) and writes `logs/ablation_results.csv`.

---

## Step 5: Generate Paper Figures

```bash
python scripts/plot_results.py
```

Produces in `figures/`:
- `training_curves.png` — loss per epoch for each method
- `pass_at_1_curves.png` — Pass@1 per epoch for each method
- `eval_bar.png` — final Pass@k bar chart
- `ablation_heatmap.png` — α vs β grid with Pass@1

---

## Step 6: Final Evaluation (full Pass@k)

```bash
# evaluate the omni checkpoint
python scripts/eval.py --checkpoint checkpoints/omni_epoch2.pt

# evaluate any other method
python scripts/eval.py --checkpoint checkpoints/sdft_epoch2.pt
python scripts/eval.py --checkpoint checkpoints/grpo_epoch2.pt
```

---

## Why These Datasets Are New Here

**APPS** — has both gold solutions (→ SDFT) and test cases (→ SDPO execution feedback). Never used for on-policy self-distillation conditioned on execution traces.

**LiveCodeBench** — the SDPO paper's benchmark. They used execution feedback only. We add gold-solution conditioning — first combined-signal evaluation here.

**KernelBench** — rich compiler errors + speedup metrics as feedback. Never used for self-distillation of any kind.

**The novelty:** no prior work conditions a single teacher on both (1) expert gold solution and (2) execution traces. SDFT uses only (1). SDPO uses only (2). We fuse both.
