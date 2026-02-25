# Omni-Teacher

**CS224N Custom Project** — Execution-grounded self-distillation for code generation.

SDFT (Shenfeld 2026) teaches from expert demos. SDPO teaches from execution feedback. **Omni-Teacher combines both** — the teacher sees prompt + gold solution + student attempt + execution feedback simultaneously.

```
L(θ) = α · KL_sdft + β · KL_sdpo + γ · NLL_gold
```

## Quick Start

```bash
pip install -e .
python scripts/run.py --mode omni          # train Omni-Teacher
python scripts/run.py --mode sft           # or any baseline
python scripts/eval.py --checkpoint checkpoints/omni_epoch2.pt
bash scripts/sweep_all.sh                  # all 5 methods + eval
```

## Structure

```
seca/
├── data/       problem.py  loader.py  apps.py  livecodebench.py  kernelbench.py
├── models/     base.py  sdft.py  sdpo.py  omni.py
├── sandbox/    executor.py
├── train/      trainer.py
└── eval/       metrics.py
scripts/        run.py  eval.py  sweep_all.sh  sweep_ablation.py  inspect_data.py  plot_results.py
```

## Datasets

| Dataset | HF ID | Why |
|---|---|---|
| **APPS** | `codeparrot/apps` | 10k problems, 3 tiers, test cases + gold solutions |
| **LiveCodeBench** | `livecodebench/code_generation_lite` | SDPO paper's benchmark |
| **KernelBench** | `ScalingIntelligence/KernelBench` | CUDA kernels, rich compiler feedback |

## Methods

| Mode | Teacher sees |
|---|---|
| `sft` | — (NLL on gold only) |
| `sdft` | prompt + gold + attempt |
| `sdpo` | prompt + attempt + execution feedback |
| **`omni`** | **all four** (our contribution) |
| `grpo` | — (reward = test pass rate) |
