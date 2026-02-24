# SECA — Selective Externalization for Continual Adaptation

> Can a learned routing policy reduce catastrophic forgetting by externalizing factual knowledge while reserving parametric updates for procedural skills?

## Status

| Component | Status | Notes |
|---|---|---|
| Data pipeline (`Episode`, stream builder, HF loaders) | ✅ Done | ToolAlpaca + NQ-style loaders ready |
| Base model wrapper (HF causal LM) | ✅ Done | Generation, embedding, snapshotting |
| SDFT operator (on-policy self-distillation) | ✅ Done | KL + NLL, teacher conditioned on demo |
| Router policy (MLP + REINFORCE) | ✅ Done | Contextual bandit, EMA baseline |
| Vector memory store (FAISS) | ✅ Done | Sentence-transformer encoder, brute-force fallback |
| Continual learning loop (`SECALoop`) | ✅ Done | Route → execute → evaluate → update router |
| Baselines (SFT-All, SDFT-All, RAG-All, Heuristic) | ✅ Done | Single dispatch API |
| Evaluation (EM, token-F1, forgetting metrics) | ✅ Done | Anchor-suite retention tracking |
| Config system (YAML) | ✅ Done | `configs/default.yaml` |
| Entry-point scripts | ✅ Done | `scripts/run.py`, `scripts/eval.py` |

## Next Steps

1. **Data curation** — Download/prepare actual skill and knowledge datasets. Place JSONL files in `data/skills/` and `data/knowledge/`, or configure HuggingFace dataset names in `configs/default.yaml`.
2. **Smoke-test run** — `pip install -e .` then `python scripts/run.py --mode seca --config configs/default.yaml` on a small stream (set `stream_length: 10`).
3. **Baseline sweeps** — Run all four baselines and compare against SECA on the same stream.
4. **Router ablations** — Vary `lambda_forget`, `mu_cost`; try different router architectures (deeper MLP, attention pooling).
5. **Scaling** — Move from Llama-3.2-1B to a larger model; add LoRA/QLoRA for memory-efficient SDFT updates.
6. **Logging / viz** — Add W&B or TensorBoard integration; plot anchor-perf curves and routing histograms.
7. **Paper figures** — Automate retention curves, routing heatmaps, and per-type breakdown tables from `run_log.jsonl`.

## Codebase Structure

```
seca/
├── configs/
│   └── default.yaml            # all hyperparams (model, SDFT, router, memory, data, eval)
│
├── seca/                       # main Python package
│   ├── data/
│   │   ├── episode.py          # Episode dataclass, EpisodeType enum, build_stream()
│   │   ├── skills.py           # load_skill_episodes() — ToolAlpaca / local JSONL
│   │   └── knowledge.py        # load_knowledge_episodes() — NQ-style / local JSONL
│   │
│   ├── models/
│   │   ├── base.py             # BaseModel — HF causal LM wrapper (generate, embed, snapshot)
│   │   ├── sdft.py             # SDFTOperator — on-policy distillation (UPDATE operator)
│   │   └── router.py           # RouterPolicy (MLP) + RouterTrainer (REINFORCE)
│   │
│   ├── memory/
│   │   └── store.py            # VectorStore — FAISS index + sentence-transformer encoder
│   │
│   ├── train/
│   │   ├── continual.py        # SECALoop — main continual-learning orchestrator
│   │   └── baselines.py        # SFT-All, SDFT-All, RAG-All, Heuristic Router
│   │
│   └── eval/
│       └── metrics.py          # exact_match, token_f1, forgetting(), anchor-suite eval
│
├── scripts/
│   ├── run.py                  # CLI entry point (--mode seca|sft_all|sdft_all|rag_all|heuristic)
│   └── eval.py                 # Post-hoc log analysis (reads run_log.jsonl)
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Key design decisions

- **Episode-centric**: everything revolves around the `Episode` dataclass — loaders produce them, the stream is a list of them, operators consume them, metrics score them.
- **Operator pattern**: UPDATE (SDFT) and STORE (vector DB) are independent operators selected by the router. Easy to add new operators (e.g., LoRA adapter merging).
- **Reward-driven routing**: the router is trained with a contextual bandit objective (`R = Δnew − λ·Δanchor − μ·cost`), avoiding the need to differentiate through SDFT.
- **Baseline parity**: all baselines use the same evaluation pipeline, so numbers are directly comparable.

## Quick Start

```bash
pip install -e .
# small smoke test (10 episodes, will try to load from HF)
python scripts/run.py --mode seca --config configs/default.yaml
# run a baseline for comparison
python scripts/run.py --mode rag_all
# analyse results
python scripts/eval.py logs/run_log.jsonl
```
