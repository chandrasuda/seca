"""Evaluation metrics for SECA.

Covers:
  - Exact match (knowledge QA)
  - Fuzzy / token-F1 match
  - Anchor-suite aggregate (retention / forgetting)
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional

import torch

from seca.data.episode import Episode, EpisodeType


# ── public API ──

def evaluate_episode(
    model,             # BaseModel
    memory,            # VectorStore | None
    episode: Episode,
) -> float:
    """Score the model on a single episode (0-1)."""
    prediction = _get_prediction(model, memory, episode)
    if episode.etype == EpisodeType.KNOWLEDGE:
        return exact_match(prediction, episode.reference)
    else:
        return token_f1(prediction, episode.reference)


def evaluate_anchor_suite(
    model,
    memory,
    anchor_episodes: list[Episode],
) -> float:
    """Mean score across all anchor episodes."""
    if not anchor_episodes:
        return 0.0
    scores = [evaluate_episode(model, memory, ep) for ep in anchor_episodes]
    return sum(scores) / len(scores)


# ── scoring functions ──

def exact_match(pred: str, gold: str) -> float:
    return float(_normalise(pred) == _normalise(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalise(pred).split()
    gold_toks = _normalise(gold).split()
    if not gold_toks:
        return float(not pred_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks) if pred_toks else 0.0
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── retention helpers ──

def forgetting(anchor_scores: list[float]) -> dict:
    """Compute forgetting metrics over time-series of anchor scores.

    Returns max_drop and area_under_curve (higher AUC = better retention).
    """
    if not anchor_scores:
        return {"max_drop": 0.0, "auc": 0.0}
    peak = anchor_scores[0]
    drops = [peak - s for s in anchor_scores]
    return {
        "max_drop": max(drops),
        "auc": sum(anchor_scores) / len(anchor_scores),
    }


# ── internal helpers ──

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation / articles / extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


@torch.no_grad()
def _get_prediction(model, memory, episode: Episode) -> str:
    """Generate model prediction, optionally augmented with retrieval."""
    prompt = episode.prompt
    if memory is not None and len(memory) > 0:
        docs = memory.retrieve(prompt)
        if docs:
            ctx = "\n".join(f"- {d}" for d in docs)
            prompt = f"Context:\n{ctx}\n\nQuestion: {prompt}\nAnswer:"
    outputs = model.generate([prompt], max_new_tokens=128, temperature=0.0, do_sample=False)
    return outputs[0]
