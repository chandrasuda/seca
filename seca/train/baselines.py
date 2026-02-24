"""Baseline runners: SFT-All, SDFT-All, RAG-All, Heuristic Router."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.optim import AdamW

from seca.data.episode import Episode, EpisodeType
from seca.models.base import BaseModel
from seca.models.sdft import SDFTOperator
from seca.memory.store import VectorStore
from seca.eval.metrics import evaluate_episode, evaluate_anchor_suite

log = logging.getLogger(__name__)


def run_baseline(
    name: str,
    model: BaseModel,
    stream: list[Episode],
    anchor_suite: list[Episode],
    cfg: dict,
) -> list[dict]:
    """Dispatch to the correct baseline by *name*."""
    runners = {
        "sft_all": _run_sft_all,
        "sdft_all": _run_sdft_all,
        "rag_all": _run_rag_all,
        "heuristic": _run_heuristic,
    }
    if name not in runners:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(runners)}")
    return runners[name](model, stream, anchor_suite, cfg)


# ── SFT-All: naive sequential fine-tuning ──

def _run_sft_all(
    model: BaseModel, stream: list[Episode],
    anchor_suite: list[Episode], cfg: dict,
) -> list[dict]:
    optimizer = AdamW(model.model.parameters(), lr=cfg["sdft"]["lr"])
    history: list[dict] = []
    prev_anchor = evaluate_anchor_suite(model, None, anchor_suite) if anchor_suite else 0.0

    for t, ep in enumerate(stream):
        # simple NLL on gold
        enc = model.tokenizer(
            [f"{ep.prompt}\n{ep.reference}"],
            return_tensors="pt", padding=True,
            truncation=True, max_length=model.max_len,
        ).to(model.device)
        loss = model.model(**enc, labels=enc["input_ids"]).loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        new_perf = evaluate_episode(model, None, ep)
        anchor_perf = prev_anchor
        if anchor_suite and (t + 1) % cfg["eval"]["anchor_eval_every"] == 0:
            anchor_perf = evaluate_anchor_suite(model, None, anchor_suite)
        prev_anchor = anchor_perf

        history.append({"t": t, "baseline": "sft_all", "new_perf": new_perf,
                        "anchor_perf": anchor_perf, "loss": loss.item()})
    return history


# ── SDFT-All: apply SDFT to every episode ──

def _run_sdft_all(
    model: BaseModel, stream: list[Episode],
    anchor_suite: list[Episode], cfg: dict,
) -> list[dict]:
    sdft = SDFTOperator(cfg["sdft"])
    history: list[dict] = []
    prev_anchor = evaluate_anchor_suite(model, None, anchor_suite) if anchor_suite else 0.0

    for t, ep in enumerate(stream):
        metrics = sdft.update(model, ep)
        new_perf = evaluate_episode(model, None, ep)
        anchor_perf = prev_anchor
        if anchor_suite and (t + 1) % cfg["eval"]["anchor_eval_every"] == 0:
            anchor_perf = evaluate_anchor_suite(model, None, anchor_suite)
        prev_anchor = anchor_perf

        history.append({"t": t, "baseline": "sdft_all", "new_perf": new_perf,
                        "anchor_perf": anchor_perf, **metrics})
    return history


# ── RAG-All: no weight updates, always store + retrieve ──

def _run_rag_all(
    model: BaseModel, stream: list[Episode],
    anchor_suite: list[Episode], cfg: dict,
) -> list[dict]:
    mem = VectorStore(cfg["memory"]["embedding_model"], cfg["memory"]["top_k"])
    history: list[dict] = []
    anchor_perf = evaluate_anchor_suite(model, mem, anchor_suite) if anchor_suite else 0.0

    for t, ep in enumerate(stream):
        docs = ep.documents or [f"Q: {ep.prompt}\nA: {ep.reference}"]
        mem.add(docs)
        new_perf = evaluate_episode(model, mem, ep)
        if anchor_suite and (t + 1) % cfg["eval"]["anchor_eval_every"] == 0:
            anchor_perf = evaluate_anchor_suite(model, mem, anchor_suite)

        history.append({"t": t, "baseline": "rag_all", "new_perf": new_perf,
                        "anchor_perf": anchor_perf, "mem_size": len(mem)})
    return history


# ── Heuristic Router: KNOWLEDGE→STORE, SKILL→UPDATE ──

def _run_heuristic(
    model: BaseModel, stream: list[Episode],
    anchor_suite: list[Episode], cfg: dict,
) -> list[dict]:
    sdft = SDFTOperator(cfg["sdft"])
    mem = VectorStore(cfg["memory"]["embedding_model"], cfg["memory"]["top_k"])
    history: list[dict] = []
    prev_anchor = evaluate_anchor_suite(model, mem, anchor_suite) if anchor_suite else 0.0

    for t, ep in enumerate(stream):
        if ep.etype == EpisodeType.SKILL:
            metrics = sdft.update(model, ep)
            action = "UPDATE"
        else:
            docs = ep.documents or [f"Q: {ep.prompt}\nA: {ep.reference}"]
            mem.add(docs)
            metrics = {"store_docs": len(docs)}
            action = "STORE"

        new_perf = evaluate_episode(model, mem, ep)
        anchor_perf = prev_anchor
        if anchor_suite and (t + 1) % cfg["eval"]["anchor_eval_every"] == 0:
            anchor_perf = evaluate_anchor_suite(model, mem, anchor_suite)
        prev_anchor = anchor_perf

        history.append({"t": t, "baseline": "heuristic", "action": action,
                        "new_perf": new_perf, "anchor_perf": anchor_perf, **metrics})
    return history
