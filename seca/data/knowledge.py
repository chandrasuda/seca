"""Knowledge-episode loader (evidence-linked QA)."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from seca.data.episode import Episode, EpisodeType


def load_knowledge_episodes(
    path_or_hf: str = "natural_questions",
    split: str = "train",
    max_episodes: int | None = None,
) -> list[Episode]:
    """Load knowledge QA episodes.

    Expects records with ``question``, ``answer``, and optionally
    ``context`` / ``document`` fields.
    """
    path = Path(path_or_hf)
    if path.exists():
        records = _load_local(path, max_episodes)
    else:
        records = _load_hf(path_or_hf, split, max_episodes)

    episodes: list[Episode] = []
    for r in records:
        question = r.get("question", r.get("input", ""))
        # handle NQ-style nested answers
        answer = r.get("answer", "")
        if isinstance(answer, dict):
            answer = answer.get("value", str(answer))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        docs = r.get("documents", [])
        if isinstance(docs, str):
            docs = [docs]
        ctx = r.get("context", "")
        if ctx and not docs:
            docs = [ctx]

        episodes.append(Episode(
            etype=EpisodeType.KNOWLEDGE,
            prompt=question,
            reference=str(answer),
            documents=docs,
            meta={"source": path_or_hf},
        ))
    return episodes


def _load_local(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _load_hf(name: str, split: str, limit: int | None) -> list[dict]:
    ds = load_dataset(name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return list(ds)
