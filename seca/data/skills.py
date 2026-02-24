"""Skill-episode loader (ToolAlpaca-style instruction→action demos)."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from seca.data.episode import Episode, EpisodeType


def load_skill_episodes(
    path_or_hf: str = "tangqiaoyu/ToolAlpaca",
    split: str = "train",
    max_episodes: int | None = None,
) -> list[Episode]:
    """Load skill episodes from a local JSONL or HuggingFace dataset.

    Each record should have at minimum ``instruction`` and ``output`` fields
    (ToolAlpaca format).  Adapt the key mapping below for other schemas.
    """
    path = Path(path_or_hf)
    if path.exists():
        records = _load_local(path, max_episodes)
    else:
        records = _load_hf(path_or_hf, split, max_episodes)

    episodes: list[Episode] = []
    for r in records:
        episodes.append(Episode(
            etype=EpisodeType.SKILL,
            prompt=r.get("instruction", r.get("input", "")),
            reference=r.get("output", r.get("response", "")),
            documents=[],
            meta={"source": path_or_hf},
        ))
    return episodes


# ── internal helpers ──

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
