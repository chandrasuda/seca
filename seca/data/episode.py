"""Episode abstraction and continual-stream builder."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class EpisodeType(str, Enum):
    SKILL = "skill"
    KNOWLEDGE = "knowledge"


@dataclass
class Episode:
    """Single continual-learning episode.

    Fields
    ------
    etype : EpisodeType
        Whether this episode targets skill acquisition or knowledge.
    prompt : str
        Input / instruction text.
    reference : str
        Gold completion (demo for skills, answer for knowledge).
    documents : list[str]
        Supporting docs (used by STORE operator & RAG baselines).
    meta : dict
        Arbitrary metadata (source dataset, timestamp, …).
    """
    etype: EpisodeType
    prompt: str
    reference: str
    documents: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ── featurisation for the router ──
    def feature_text(self) -> str:
        """Text representation used to compute router features."""
        return f"[{self.etype.value}] {self.prompt[:512]}"


def build_stream(
    skill_episodes: list[Episode],
    knowledge_episodes: list[Episode],
    length: int,
    skill_ratio: float = 0.5,
    seed: int = 42,
) -> list[Episode]:
    """Build an interleaved continual stream of *length* episodes.

    Samples with replacement from each pool according to *skill_ratio*.
    """
    rng = random.Random(seed)
    stream: list[Episode] = []
    for _ in range(length):
        if rng.random() < skill_ratio:
            stream.append(rng.choice(skill_episodes))
        else:
            stream.append(rng.choice(knowledge_episodes))
    return stream
