"""Router policy: π_φ(r | features) → {UPDATE, STORE}.

Trained with REINFORCE (contextual bandit, no trajectory).
"""
from __future__ import annotations

from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from seca.models.base import BaseModel
from seca.data.episode import Episode


class Action(IntEnum):
    UPDATE = 0
    STORE = 1


class RouterPolicy(nn.Module):
    """2-layer MLP that outputs a Bernoulli over {UPDATE, STORE}."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),   # logits for [UPDATE, STORE]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits (B, 2)."""
        return self.net(x)

    def sample(self, x: torch.Tensor) -> tuple[Action, torch.Tensor]:
        """Sample an action and return (action, log_prob)."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        return Action(a.item()), dist.log_prob(a)


class RouterTrainer:
    """REINFORCE with EMA baseline for the router."""

    def __init__(self, policy: RouterPolicy, cfg: dict):
        self.policy = policy
        self.optimizer = Adam(policy.parameters(), lr=cfg.get("lr", 3e-4))
        self.ema_alpha = cfg.get("baseline_ema", 0.99)
        self.baseline = 0.0

    def step(self, log_prob: torch.Tensor, reward: float):
        """Single REINFORCE gradient step."""
        advantage = reward - self.baseline
        loss = -log_prob * advantage

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update EMA baseline
        self.baseline = self.ema_alpha * self.baseline + (1 - self.ema_alpha) * reward
        return {"router_loss": loss.item(), "baseline": self.baseline}
