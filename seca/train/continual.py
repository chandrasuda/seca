"""Main continual-learning loop for SECA."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from seca.data.episode import Episode, EpisodeType
from seca.models.base import BaseModel
from seca.models.sdft import SDFTOperator
from seca.models.router import RouterPolicy, RouterTrainer, Action
from seca.memory.store import VectorStore
from seca.eval.metrics import evaluate_episode, evaluate_anchor_suite

log = logging.getLogger(__name__)


class SECALoop:
    """Orchestrates the full SECA continual-learning run.

    For each episode:
      1. Router selects action (UPDATE / STORE).
      2. Execute the chosen operator.
      3. Evaluate on new episode + periodic anchor suite.
      4. Compute reward and update router via REINFORCE.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        # ── base LM ──
        self.model = BaseModel(**cfg["model"])

        # ── operators ──
        self.sdft = SDFTOperator(cfg["sdft"])
        self.memory = VectorStore(
            model_name=cfg["memory"]["embedding_model"],
            top_k=cfg["memory"]["top_k"],
        )

        # ── router ──
        feat_dim = cfg["router"].get("feature_dim") or self.model.hidden_size
        self.router = RouterPolicy(feat_dim, cfg["router"]["hidden_dim"])
        self.router.to(self.model.device)
        self.router_trainer = RouterTrainer(self.router, cfg["router"])

        # ── reward params ──
        self.lam = cfg["reward"]["lambda_forget"]
        self.mu = cfg["reward"]["mu_cost"]

        # ── bookkeeping ──
        self.anchor_suite: list[Episode] = []
        self.eval_every = cfg["eval"]["anchor_eval_every"]
        self.log_dir = Path(cfg["eval"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []
        self.prev_anchor_perf: float = 0.0

    def set_anchor_suite(self, episodes: list[Episode]):
        self.anchor_suite = episodes
        self.prev_anchor_perf = evaluate_anchor_suite(
            self.model, self.memory, self.anchor_suite,
        )

    # ── main loop ──

    def run(self, stream: list[Episode]):
        for t, episode in enumerate(stream):
            log.info(f"Episode {t}/{len(stream)}  type={episode.etype.value}")

            # 1. extract features
            feat = self.model.embed([episode.feature_text()])  # (1, D)

            # 2. router decision
            action, log_prob = self.router.sample(feat)
            log.info(f"  action={action.name}")

            # 3. execute operator
            op_metrics = self._execute(action, episode)

            # 4. evaluate new-episode performance
            new_perf = evaluate_episode(self.model, self.memory, episode)

            # 5. periodic anchor evaluation
            anchor_perf = self.prev_anchor_perf
            if self.anchor_suite and (t + 1) % self.eval_every == 0:
                anchor_perf = evaluate_anchor_suite(
                    self.model, self.memory, self.anchor_suite,
                )

            # 6. compute reward
            delta_new = new_perf
            delta_anchor = self.prev_anchor_perf - anchor_perf  # positive = forgetting
            cost = 1.0 if action == Action.UPDATE else 0.0
            reward = delta_new - self.lam * delta_anchor - self.mu * cost

            # 7. router gradient
            router_metrics = self.router_trainer.step(log_prob, reward)

            # 8. log
            entry = {
                "t": t,
                "etype": episode.etype.value,
                "action": action.name,
                "new_perf": new_perf,
                "anchor_perf": anchor_perf,
                "reward": reward,
                **op_metrics,
                **router_metrics,
            }
            self.history.append(entry)
            log.info(f"  reward={reward:.4f}  new={new_perf:.4f}  anchor={anchor_perf:.4f}")

            self.prev_anchor_perf = anchor_perf

        self._save_log()
        return self.history

    # ── operator dispatch ──

    def _execute(self, action: Action, episode: Episode) -> dict:
        if action == Action.UPDATE:
            return self.sdft.update(self.model, episode)
        else:
            # STORE: insert docs (or prompt+reference as doc) into memory
            docs = episode.documents or [
                f"Q: {episode.prompt}\nA: {episode.reference}"
            ]
            self.memory.add(docs)
            return {"store_docs": len(docs)}

    def _save_log(self):
        out = self.log_dir / "run_log.jsonl"
        with open(out, "w") as f:
            for entry in self.history:
                f.write(json.dumps(entry) + "\n")
        log.info(f"Log saved → {out}")
