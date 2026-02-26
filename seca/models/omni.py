"""Omni-Teacher — L(θ) = α·KL_sdft + β·KL_sdpo + γ·NLL_gold."""
from __future__ import annotations
from seca.models.base import BaseModel
from seca.models.sdft import SDFTOperator
from seca.models.sdpo import SDPOOperator
from seca.data.problem import CodeProblem
from seca.sandbox.executor import FeedbackBundle


class OmniTeacher:
    def __init__(self, cfg: dict):
        self.alpha = cfg.get("alpha_sdft", 0.4)
        self.beta = cfg.get("beta_sdpo", 0.4)
        self.gamma = cfg.get("gamma_nll", 0.2)
        self.sdft = SDFTOperator(cfg.get("sdft", {}))
        self.sdpo = SDPOOperator(cfg.get("sdpo", {}))

    def loss(self, model: BaseModel, teacher: BaseModel,
             problems: list[CodeProblem], completions: list[str],
             feedback_bundles: list[FeedbackBundle]) -> tuple:
        metrics = {}
        l_sdft, m = self.sdft.loss(model, teacher, problems, completions)
        metrics.update(m)
        l_sdpo, m = self.sdpo.loss(model, teacher, problems, completions, feedback_bundles)
        metrics.update(m)

        gold = [f"{p.format_prompt()}\n{p.gold_solution}" for p in problems]
        enc = model.encode(gold)
        nll = model.model(**enc, labels=enc["input_ids"]).loss
        metrics["nll_gold"] = nll.item()

        total = self.alpha * l_sdft + self.beta * l_sdpo + self.gamma * nll
        metrics["omni_loss"] = total.item()
        return total, metrics