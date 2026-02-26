"""Dual-engine model: vLLM for inference, HuggingFace for training.

vLLM handles fast rollout generation; HuggingFace handles forward passes and gradients.
Weights are synced from HF to vLLM after each optimizer step.
"""
from __future__ import annotations
import copy
import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)


class BaseModel:
    """Dual-engine: vLLM (inference) + HuggingFace (training)."""

    def __init__(
        self,
        name: str,
        dtype: str = "bfloat16",
        max_len: int = 4096,
        vllm_cfg: dict[str, Any] | None = None,
    ):
        self.name = name
        self.max_len = max_len
        self.dtype = getattr(torch, dtype)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vllm_cfg = vllm_cfg or {}

        # 1. Load vLLM first (pre-allocates GPU memory)
        log.info("Loading vLLM engine for inference...")
        self.llm = LLM(
            model=name,
            dtype=dtype,
            gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.4),
            max_model_len=vllm_cfg.get("max_model_len", max_len),
            enforce_eager=vllm_cfg.get("enforce_eager", True),
        )
        log.info("vLLM engine loaded.")

        # 2. Load tokenizer and HuggingFace model for training
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info("Loading HuggingFace model for training...")
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.train()
        log.info("HuggingFace model loaded.")

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | None = None,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> list[str]:
        """Generate completions via vLLM (inference only, no gradients)."""
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=temperature if do_sample else 0.0,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
        outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
        return [o.outputs[0].text for o in outputs]

    def sync_vllm_weights(self) -> None:
        """Push updated training weights into vLLM. Call after optimizer.step()."""
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        try:
            llm_model = (
                self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            )
            llm_model.load_weights(list(state_dict.items()))
        except AttributeError as e:
            # vLLM API may vary by version; try alternative path
            log.warning("Primary weight sync path failed (%s), trying fallback", e)
            try:
                workers = self.llm.llm_engine.workers
                if workers:
                    worker = workers[0]
                    if hasattr(worker, "model_runner") and worker.model_runner is not None:
                        model = getattr(worker.model_runner, "model", None)
                        if model is not None and hasattr(model, "load_weights"):
                            model.load_weights(list(state_dict.items()))
            except Exception as e2:
                log.error("Weight sync failed: %s", e2)
                raise

    def snapshot(self) -> BaseModel:
        """Return a frozen copy of the HF model for use as EMA teacher.
        No vLLM copy needed — teacher only does forward passes via HF."""
        clone = copy.copy(self)
        clone.model = copy.deepcopy(self.model)
        clone.model.eval()
        for p in clone.model.parameters():
            p.requires_grad_(False)
        # Snapshot does not get vLLM reference — teacher never generates
        clone.llm = self.llm  # Share vLLM for any edge-case generation; teacher uses HF
        return clone

    def encode(self, texts: list[str], **kw) -> dict[str, torch.Tensor]:
        """Tokenize for HF forward pass."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
            **kw,
        ).to(self.device)
