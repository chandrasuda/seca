"""Dual-engine model: vLLM for inference, HuggingFace for training.

vLLM handles fast rollout generation; HuggingFace handles forward passes and gradients.
Weights are synced from HF to vLLM after each optimizer step.

IMPORTANT: enforce_eager=True must be set — CUDA graphs capture fixed weights and
will silently ignore updates without this flag.

IMPORTANT: VLLM_ENABLE_V1_MULTIPROCESSING=0 must be set so EngineCore runs in-process,
enabling direct model_executor access for weight sync. In multiprocess mode the model
lives in a subprocess and direct attribute access cannot work.
"""
from __future__ import annotations

import copy
import logging
import os
from typing import Any

# Force vLLM V1 engine to run in-process. Must be set before vLLM is imported.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from seca.models.vllm_weight_sync import (
    _assert_tp1,
    _sync_weights_via_direct_access,
)

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
        # enforce_eager=True is required for weight sync — CUDA graphs ignore updates
        log.info("Loading vLLM engine for inference...")
        self.llm = LLM(
            model=name,
            dtype=dtype,
            gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.4),
            max_model_len=vllm_cfg.get("max_model_len", max_len),
            enforce_eager=True,  # Must be True for HF→vLLM weight sync to take effect
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
        self.model.config.use_cache = False  # required for gradient checkpointing
        self.model.gradient_checkpointing_enable()
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
        """Push updated training weights into vLLM. Call after optimizer.step().

        Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 and TP=1 for direct model access.
        """
        _assert_tp1(self.llm)
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        state_dict_items = list(state_dict.items())
        _sync_weights_via_direct_access(self.llm, state_dict_items)

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
