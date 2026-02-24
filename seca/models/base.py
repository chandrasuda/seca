"""Thin wrapper around a HuggingFace causal LM."""
from __future__ import annotations

import copy
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel:
    """Loads, wraps, and exposes a causal LM for SECA operators.

    Provides helpers for:
    * tokenisation
    * forward pass (logits / loss)
    * sampling completions
    * snapshotting (for SDFT teacher)
    """

    def __init__(self, name: str, dtype: str = "bfloat16", max_len: int = 1024):
        self.name = name
        self.max_len = max_len
        self.dtype = getattr(torch, dtype)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=self.dtype, device_map="auto",
        )
        self.model.train()

    # ── generation ──

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> list[str]:
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_len,
        ).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # decode only the new tokens
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    # ── forward (logits + optional loss) ──

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return self.model(input_ids=input_ids, labels=labels)

    # ── snapshot for teacher ──

    def snapshot(self) -> BaseModel:
        """Return a frozen copy (used as SDFT teacher)."""
        clone = copy.copy(self)
        clone.model = copy.deepcopy(self.model)
        clone.model.eval()
        for p in clone.model.parameters():
            p.requires_grad_(False)
        return clone

    # ── hidden-state extraction (for router features) ──

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        """Mean-pooled last hidden state — cheap episode features."""
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_len,
        ).to(self.device)
        hs = self.model(
            **enc, output_hidden_states=True
        ).hidden_states[-1]              # (B, L, D)
        mask = enc["attention_mask"].unsqueeze(-1)
        return (hs * mask).sum(1) / mask.sum(1)  # (B, D)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size
