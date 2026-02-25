"""HuggingFace causal LM wrapper: generate, encode, snapshot."""
from __future__ import annotations
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel:
    def __init__(self, name: str, dtype: str = "bfloat16", max_len: int = 2048):
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

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 512,
                 temperature: float = 1.0, do_sample: bool = True) -> list[str]:
        enc = self.encode(prompts)
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=do_sample, pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True)

    def snapshot(self) -> BaseModel:
        """Frozen copy for use as teacher."""
        clone = copy.copy(self)
        clone.model = copy.deepcopy(self.model)
        clone.model.eval()
        for p in clone.model.parameters():
            p.requires_grad_(False)
        return clone

    def encode(self, texts: list[str], **kw):
        return self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_len, **kw,
        ).to(self.device)
