"""Tokenizer utilities for Qwen and other models with thinking mode."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def make_no_thinking_tokenizer(tokenizer: "PreTrainedTokenizerBase") -> "PreTrainedTokenizerBase":
    """Wrap tokenizer so apply_chat_template uses enable_thinking=False.

    Qwen3 models output <think> tokens by default. This wrapper ensures
    the model generates only the direct response (code), not reasoning.
    """
    orig_apply = tokenizer.apply_chat_template

    def _apply(messages, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return orig_apply(messages, **kwargs)

    if hasattr(tokenizer, "apply_chat_template"):
        tokenizer.apply_chat_template = _apply
    return tokenizer
