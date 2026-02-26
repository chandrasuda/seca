"""vLLM V1 engine weight sync extension.

Syncs HuggingFace training weights into vLLM via direct model_executor access.
Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 so EngineCore runs in-process.
Both sync paths assume TP=1; multi-GPU tensor parallelism would require
proper sharded weight transfer.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger(__name__)


def _assert_tp1(llm: Any) -> None:
    """Raise if TP > 1. Both sync paths write full HF state dict; TP > 1 would corrupt shards."""
    try:
        llm_engine = getattr(llm, "llm_engine", None)
        if llm_engine is None:
            return
        vllm_config = getattr(llm_engine, "vllm_config", None)
        if vllm_config is None:
            return
        parallel_config = getattr(vllm_config, "parallel_config", None)
        if parallel_config is None:
            return
        tp = getattr(parallel_config, "tensor_parallel_size", 1)
        if tp > 1:
            raise RuntimeError(
                f"sync_vllm_weights is only safe for TP=1; tensor_parallel_size={tp}. "
                "Use collective_rpc with proper sharded weight transfer for TP > 1."
            )
    except RuntimeError:
        raise
    except Exception:
        pass


def _sync_weights_via_direct_access(
    llm: Any, state_dict_items: list[tuple[str, torch.Tensor]]
) -> None:
    """Walk llm.llm_engine to model_executor -> worker -> model_runner.model and load weights.

    Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 so EngineCore runs in-process.
    Raises AttributeError with helpful message if traversal fails.
    """
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        raise AttributeError("llm has no llm_engine")

    # V1 in-process: model_executor is set on engine for v0 compatibility
    executor = getattr(engine, "model_executor", None)
    if executor is None:
        # Belt-and-suspenders: try V1 engine_core path (InprocClient wraps core)
        engine_core = getattr(engine, "engine_core", None)
        inner = getattr(engine_core, "engine_core", engine_core) if engine_core else None
        executor = getattr(inner, "model_executor", None) if inner else None

    if executor is None:
        raise AttributeError(
            "Cannot find model_executor on engine. "
            "Ensure VLLM_ENABLE_V1_MULTIPROCESSING=0 is set before LLM() is constructed."
        )

    worker = None
    for attr in ("driver_worker", "worker", "_worker"):
        worker = getattr(executor, attr, None)
        if worker is not None:
            break

    if worker is None:
        attrs = [a for a in dir(executor) if not a.startswith("__")]
        raise AttributeError(
            f"Cannot find worker on {type(executor).__name__}. "
            f"Attrs: {attrs}"
        )

    # Some paths have nested worker.worker
    if hasattr(worker, "worker") and worker.worker is not None:
        worker = worker.worker

    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        raise AttributeError(
            f"worker {type(worker).__name__} has no model_runner. "
            f"Attrs: {[a for a in dir(worker) if not a.startswith('__')]}"
        )

    model = getattr(model_runner, "model", None)
    if model is None or not hasattr(model, "load_weights"):
        raise AttributeError(
            f"model_runner has no model with load_weights. "
            f"model={model!r}, has_load_weights={hasattr(model, 'load_weights') if model else False}"
        )

    model.load_weights(state_dict_items)
    log.info("sync_vllm_weights succeeded via direct access")
