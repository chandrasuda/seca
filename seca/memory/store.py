"""Vector memory store (FAISS) for the STORE operator.

Handles embedding, insertion, and retrieval of episode documents.
Uses a sentence-transformer for dense embeddings.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    faiss = None  # graceful fallback to brute-force


class VectorStore:
    """Flat L2 FAISS index backed by a sentence-transformer encoder."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", top_k: int = 5):
        self.encoder = SentenceTransformer(model_name)
        self.dim = self.encoder.get_sentence_embedding_dimension()
        self.top_k = top_k
        self.texts: list[str] = []

        if faiss is not None:
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            self._vecs: list[np.ndarray] = []
            self.index = None

    # ── write ──

    def add(self, documents: list[str]):
        """Embed and insert documents."""
        if not documents:
            return
        vecs = self.encoder.encode(documents, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype=np.float32)
        if self.index is not None:
            self.index.add(vecs)
        else:
            self._vecs.append(vecs)
        self.texts.extend(documents)

    # ── read ──

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[str]:
        """Return top-k most similar documents to *query*."""
        k = top_k or self.top_k
        if not self.texts:
            return []
        qvec = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)

        if self.index is not None:
            _, idxs = self.index.search(qvec, min(k, len(self.texts)))
            return [self.texts[i] for i in idxs[0] if i < len(self.texts)]
        else:
            # brute-force fallback
            all_vecs = np.concatenate(self._vecs, axis=0)
            sims = all_vecs @ qvec.T
            top = np.argsort(sims.squeeze())[::-1][:k]
            return [self.texts[i] for i in top]

    def __len__(self) -> int:
        return len(self.texts)

    def clear(self):
        self.texts.clear()
        if self.index is not None:
            self.index.reset()
        else:
            self._vecs.clear()
