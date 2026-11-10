"""Shared test fixtures.

The heavy pieces of this project — the sentence-transformer and the Anthropic
API — are both replaced here with tiny deterministic fakes, so the whole suite
runs fast, offline, and with no API key. That is the point: the tests pin the
project's *logic* (chunking, index round-trips, the agent loop, the metrics),
which must be correct regardless of which embedding model or LLM sits behind it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src import config


# --------------------------------------------------------------------------- #
# A toy embedder — deterministic, dependency-free, good enough for tests.
# --------------------------------------------------------------------------- #
def _toy_embed(texts: list[str]) -> np.ndarray:
    """Hash each lowercased word into a fixed-width bag-of-words vector.

    This is not a good embedding — it is a deterministic stand-in so retrieval
    tests don't download a 90 MB model. Documents that share words end up with
    similar vectors, which is enough to test that "planted query finds planted
    doc" and that the index round-trips.
    """
    dim = config.EMBED_DIM
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for word in t.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
            vecs[i, h] += 1.0
        norm = np.linalg.norm(vecs[i])
        if norm > 0:
            vecs[i] /= norm  # unit-normalize so inner product == cosine
    return vecs


@pytest.fixture
def toy_embeddings(monkeypatch):
    """Patch the embedder used by the vector store with the toy embedder."""
    from src import embeddings, vectorstore

    monkeypatch.setattr(vectorstore, "embed_texts", _toy_embed)
    monkeypatch.setattr(vectorstore, "embed_query", lambda q: _toy_embed([q]))
    monkeypatch.setattr(embeddings, "embed_texts", _toy_embed)
    return _toy_embed


@pytest.fixture
def small_chunks() -> pd.DataFrame:
    """A tiny planted corpus with clearly-separated topics."""
    return pd.DataFrame(
        {
            "chunk_id": ["a:0", "b:0", "c:0"],
            "arxiv_id": ["1111.0001", "2222.0002", "3333.0003"],
            "title": ["Transformers", "Vitamins", "Volcanoes"],
            "text": [
                "the transformer architecture uses self attention over tokens",
                "vitamin c is an antioxidant found in citrus fruit",
                "a volcano erupts when magma rises through the crust",
            ],
        }
    )
