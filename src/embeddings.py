"""Turn text into vectors with a local sentence-transformer.

An *embedding* is the whole game behind semantic retrieval: it maps a piece of
text to a point in a high-dimensional space such that texts about similar things
land near each other. "How do transformers attend to tokens?" and "the attention
mechanism computes weighted sums over the sequence" have almost no words in
common, but their embeddings are close — so a query can find a chunk by *meaning*,
not by keyword overlap.

We use `all-MiniLM-L6-v2` (chosen and justified in config.py). It runs on CPU,
needs no API key, and is deterministic given a fixed seed — which is what keeps
the entire retrieval path reproducible on any machine.

The model is loaded lazily and cached: importing this module is cheap, and the
(~90 MB) model downloads only the first time it is actually used.
"""

from __future__ import annotations

import functools

import numpy as np

from . import config


@functools.lru_cache(maxsize=1)
def get_model():
    """Load (and cache) the sentence-transformer.

    Import is inside the function so that merely importing this module — which
    the tests and the config-reading tools do — does not require
    sentence-transformers to be installed or the model to be downloaded.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into an (n, EMBED_DIM) float32 array.

    Vectors are L2-normalized so that an inner product between two of them equals
    their cosine similarity. That equivalence is what lets the FAISS index use a
    fast inner-product search and have it mean "most semantically similar" —
    see vectorstore.py.
    """
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,      # unit vectors → inner product = cosine
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query into a (1, EMBED_DIM) array, ready for search."""
    return embed_texts([query])
