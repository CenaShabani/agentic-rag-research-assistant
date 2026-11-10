"""The cross-encoder reranker — accuracy bought with latency.

Everything else in this project is a **bi-encoder** design: the document is
embedded once at index time, the query is embedded at query time, and relevance
is the similarity between two vectors that never met. That is what makes it fast
— all the document work happens before the query arrives — and it is also its
ceiling. The document's vector was computed without knowing the query, so it has
to be a summary of everything the document might ever be asked about.

A **cross-encoder** does the opposite. It takes the query and the document
*together* as one input, runs them through a transformer that can attend across
both, and outputs a single relevance score. Nothing can be precomputed: every
(query, document) pair is a full forward pass at query time.

    bi-encoder    : encode(query) . encode(document)      <- one vector each
    cross-encoder : model(query + document) -> score      <- one pass per pair

The cross-encoder ranks better, often by a lot, because it can see interactions a
pair of independent summaries cannot. And it is orders of magnitude slower,
because the cost is per candidate rather than per query.

That is why real systems use both, in the arrangement this module implements:
a cheap first stage retrieves a shortlist, and the expensive model only rescores
that shortlist. The size of the shortlist (`config.RERANK_CANDIDATES`) is the
dial between the two. Measuring what that dial buys and what it costs is the
point of the benchmark.

The model is loaded lazily and cached, exactly like `embeddings.get_model` — the
~90 MB download happens on first use, not on import.
"""

from __future__ import annotations

import functools

import numpy as np

from . import config


@functools.lru_cache(maxsize=1)
def get_cross_encoder():
    """Load (and cache) the cross-encoder.

    `ms-marco-MiniLM-L-6-v2` is trained on MS MARCO passage ranking — real search
    queries against real passages — which is the task shape this benchmark
    measures. Import is inside the function so importing this module stays cheap
    and dependency-free, matching the pattern in `embeddings.py`.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def rerank_scores(query: str, documents: list[str]) -> np.ndarray:
    """Score every (query, document) pair. Higher is more relevant.

    Returned scores are raw logits, not probabilities and not bounded to any
    range. That is fine and deliberate — they are only ever used to *order* the
    candidates, never compared against another arm's scores or thresholded.
    """
    if not documents:
        return np.zeros(0, dtype=np.float32)
    model = get_cross_encoder()
    scores = model.predict(
        [(query, d) for d in documents],
        show_progress_bar=False,
    )
    return np.asarray(scores, dtype=np.float32)
