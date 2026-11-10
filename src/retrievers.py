"""The four retrieval arms, behind one interface.

The benchmark compares four ways of answering "which chunks are relevant to this
query?". Each is a `Retriever` with a `name` and a `search(query, k)` returning
the same `Retrieved` objects `VectorStore.search` already returns — so the metric
code, the figures, and the notebook never learn which arm they are looking at.

    dense    -> the current system: embed the query, cosine-search FAISS
    bm25     -> lexical term overlap, no neural model at all
    hybrid   -> dense and bm25 fused by reciprocal-rank fusion
    rerank   -> hybrid's shortlist, rescored by a cross-encoder

**The single-definition invariant.** `vectorstore.py` says it is "the single
definition of search the corpus", and `rag.py` and `agent.py` both depend on
that. Nothing here breaks it: `DenseRetriever` *delegates* to
`VectorStore.search` rather than reimplementing cosine search. If it
reimplemented it, the benchmark would be measuring a retriever the shipped system
does not use, and every published number would be about the wrong thing.
`tests/test_retrievers.py` pins this by asserting the two return identical hits.

**Why the arms are ordered this way.** Each one is the previous one plus exactly
one idea, so the benchmark can attribute a change to a cause:

  1. dense alone     — does semantic matching work here?
  2. bm25 alone      — could plain word overlap have done it?
  3. hybrid          — are their mistakes different, or the same mistakes twice?
  4. rerank          — does reading query and document jointly beat comparing
                       two independent summaries, and what does it cost?

An arm that adds two ideas at once is unattributable, which is why there is no
"hybrid + query expansion + rerank" arm.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from . import config
from .lexical import BM25
from .vectorstore import Retrieved, VectorStore


class Retriever(Protocol):
    """Anything that can rank chunks for a query.

    Deliberately the same shape as `VectorStore.search`, so the store itself
    already satisfies it and no adapter is needed for the shipped system.
    """

    name: str

    def search(self, query: str, k: int = config.TOP_K) -> list[Retrieved]:
        ...


def _row_to_retrieved(meta: pd.DataFrame, row: int, score: float) -> Retrieved:
    """Build a `Retrieved` from a metadata row index and a score.

    Every arm ends up here, which is what guarantees they all return the same
    citation fields no matter how they ranked.
    """
    m = meta.iloc[int(row)]
    return Retrieved(
        chunk_id=m["chunk_id"],
        arxiv_id=m["arxiv_id"],
        title=m["title"],
        text=m["text"],
        score=float(score),
    )


# --------------------------------------------------------------------------- #
# 1. Dense — today's system
# --------------------------------------------------------------------------- #
class DenseRetriever:
    """The shipped retriever: embed the query, exact cosine search over FAISS.

    A thin delegation on purpose. This arm must be *the same code path* the RAG
    pipeline and the agent use, or the benchmark is measuring a lookalike.
    """

    name = "dense"

    def __init__(self, store: VectorStore):
        self._store = store

    def search(self, query: str, k: int = config.TOP_K) -> list[Retrieved]:
        return self._store.search(query, k=k)


# --------------------------------------------------------------------------- #
# 2. BM25 — the lexical baseline
# --------------------------------------------------------------------------- #
class BM25Retriever:
    """Rank by term overlap. No embeddings, no model, no GPU.

    Built from the vector store's own metadata frame so the two arms index the
    same chunks in the same row order — which is what lets the hybrid arm fuse
    them by row id.
    """

    name = "bm25"

    def __init__(self, store: VectorStore):
        self._meta = store.meta
        self._bm25 = BM25(self._meta["text"].tolist())

    def search(self, query: str, k: int = config.TOP_K) -> list[Retrieved]:
        return [
            _row_to_retrieved(self._meta, row, score)
            for row, score in self._bm25.top_k(query, k)
        ]


# --------------------------------------------------------------------------- #
# 3. Hybrid — reciprocal-rank fusion
# --------------------------------------------------------------------------- #
class HybridRRFRetriever:
    """Fuse the dense and lexical rankings by reciprocal-rank fusion.

    **Why fuse ranks and not scores.** A cosine similarity lives in [-1, 1]; a
    BM25 score is unbounded and depends on the collection's term statistics. They
    are not on a common scale and no fixed weighting makes them comparable, so
    adding or averaging them is meaningless. RRF throws the scores away and keeps
    only the *positions*:

        score(d) = SUM over arms of  1 / (RRF_K + rank of d in that arm)

    A document ranked 1st by one arm and unranked by the other still scores well;
    a document ranked 3rd by *both* can beat it. That is the behaviour worth
    having — agreement between two different kinds of evidence is a stronger
    signal than one arm's confidence.

    `RRF_K = 60` (Cormack et al., 2009) damps the top ranks so a single arm's
    confident-but-wrong first hit cannot dominate the fusion.

    Each arm is asked for a *deeper* list than the caller wants, because a
    document must be ranked by at least one arm to be fusable at all — fusing two
    top-10 lists can only ever consider 20 candidates.
    """

    name = "hybrid"

    def __init__(self, store: VectorStore, depth: int = config.RERANK_CANDIDATES):
        self._store = store
        self._meta = store.meta
        self._bm25 = BM25(self._meta["text"].tolist())
        self._depth = depth

    def _fused_rows(self, query: str, depth: int) -> list[tuple[int, float]]:
        """Fused (row, rrf_score) pairs, best first — the shared ranking step."""
        dense_rows = self._store.search_rows(query, k=depth)
        bm25_rows = self._bm25.top_k(query, depth)

        fused: dict[int, float] = {}
        for ranking in (dense_rows, bm25_rows):
            for rank, (row, _score) in enumerate(ranking, start=1):
                fused[row] = fused.get(row, 0.0) + 1.0 / (config.RRF_K + rank)

        return sorted(fused.items(), key=lambda kv: -kv[1])

    def search(self, query: str, k: int = config.TOP_K) -> list[Retrieved]:
        ranked = self._fused_rows(query, self._depth)[:k]
        return [_row_to_retrieved(self._meta, row, score) for row, score in ranked]


# --------------------------------------------------------------------------- #
# 4. Rerank — cross-encoder over the hybrid shortlist
# --------------------------------------------------------------------------- #
class RerankRetriever:
    """Retrieve a shortlist cheaply, then rescore it with a cross-encoder.

    The standard production arrangement, and the only arm here whose cost scales
    with the shortlist rather than the collection. It reranks the *hybrid* arm's
    candidates rather than the dense arm's, so recall going into the reranker is
    as high as this project can make it — a cross-encoder can only reorder what it
    is given, and cannot rescue a document the first stage never retrieved.
    """

    name = "rerank"

    def __init__(self, hybrid: HybridRRFRetriever,
                 candidates: int = config.RERANK_CANDIDATES):
        self._hybrid = hybrid
        self._meta = hybrid._meta
        self._candidates = candidates

    def search(self, query: str, k: int = config.TOP_K) -> list[Retrieved]:
        from .rerank import rerank_scores

        shortlist = self._hybrid._fused_rows(query, self._candidates)[:self._candidates]
        if not shortlist:
            return []

        rows = [row for row, _ in shortlist]
        texts = [self._meta.iloc[r]["text"] for r in rows]
        scores = rerank_scores(query, texts)

        order = np.argsort(-scores, kind="stable")[:k]
        return [_row_to_retrieved(self._meta, rows[i], float(scores[i])) for i in order]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_all(store: VectorStore) -> list[Retriever]:
    """Construct all four arms, sharing what can be shared.

    The BM25 index is built twice (once for the lexical arm, once inside the
    hybrid arm) — a couple of seconds, and worth it to keep each arm a
    self-contained object that can be constructed and reasoned about alone.
    Returned in increasing order of cost, which is the order the figures use.
    """
    hybrid = HybridRRFRetriever(store)
    return [
        DenseRetriever(store),
        BM25Retriever(store),
        hybrid,
        RerankRetriever(hybrid),
    ]
