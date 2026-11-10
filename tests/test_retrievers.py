"""Tests for the four retrieval arms.

The cross-encoder is stubbed rather than downloaded: these tests pin the *wiring*
— that each arm returns well-formed, ordered, deduplicated hits over the shared
metadata frame — which must hold regardless of which model does the scoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config, retrievers
from src.vectorstore import Retrieved, VectorStore


@pytest.fixture
def store(small_chunks, toy_embeddings) -> VectorStore:
    return VectorStore.build(small_chunks)


@pytest.fixture
def stub_cross_encoder(monkeypatch):
    """Score candidates by length — deterministic, offline, no 90 MB download.

    A nonsense relevance function on purpose: what is being tested is that the
    reranker reorders its shortlist by whatever the model says, not that this
    particular model is any good.
    """
    from src import rerank

    monkeypatch.setattr(
        rerank, "rerank_scores",
        lambda query, documents: np.array([float(len(d)) for d in documents],
                                          dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# The invariant that matters most
# --------------------------------------------------------------------------- #
def test_dense_arm_is_the_shipped_retriever(store):
    """The dense arm must be byte-identical to `VectorStore.search`.

    `vectorstore.py` calls itself "the single definition of search the corpus",
    and `rag.py` and `agent.py` both rely on that. If the benchmark's dense arm
    re-implemented cosine search, every published number would describe a
    retriever the shipped system does not use. This test is what stops that
    happening by accident.
    """
    arm = retrievers.DenseRetriever(store)
    query = "self attention over tokens"
    assert arm.search(query, k=3) == store.search(query, k=3)


def test_search_rows_and_search_agree(store):
    """The two entry points must return the same rows in the same order."""
    query = "self attention over tokens"
    rows = [r for r, _ in store.search_rows(query, k=3)]
    hits = store.search(query, k=3)
    assert [store.meta.iloc[r]["chunk_id"] for r in rows] == [h.chunk_id for h in hits]


# --------------------------------------------------------------------------- #
# Shape and ordering, for every arm
# --------------------------------------------------------------------------- #
def test_every_arm_returns_ordered_retrieved_objects(store, stub_cross_encoder):
    arms = retrievers.build_all(store)
    assert [a.name for a in arms] == ["dense", "bm25", "hybrid", "rerank"]

    for arm in arms:
        hits = arm.search("attention over tokens", k=2)
        assert hits, f"{arm.name} returned nothing"
        assert all(isinstance(h, Retrieved) for h in hits)
        assert len(hits) <= 2
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True), f"{arm.name} is not ordered"


def test_every_arm_cites_real_chunks(store, stub_cross_encoder):
    """Citation fields must come from the metadata frame, not be invented."""
    known = set(store.meta["chunk_id"])
    for arm in retrievers.build_all(store):
        for hit in arm.search("citrus fruit antioxidant", k=2):
            assert hit.chunk_id in known
            assert hit.title and hit.arxiv_id


# --------------------------------------------------------------------------- #
# Fusion behaviour
# --------------------------------------------------------------------------- #
def test_hybrid_rewards_agreement_between_the_two_arms(store):
    """A chunk both arms rank highly should beat one only a single arm likes.

    This is the property RRF is chosen for; without it the hybrid arm is just a
    slower way to run one of its halves.
    """
    hybrid = retrievers.HybridRRFRetriever(store)
    fused = hybrid._fused_rows("transformer self attention tokens", depth=3)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
    # The top chunk is the one both arms surface, so it accumulated two
    # reciprocal-rank contributions rather than one.
    assert scores[0] > 1.0 / (config.RRF_K + 1)


def test_hybrid_finds_what_dense_alone_misses(store):
    """Fusion must be able to surface a lexical-only match.

    The toy embedder is bag-of-words, so this is a weak demonstration — the real
    version of this claim is measured by the benchmark, not asserted here. What
    is pinned is that the BM25 half genuinely contributes to the fused ranking.
    """
    hybrid = retrievers.HybridRRFRetriever(store)
    fused_rows = {r for r, _ in hybrid._fused_rows("magma crust", depth=3)}
    bm25_rows = {r for r, _ in hybrid._bm25.top_k("magma crust", 3)}
    assert bm25_rows and bm25_rows <= fused_rows


def test_rerank_reorders_its_shortlist(store, stub_cross_encoder):
    """The reranker's output order comes from the cross-encoder, not the fusion.

    With the stub scoring by document length, the longest candidate must come
    first — proving the rerank step actually replaces the incoming order.
    """
    hybrid = retrievers.HybridRRFRetriever(store)
    arm = retrievers.RerankRetriever(hybrid, candidates=3)
    hits = arm.search("attention tokens citrus magma", k=3)
    lengths = [len(h.text) for h in hits]
    assert lengths == sorted(lengths, reverse=True)


def test_rerank_cannot_invent_documents_outside_the_shortlist(store, stub_cross_encoder):
    """A cross-encoder reorders candidates; it never retrieves.

    Worth pinning because it is the arm's central limitation: anything the first
    stage missed is lost for good, which is why the shortlist is fused rather
    than dense-only.
    """
    hybrid = retrievers.HybridRRFRetriever(store)
    shortlist = {r for r, _ in hybrid._fused_rows("citrus", depth=2)[:2]}
    arm = retrievers.RerankRetriever(hybrid, candidates=2)
    hits = arm.search("citrus", k=5)
    returned = {
        int(store.meta.index[store.meta["chunk_id"] == h.chunk_id][0]) for h in hits
    }
    assert returned <= shortlist
