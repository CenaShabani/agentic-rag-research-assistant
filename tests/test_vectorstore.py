"""Tests for the FAISS vector store: round-trip, ordering, dimensions."""

from __future__ import annotations

from src import config
from src.vectorstore import VectorStore


def test_build_indexes_every_chunk(toy_embeddings, small_chunks):
    store = VectorStore.build(small_chunks)
    assert store.size == len(small_chunks)


def test_search_returns_at_most_k(toy_embeddings, small_chunks):
    store = VectorStore.build(small_chunks)
    hits = store.search("attention", k=2)
    assert len(hits) <= 2


def test_search_ranks_relevant_chunk_first(toy_embeddings, small_chunks):
    """A query about transformers should return the transformer chunk first."""
    store = VectorStore.build(small_chunks)
    hits = store.search("self attention transformer tokens", k=3)
    assert hits[0].arxiv_id == "1111.0001"


def test_scores_are_descending(toy_embeddings, small_chunks):
    store = VectorStore.build(small_chunks)
    hits = store.search("vitamin citrus", k=3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_save_and_load_round_trip(toy_embeddings, small_chunks, tmp_path, monkeypatch):
    """A saved index reloads to the same size and returns the same top hit."""
    monkeypatch.setattr(config, "FAISS_INDEX_FILE", tmp_path / "faiss.index")
    monkeypatch.setattr(config, "CHUNK_META_FILE", tmp_path / "meta.parquet")

    store = VectorStore.build(small_chunks)
    top_before = store.search("volcano magma", k=1)[0].arxiv_id
    store.save()

    reloaded = VectorStore.load()
    assert reloaded.size == store.size
    top_after = reloaded.search("volcano magma", k=1)[0].arxiv_id
    assert top_before == top_after == "3333.0003"
