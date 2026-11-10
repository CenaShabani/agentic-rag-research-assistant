"""Tests for the retrieval metrics — the local, key-free half of evaluation."""

from __future__ import annotations

from src import evaluation as ev
from src.vectorstore import VectorStore


def test_hit_at_k():
    assert ev.hit_at_k(["a", "b"], ["b"]) == 1.0
    assert ev.hit_at_k(["a", "b"], ["c"]) == 0.0


def test_recall_at_k_partial():
    """Retrieving one of two relevant papers is recall 0.5."""
    assert ev.recall_at_k(["a", "x"], ["a", "b"]) == 0.5


def test_recall_at_k_full():
    assert ev.recall_at_k(["a", "b", "x"], ["a", "b"]) == 1.0


def test_mrr_rewards_high_rank():
    """Relevant paper in slot 1 scores 1.0; in slot 3 scores 1/3."""
    assert ev.mrr(["a", "b", "c"], ["a"]) == 1.0
    assert ev.mrr(["x", "y", "c"], ["c"]) == 1.0 / 3


def test_mrr_zero_when_absent():
    assert ev.mrr(["x", "y"], ["z"]) == 0.0


def test_retrieval_ids_are_deduped_to_paper_level(toy_embeddings):
    """Two chunks from one paper count as one retrieved id, order preserved."""
    import pandas as pd

    chunks = pd.DataFrame(
        {
            "chunk_id": ["p:0", "p:1", "q:0"],
            "arxiv_id": ["1111.0001", "1111.0001", "2222.0002"],
            "title": ["P", "P", "Q"],
            "text": ["attention attention attention", "attention tokens", "citrus"],
        }
    )
    store = VectorStore.build(chunks)
    ids = ev.retrieval_ids(store, "attention tokens", k=3)
    assert ids.count("1111.0001") == 1  # deduped
