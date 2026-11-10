"""Tests for the hand-written BM25 baseline.

BM25 is implemented rather than imported (see src/lexical.py for why), so its
behaviour has to be pinned here: an unchecked baseline that is quietly broken
would make the dense arm look good for the wrong reason.
"""

from __future__ import annotations

from src.lexical import BM25

DOCS = [
    "the transformer architecture uses self attention over tokens",
    "vitamin c is an antioxidant found in citrus fruit",
    "a volcano erupts when magma rises through the crust",
    "attention is a weighted sum over a sequence of tokens",
]


def test_ranks_a_matching_document_first():
    bm25 = BM25(DOCS)
    top = bm25.top_k("citrus vitamin", k=2)
    assert top[0][0] == 1


def test_non_matching_documents_are_dropped_not_padded():
    """Asking for 4 hits when only 2 documents share a term returns 2.

    Returning arbitrary zero-scoring documents as filler would inflate every
    recall number the benchmark reports.
    """
    bm25 = BM25(DOCS)
    hits = bm25.top_k("attention tokens", k=4)
    assert len(hits) == 2
    assert {i for i, _ in hits} == {0, 3}


def test_unknown_query_terms_score_nothing_and_do_not_crash():
    bm25 = BM25(DOCS)
    assert bm25.top_k("xylophone bassoon", k=3) == []


def test_scores_are_returned_for_every_document():
    bm25 = BM25(DOCS)
    scores = bm25.scores("attention")
    assert scores.shape == (len(DOCS),)
    assert scores[0] > 0 and scores[1] == 0


def test_idf_prefers_the_rare_term():
    """A term in one document outranks a term in most of them.

    'the' appears in three of the four documents and 'volcano' in one, so the
    volcano document must win — this is the whole reason IDF exists, and getting
    the sign wrong would silently invert the baseline.
    """
    bm25 = BM25(DOCS)
    assert bm25.top_k("volcano", k=1)[0][0] == 2
    assert bm25.scores("volcano")[2] > bm25.scores("the")[2]


def test_results_are_ordered_best_first():
    bm25 = BM25(DOCS)
    scores = [s for _, s in bm25.top_k("attention tokens sequence", k=4)]
    assert scores == sorted(scores, reverse=True)


def test_size_reports_the_collection():
    assert BM25(DOCS).size == len(DOCS)
