"""Tests for the generated known-item evaluation set.

The set is the ground truth every benchmark number is measured against. If it is
not deterministic, published numbers cannot be reproduced; if its labels point at
papers the corpus does not contain, every arm is scored against an impossible
task. Both are checked here.
"""

from __future__ import annotations

from src import config, evalset
from src.acquisition import Paper


def _paper(i: int, title: str, abstract: str) -> Paper:
    return Paper(
        arxiv_id=f"9999.{i:04d}",
        title=title,
        abstract=abstract,
        authors=["A. Author"],
        categories=["cs.CL"],
        published="2026-01-01T00:00:00Z",
        license=None,
    )


def _corpus(n: int = 12) -> list[Paper]:
    return [
        _paper(
            i,
            f"A Study of Retrieval Augmented Generation Number {i} in Long Documents",
            f"We investigate retrieval method number {i} across several corpora. "
            f"The proposed approach improves recall by {i} points over the baseline "
            f"on a held-out evaluation set. We release our code.",
        )
        for i in range(n)
    ]


def test_one_query_per_style_per_sampled_paper():
    """Every paper contributes all three styles, so the slices stay paired.

    If a paper could contribute to one style and not another, comparing an arm
    across styles would compare two different samples of papers, and the slope
    in figure 06 would be uninterpretable.
    """
    rows = evalset.build_known_item_set(_corpus(10))
    assert len(rows) == 30
    assert {r["flavour"] for r in rows} == {"title", "abstract", "keyword"}
    by_paper: dict[str, set[str]] = {}
    for row in rows:
        by_paper.setdefault(row["relevant_ids"][0], set()).add(row["flavour"])
    assert all(styles == {"title", "abstract", "keyword"} for styles in by_paper.values())


def test_generation_is_deterministic_under_the_seed():
    """A published number is only reproducible if the question set is."""
    corpus = _corpus()
    assert evalset.build_known_item_set(corpus) == evalset.build_known_item_set(corpus)


def test_input_order_does_not_change_the_sample():
    """The corpus is sorted before sampling, so JSONL write order cannot leak in."""
    corpus = _corpus()
    assert (evalset.build_known_item_set(corpus)
            == evalset.build_known_item_set(list(reversed(corpus))))


def test_every_label_points_at_exactly_one_real_paper():
    corpus = _corpus()
    ids = {p.arxiv_id for p in corpus}
    for row in evalset.build_known_item_set(corpus):
        assert len(row["relevant_ids"]) == 1
        assert row["relevant_ids"][0] in ids


def test_the_title_query_is_the_title():
    corpus = _corpus()
    by_id = {p.arxiv_id: p for p in corpus}
    for row in evalset.build_known_item_set(corpus):
        if row["flavour"] == "title":
            assert row["question"] == by_id[row["relevant_ids"][0]].title


def test_boilerplate_sentences_are_never_used_as_queries():
    """"We release our code" identifies no paper, so it must not become a query."""
    for row in evalset.build_known_item_set(_corpus()):
        assert not row["question"].lower().startswith(("we release", "in this paper"))


def test_short_sentences_are_rejected():
    assert evalset._sentences("Too short. Also brief.") == []


def test_sentences_are_returned_longest_first():
    text = ("This is a reasonably long opening sentence about retrieval systems. "
            "This one is considerably longer and describes the proposed method in "
            "much more detail than the first.")
    sentences = evalset._sentences(text)
    assert len(sentences[0]) >= len(sentences[1])


def test_papers_without_a_usable_abstract_are_skipped():
    """Both flavours must come from the same papers, or the slices aren't paired."""
    usable = _corpus(4)
    unusable = [_paper(99, "A Perfectly Reasonable Title About Retrieval Systems", "Short.")]
    rows = evalset.build_known_item_set(usable + unusable)
    assert "9999.0099" not in {r["relevant_ids"][0] for r in rows}


def test_sample_size_is_bounded_by_the_corpus():
    rows = evalset.build_known_item_set(_corpus(3))
    assert len(rows) == 9


def test_keyword_queries_drop_grammar_and_stopwords():
    """The keyword style must not read like a sentence — that is its whole job."""
    query = evalset._keyword_query(
        "We propose a novel retrieval method for scientific abstracts. "
        "The approach improves recall substantially over strong baselines.",
        seed=1,
    )
    words = query.split()
    assert len(words) == config.KEYWORD_QUERY_WORDS
    assert not (set(words) & evalset._STOPWORDS)
    assert query == query.lower()


def test_keyword_queries_are_deterministic_per_paper():
    abstract = "A study of dense retrieval over scientific abstracts and long documents."
    assert evalset._keyword_query(abstract, 7) == evalset._keyword_query(abstract, 7)


def test_keyword_query_is_not_a_substring_of_the_abstract():
    """Sampling across the whole abstract stops it being a verbatim-match exercise.

    An abstract-sentence query is a literal substring of its target passage,
    which is why every arm solves that slice. The keyword style must not be.
    """
    abstract = ("We propose a scalable retrieval architecture for scientific text. "
                "Experiments across multiple corpora show consistent gains in recall "
                "and precision over established lexical baselines.")
    assert evalset._keyword_query(abstract, seed=3) not in abstract.lower()


def test_qids_are_unique():
    rows = evalset.build_known_item_set(_corpus())
    assert len({r["qid"] for r in rows}) == len(rows)


def test_the_committed_set_matches_the_real_corpus():
    """Every label in the committed set must exist in the indexed corpus.

    This is the test that catches the set and the index drifting apart — a stale
    question file would score every arm against papers that are not there.
    """
    import pandas as pd

    if not (config.KNOWN_ITEM_FILE.exists() and config.CHUNK_META_FILE.exists()):
        import pytest
        pytest.skip("no committed eval set or index available")

    indexed = set(pd.read_parquet(config.CHUNK_META_FILE)["arxiv_id"])
    for row in evalset.load_known_item_set():
        assert row["relevant_ids"][0] in indexed
