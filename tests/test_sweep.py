"""The cutoff sweep, pinned on a retriever whose ranking is known exactly.

The sweep's whole claim is that it searches *once* and truncates, so the curve
isolates the cutoff rather than mixing in a retriever reconfigured per k. That is
testable directly: count the searches, and check the metrics against a ranking
written out by hand.
"""

from __future__ import annotations

import pytest

from src import evaluation
from src.evaluation import EvalQuestion
from src.vectorstore import Retrieved


class ScriptedRetriever:
    """Returns a fixed ranking, and records every k it was asked for.

    `Retrieved` carries chunk-level fields, but the metrics only ever read
    `arxiv_id` — the rest is filled in to keep the object honest rather than
    because anything under test looks at it.
    """

    def __init__(self, name: str, ranking: list[str]):
        self.name = name
        self._ranking = ranking
        self.calls: list[int] = []

    def search(self, query: str, k: int = 10) -> list[Retrieved]:
        self.calls.append(k)
        return [
            Retrieved(chunk_id=f"{aid}:0", arxiv_id=aid, title=aid,
                      text="text", score=1.0 / (i + 1))
            for i, aid in enumerate(self._ranking[:k])
        ]


@pytest.fixture
def question() -> EvalQuestion:
    """One relevant paper, planted at rank 3 of the scripted ranking."""
    return EvalQuestion(qid="q1", question="anything", relevant_ids=["target"],
                        hop="single")


@pytest.fixture
def retriever() -> ScriptedRetriever:
    ranking = ["miss1", "miss2", "target"] + [f"filler{i}" for i in range(60)]
    return ScriptedRetriever("scripted", ranking)


def test_one_search_per_question_at_the_deepest_cutoff(retriever, question):
    """Six cutoffs, one search — the property the whole design rests on."""
    evaluation.sweep_k([retriever], [question], ks=(1, 5, 10, 50), warmup=False)

    assert retriever.calls == [50], (
        "the sweep must search once at max(ks) and truncate; searching per "
        "cutoff would let a reranker return a different ordering per k"
    )


def test_warmup_search_is_separate_from_the_measured_ones(retriever, question):
    evaluation.sweep_k([retriever], [question], ks=(1, 10), warmup=True)

    assert retriever.calls == [10, 10]  # one warm-up, one measured


def test_recall_steps_up_exactly_where_the_relevant_paper_sits(retriever, question):
    """The target is at rank 3, so recall is 0 at k<3 and 1 from k=3 onward."""
    rows = evaluation.sweep_k([retriever], [question], ks=(1, 2, 3, 10),
                              warmup=False)
    recall = {r["k"]: r["recall@k"] for r in rows}

    assert recall == {1: 0.0, 2: 0.0, 3: 1.0, 10: 1.0}


def test_ndcg_at_the_cutoff_matches_the_hand_computation(retriever, question):
    """One relevant document at rank 3: DCG = 1/log2(4), ideal DCG = 1/log2(2).

    So nDCG = 1/2 exactly, at every k that includes rank 3. Checking a literal
    value rather than a monotonicity property is the point — a discount applied
    with the wrong offset still increases and decreases in the right places.
    """
    rows = evaluation.sweep_k([retriever], [question], ks=(3, 10), warmup=False)
    ndcg = {r["k"]: r["ndcg@k"] for r in rows}

    assert ndcg[3] == pytest.approx(0.5)
    assert ndcg[10] == pytest.approx(0.5)


def test_every_arm_and_cutoff_gets_a_row_with_an_interval(question):
    arms = [ScriptedRetriever("a", ["target"]), ScriptedRetriever("b", ["miss"])]
    rows = evaluation.sweep_k(arms, [question], ks=(1, 5), warmup=False)

    assert len(rows) == 4
    for row in rows:
        assert row["ndcg@k_lo"] <= row["ndcg@k"] <= row["ndcg@k_hi"]
        assert row["recall@k_lo"] <= row["recall@k"] <= row["recall@k_hi"]


def test_truncation_cannot_help_an_arm_that_never_finds_it(question):
    """An arm that misses at k=50 must miss at every smaller cutoff too."""
    arm = ScriptedRetriever("blind", [f"miss{i}" for i in range(60)])
    rows = evaluation.sweep_k([arm], [question], warmup=False)

    assert all(r["recall@k"] == 0.0 for r in rows)
