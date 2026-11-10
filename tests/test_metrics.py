"""Tests for the ranking metrics — nDCG, MAP, and the bootstrap.

These are the numbers the benchmark's headline claim rests on, so they are
checked against values computed by hand rather than against whatever the
implementation happened to produce.
"""

from __future__ import annotations

import math

import pytest

from src import evaluation as ev


# --------------------------------------------------------------------------- #
# nDCG
# --------------------------------------------------------------------------- #
def test_ndcg_perfect_ranking_is_one():
    """The one relevant paper in slot 1 is the ideal ranking."""
    assert ev.ndcg_at_k(["a", "b", "c"], ["a"], 10) == 1.0


def test_ndcg_discounts_by_position():
    """A relevant paper in slot 2 is worth 1/log2(3) of a perfect ranking."""
    assert ev.ndcg_at_k(["x", "a"], ["a"], 10) == pytest.approx(1 / math.log2(3))


def test_ndcg_two_relevant_at_ranks_one_and_three():
    """DCG = 1 + 1/log2(4); ideal = 1 + 1/log2(3)."""
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ev.ndcg_at_k(["a", "x", "b"], ["a", "b"], 10) == expected


def test_ndcg_is_zero_when_nothing_relevant_retrieved():
    assert ev.ndcg_at_k(["x", "y"], ["a"], 10) == 0.0


def test_ndcg_respects_the_cutoff():
    """A relevant paper at rank 3 is invisible to nDCG@2 — that is the point of k."""
    assert ev.ndcg_at_k(["x", "y", "a"], ["a"], 2) == 0.0
    assert ev.ndcg_at_k(["x", "y", "a"], ["a"], 3) > 0.0


def test_ndcg_ideal_is_capped_at_k():
    """With 5 relevant papers but k=2, retrieving 2 of them is still a perfect @2.

    Without capping the ideal DCG at k, an arm could never score 1.0 on a
    question with more relevant papers than the cutoff allows — which would make
    scores incomparable across questions.
    """
    assert ev.ndcg_at_k(["a", "b"], ["a", "b", "c", "d", "e"], 2) == 1.0


# --------------------------------------------------------------------------- #
# Average precision / MAP
# --------------------------------------------------------------------------- #
def test_average_precision_hand_computed():
    """Hits at ranks 1 and 3: (1/1 + 2/3) / 2."""
    assert ev.average_precision(["a", "x", "b"], ["a", "b"], 10) == (1 + 2 / 3) / 2


def test_average_precision_perfect_and_empty():
    assert ev.average_precision(["a", "b"], ["a", "b"], 10) == 1.0
    assert ev.average_precision(["x", "y"], ["a"], 10) == 0.0


def test_average_precision_penalises_a_missed_relevant_paper():
    """Finding one of two relevant papers cannot score 1.0.

    The divisor is the number of relevant papers, not the number found — so a
    miss costs, rather than being quietly excluded from the average.
    """
    assert ev.average_precision(["a", "x", "y"], ["a", "b"], 10) == 0.5


def test_ap_and_ndcg_disagree_informatively():
    """AP punishes a late second hit harder than nDCG's log discount does.

    Both metrics are published because this disagreement is the signal: a strong
    nDCG with a weak MAP means the arm finds one good document and then drifts.
    """
    ranking = ["a", "x", "y", "z", "b"]
    relevant = ["a", "b"]
    assert ev.average_precision(ranking, relevant, 10) < ev.ndcg_at_k(ranking, relevant, 10)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_is_reproducible_under_the_seed():
    """A published interval has to be re-derivable to the digit."""
    values = [0.0, 1.0] * 30
    assert ev.bootstrap_ci(values) == ev.bootstrap_ci(values)


def test_bootstrap_ci_brackets_the_mean():
    values = [0.2, 0.4, 0.6, 0.8] * 10
    lo, hi = ev.bootstrap_ci(values)
    assert lo < sum(values) / len(values) < hi


def test_bootstrap_ci_collapses_with_no_variance():
    """Every question scoring 1.0 leaves nothing to be uncertain about."""
    assert ev.bootstrap_ci([1.0] * 20) == (1.0, 1.0)


def test_bootstrap_ci_narrows_as_the_sample_grows():
    """More questions, tighter interval — the reason the set was grown to 120."""
    small = ev.bootstrap_ci([0.0, 1.0] * 6)
    large = ev.bootstrap_ci([0.0, 1.0] * 60)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_handles_empty_input():
    assert ev.bootstrap_ci([]) == (0.0, 0.0)
