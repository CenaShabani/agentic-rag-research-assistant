"""The paired tests, pinned against cases whose answers are known by hand.

A significance test is the last thing in a project that should be trusted because
it looks plausible: it outputs a small number either way, and a broken one is
indistinguishable from a working one by eye. So every statistic here is checked
against a case where the right answer can be worked out on paper.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import significance


def _rows(scores: dict[str, list[float]], metric: str = "ndcg@k") -> list[dict]:
    """Benchmark-shaped rows from {arm: [per-question scores]}."""
    return [
        {"arm": arm, "qid": f"q{i}", metric: s}
        for arm, values in scores.items()
        for i, s in enumerate(values)
    ]


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #
def test_differences_are_aligned_by_qid_not_by_position():
    """Rows arrive arm-block by arm-block; nothing guarantees a shared order.

    The rows below are deliberately shuffled for one arm. Positional subtraction
    would give [0.5, -0.5, 0.0]; correct pairing gives [0.1, 0.1, 0.1].
    """
    rows = [
        {"arm": "a", "qid": "q1", "ndcg@k": 0.9},
        {"arm": "a", "qid": "q2", "ndcg@k": 0.5},
        {"arm": "a", "qid": "q3", "ndcg@k": 0.7},
        {"arm": "b", "qid": "q2", "ndcg@k": 0.4},
        {"arm": "b", "qid": "q3", "ndcg@k": 0.6},
        {"arm": "b", "qid": "q1", "ndcg@k": 0.8},
    ]
    diffs, qids = significance.paired_differences(rows, "a", "b")

    assert qids == ["q1", "q2", "q3"]
    assert diffs == pytest.approx([0.1, 0.1, 0.1])


def test_mismatched_question_sets_are_refused():
    """An unpaired comparison must fail loudly, not silently drop questions."""
    rows = _rows({"a": [1.0, 1.0, 1.0], "b": [0.5, 0.5]})

    with pytest.raises(ValueError, match="same questions"):
        significance.paired_differences(rows, "a", "b")


# --------------------------------------------------------------------------- #
# The randomization test
# --------------------------------------------------------------------------- #
def test_uniform_difference_gives_the_smallest_attainable_p():
    """Every question favouring the same arm is the extreme case.

    With all differences the same sign and magnitude, |mean| under a sign flip
    only reaches the observed value when every sign flips together — 2 of the
    2**n assignments. Sampling 10,000 of them, the observed arrangement itself is
    the +1 in the numerator, so p lands at the floor of 1/(N+1).
    """
    diffs = np.full(30, 0.2)
    p = significance.randomization_p_value(diffs, n_permutations=999, seed=1)

    assert p == pytest.approx(1 / 1000, abs=2e-3)


def test_symmetric_differences_are_not_significant():
    """Differences that cancel exactly cannot distinguish the arms."""
    diffs = np.array([0.3, -0.3, 0.2, -0.2, 0.1, -0.1])
    p = significance.randomization_p_value(diffs, n_permutations=2000, seed=1)

    assert p > 0.9


def test_p_value_can_never_be_zero():
    """A finite resample cannot support p = 0; the +1 correction guarantees it."""
    diffs = np.full(200, 1.0)
    p = significance.randomization_p_value(diffs, n_permutations=100, seed=1)

    assert p > 0.0


def test_empty_difference_vector_is_not_significant():
    assert significance.randomization_p_value(np.array([])) == 1.0


def test_p_value_is_seeded_and_reproducible():
    diffs = np.array([0.1, -0.05, 0.2, 0.02, -0.01, 0.3, 0.15])
    a = significance.randomization_p_value(diffs, n_permutations=500, seed=7)
    b = significance.randomization_p_value(diffs, n_permutations=500, seed=7)

    assert a == b


# --------------------------------------------------------------------------- #
# Holm
# --------------------------------------------------------------------------- #
def test_holm_matches_the_hand_computation():
    """Worked by hand: m = 4, sorted [0.01, 0.02, 0.03, 0.04].

    Multipliers step down 4, 3, 2, 1 -> [0.04, 0.06, 0.06, 0.04], then the
    monotonicity pass drags each value up to the running maximum, giving
    [0.04, 0.06, 0.06, 0.06]. The last entry is where naive Bonferroni-style
    stepping would report *less* than the one before it.
    """
    assert significance.holm_adjust([0.01, 0.02, 0.03, 0.04]) == pytest.approx(
        [0.04, 0.06, 0.06, 0.06]
    )


def test_holm_returns_values_in_the_input_order():
    adjusted = significance.holm_adjust([0.04, 0.01])

    # The 0.01 is the smallest, so it takes the largest multiplier (m - 0 = 2).
    assert adjusted == pytest.approx([0.04, 0.02])


def test_holm_never_exceeds_one():
    assert significance.holm_adjust([0.6, 0.7, 0.8]) == pytest.approx([1.0, 1.0, 1.0])


def test_holm_on_an_empty_family():
    assert significance.holm_adjust([]) == []


# --------------------------------------------------------------------------- #
# The assembled comparison
# --------------------------------------------------------------------------- #
def test_pairs_are_oriented_so_the_winner_is_first():
    """Every row must read "A beat B by X" — a negative mean_diff is a bug."""
    rows = _rows({"weak": [0.2] * 20, "strong": [0.8] * 20})
    records = significance.compare_arms(rows)

    assert len(records) == 1
    assert records[0]["arm_a"] == "strong"
    assert records[0]["mean_diff"] > 0


def test_correction_is_applied_across_the_whole_family():
    """Three arms is three pairs, and every one carries a corrected p-value."""
    rng = np.random.default_rng(0)
    rows = _rows({
        "a": rng.uniform(0.7, 0.9, 40).tolist(),
        "b": rng.uniform(0.4, 0.6, 40).tolist(),
        "c": rng.uniform(0.1, 0.3, 40).tolist(),
    })
    records = significance.compare_arms(rows)

    assert len(records) == 3
    for rec in records:
        assert rec["p_value_holm"] >= rec["p_value"]
        assert rec["n_wins_a"] + rec["n_wins_b"] + rec["n_ties"] == rec["n_questions"]


def test_identical_arms_are_not_declared_different():
    """The null case. An arm compared against a copy of itself must not win."""
    scores = np.random.default_rng(3).uniform(0, 1, 50).tolist()
    records = significance.compare_arms(_rows({"a": scores, "b": list(scores)}))

    assert records[0]["mean_diff"] == pytest.approx(0.0)
    assert records[0]["significant"] is False


def test_a_tiny_but_perfectly_consistent_edge_is_detected():
    """The case marginal intervals miss, and the reason to pair at all.

    Both arms swing from 0.10 to 0.98 across questions, so each one's own
    interval is enormous and the two overlap almost entirely. But arm `a` wins
    every single question by exactly 0.01. Paired, that is unambiguous.
    """
    base = np.linspace(0.10, 0.98, 60)
    records = significance.compare_arms(
        _rows({"a": (base + 0.01).tolist(), "b": base.tolist()})
    )

    assert records[0]["mean_diff"] == pytest.approx(0.01)
    assert records[0]["ci_excludes_zero"] is True
    assert records[0]["significant"] is True
