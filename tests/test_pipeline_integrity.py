"""Integrity tests that pin the project's claims to its data and results.

These guard the evaluation's validity and, once results are committed, the
headline numbers the README reports. They run keyless: the eval-set checks are
pure data validation, and the results checks are skipped gracefully until the
committed result files exist.
"""

from __future__ import annotations

import json

import pytest

from src import config, evaluation


def test_eval_set_loads_and_is_nonempty():
    questions = evaluation.load_eval_set()
    assert len(questions) >= 8
    assert {q.hop for q in questions} == {"single", "multi"}


def test_every_ground_truth_id_is_a_seed_paper():
    """The eval's correct answers must be papers we guarantee are in the corpus.

    If a relevant id were not in SEED_PAPER_IDS, retrieval could never find it and
    the metric would be meaningless. This test is what keeps the evaluation honest
    as the question set evolves.
    """
    seeds = set(config.SEED_PAPER_IDS)
    for q in evaluation.load_eval_set():
        for aid in q.relevant_ids:
            assert aid in seeds, f"{q.qid} references {aid}, not in SEED_PAPER_IDS"


def test_multi_hop_questions_need_multiple_papers():
    """A 'multi' question that only needs one paper is mislabelled."""
    for q in evaluation.load_eval_set():
        if q.hop == "multi":
            assert len(q.relevant_ids) >= 2, f"{q.qid} labelled multi but has <2 ids"


@pytest.mark.skipif(
    not (config.RESULTS_DIR / "retrieval_metrics.json").exists(),
    reason="retrieval metrics not generated yet (run run_eval.py)",
)
def test_retrieval_results_schema():
    with open(config.RESULTS_DIR / "retrieval_metrics.json", encoding="utf-8") as f:
        rows = json.load(f)
    assert rows
    for r in rows:
        assert {"qid", "hop", "hit@k", "recall@k", "mrr"} <= set(r)
        assert 0.0 <= r["mrr"] <= 1.0
        assert 0.0 <= r["recall@k"] <= 1.0


@pytest.mark.skipif(
    not (config.RESULTS_DIR / "answer_quality.json").exists(),
    reason="answer eval not generated yet (needs an API key once)",
)
def test_answer_results_cover_both_systems():
    with open(config.RESULTS_DIR / "answer_quality.json", encoding="utf-8") as f:
        rows = json.load(f)
    systems = {r["system"] for r in rows}
    assert systems == {"naive", "agent"}
    for r in rows:
        assert 0 <= r["groundedness"] <= 5
        assert 0 <= r["correctness"] <= 5
