"""Evaluation — does the agent actually beat naive RAG, and by how much?

This is the methodology track that carries the project. It has three parts, and
the first two need no API key:

1. **Retrieval metrics (fully local, no API key).** For each evaluation question
   we know which paper(s) genuinely answer it. We measure whether retrieval
   surfaced them:
     - hit@k   : did *any* relevant paper appear in the top-k?
     - recall@k: what fraction of the relevant papers appeared?
     - MRR     : 1 / rank of the first relevant paper (rewards ranking it high).
   These use only the vector store, so they reproduce exactly on any machine.

2. **The retrieval benchmark (fully local, no API key).** The four arms in
   `retrievers.py` — dense, BM25, hybrid RRF, cross-encoder rerank — scored on
   the same questions with nDCG@10, MAP, recall@k and MRR, each with a bootstrap
   confidence interval, plus per-arm latency. This is what turns "we built a RAG
   system" into "we measured four of them and here is what each one buys and
   costs". The question set it runs on is 120 generated known-item queries
   (`evalset.py`) alongside the 11 hand-written ones; `eval/ADJUDICATION.md`
   states how both sets were labelled and where they mislead.

3. **Answer quality (LLM-as-judge, key-gated + cached).** A separate Claude call
   grades each answer for *groundedness* (is every claim supported by the sources?)
   and *correctness* (does it actually answer the question?), for naive RAG and
   the agent, split by single-hop vs multi-hop. The grades are cached and
   committed, so the headline comparison reproduces from cache without a key.

The design goal is a fair head-to-head: same corpus, same retriever, same
questions — the only thing that differs is single-shot vs iterative retrieval.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import agent as agent_mod
from . import config, llm, rag
from .vectorstore import VectorStore


# --------------------------------------------------------------------------- #
# Evaluation set
# --------------------------------------------------------------------------- #
@dataclass
class EvalQuestion:
    """One graded question.

    `relevant_ids` are the arXiv ids that genuinely answer it — the ground truth
    for retrieval metrics. `hop` tags the question as needing one fact
    ("single") or several ("multi"); the whole hypothesis is that the agent's
    edge shows up on the multi-hop rows.
    """

    qid: str
    question: str
    relevant_ids: list[str]
    hop: str  # "single" or "multi"
    # Which construction produced this question. "curated" = hand-written by a
    # person (the 11 in qa_set.json); "title" / "abstract" = generated known-item
    # queries (see evalset.py). Defaulted so the hand-written file, which has no
    # such field, still loads unchanged. The benchmark slices on this, because
    # the two generated flavours differ in how much literal word overlap they
    # share with their target — which is the lexical-bias measurement.
    flavour: str = "curated"


def load_eval_set() -> list[EvalQuestion]:
    """Load the curated question set from eval/qa_set.json."""
    with open(config.EVAL_DIR / "qa_set.json", encoding="utf-8") as f:
        raw = json.load(f)
    return [EvalQuestion(**q) for q in raw]


def load_known_item_questions() -> list[EvalQuestion]:
    """Load the 120 generated known-item queries as `EvalQuestion`s."""
    from .evalset import load_known_item_set

    return [EvalQuestion(**q) for q in load_known_item_set()]


# --------------------------------------------------------------------------- #
# Retrieval metrics (local)
# --------------------------------------------------------------------------- #
def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1.0 if any relevant id is in the retrieved set, else 0.0."""
    return 1.0 if set(retrieved_ids) & set(relevant_ids) else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of the relevant ids that were retrieved."""
    if not relevant_ids:
        return 0.0
    found = set(retrieved_ids) & set(relevant_ids)
    return len(found) / len(set(relevant_ids))


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant id (0 if none retrieved).

    Ranks are 1-based, so a relevant paper in the top slot scores 1.0, second
    slot 0.5, and so on — rewarding retrieval for putting the right paper high,
    not merely somewhere in the list.
    """
    relevant = set(relevant_ids)
    for rank, aid in enumerate(retrieved_ids, start=1):
        if aid in relevant:
            return 1.0 / rank
    return 0.0


def retrieval_ids(store: VectorStore, question: str, k: int = config.TOP_K) -> list[str]:
    """The ordered list of distinct arXiv ids retrieved for a question.

    Chunks are de-duplicated to paper level (keeping first appearance) because
    the ground truth is per-paper: two chunks from the same paper are one hit.
    """
    ids: list[str] = []
    for h in store.search(question, k=k):
        if h.arxiv_id not in ids:
            ids.append(h.arxiv_id)
    return ids


def evaluate_retrieval(store: VectorStore, questions: list[EvalQuestion],
                       k: int = config.TOP_K) -> list[dict]:
    """Compute hit@k, recall@k, and MRR for every question. Fully local."""
    rows = []
    for q in questions:
        got = retrieval_ids(store, q.question, k=k)
        rows.append(
            {
                "qid": q.qid,
                "hop": q.hop,
                "hit@k": hit_at_k(got, q.relevant_ids),
                "recall@k": recall_at_k(got, q.relevant_ids),
                "mrr": mrr(got, q.relevant_ids),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Ranking metrics — nDCG and MAP
# --------------------------------------------------------------------------- #
# hit@k, recall@k and MRR above answer "did we find it?" and "was the first one
# early?". Neither notices where the *second* relevant document landed, and MRR
# stops looking after the first hit. The two metrics below grade the whole
# ordering, which is what a retrieval team actually reports.
def dcg(gains: list[float]) -> float:
    """Discounted cumulative gain of a ranked list of relevance gains.

    Each position's gain is divided by log2(rank + 1), so a relevant document at
    rank 1 is worth 1.00, at rank 2 about 0.63, at rank 10 about 0.29. The
    logarithm encodes an assumption worth stating: attention falls off sharply
    with position but never quite to zero — rank 10 is much worse than rank 1,
    not worthless.
    """
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Normalised DCG at k, with binary gains (relevant = 1, else 0).

    Raw DCG is not comparable across questions: a question with three relevant
    papers can score higher than one with a single relevant paper purely because
    there is more to find. Normalising by the *ideal* DCG — the score the perfect
    ranking would have achieved for this question — puts every question on a 0-1
    scale where 1.0 means "nothing could have been ordered better".

    Gains are binary here because the ground truth is binary; see
    `eval/ADJUDICATION.md` for why, and for why that makes these numbers
    incomparable to graded-relevance nDCG from public leaderboards.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    gains = [1.0 if aid in relevant else 0.0 for aid in retrieved_ids[:k]]
    # The ideal ranking puts every relevant document first — but only as many as
    # could fit in k, and only as many as exist.
    ideal = [1.0] * min(len(relevant), k)
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg else 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: list[str],
                      k: int) -> float:
    """Average precision at k — the mean of precision measured at each hit.

    Averaged over questions this is MAP, the standard single-number summary in
    the IR literature. It differs from nDCG in what it rewards: AP cares about
    precision *at every relevant document*, so burying the second relevant paper
    at rank 9 hurts it more than nDCG's smooth logarithmic discount does. Both
    are reported because they disagree in an informative way — a large nDCG with
    a weak MAP means the arm reliably finds *one* good document and then loses
    interest.

    Divided by the number of relevant documents (capped at k), not by the number
    found, so failing to retrieve a relevant document is penalised rather than
    ignored.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, aid in enumerate(retrieved_ids[:k], start=1):
        if aid in relevant:
            hits += 1
            precision_sum += hits / rank
    denom = min(len(relevant), k)
    return precision_sum / denom if denom else 0.0


# --------------------------------------------------------------------------- #
# Uncertainty — the bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: list[float], n_resamples: int = config.BOOTSTRAP_RESAMPLES,
                 confidence: float = config.BOOTSTRAP_CI,
                 seed: int = config.SEED) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`.

    **Why this is not optional.** A benchmark that reports "arm A scores 0.71,
    arm B scores 0.68" and stops has said nothing: over 120 questions that gap may
    be well inside the noise. The bootstrap answers "if I had drawn a different
    sample of questions, how much would this mean have moved?"

    **What is resampled.** Questions, with replacement — because the question set
    is the sample we are generalising over. The claim being defended is "on
    queries like these", not "on this exact list of 120", and resampling
    questions is what makes the interval mean the former.

    Seeded, so the published interval is reproducible to the digit.
    """
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    # One (n_resamples, n) draw of indices, then a row-wise mean: the whole
    # bootstrap is two array operations, no Python loop.
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return (float(lo), float(hi))


# --------------------------------------------------------------------------- #
# The arm benchmark
# --------------------------------------------------------------------------- #
def _retrieved_paper_ids(retriever, question: str, k: int) -> list[str]:
    """Ordered distinct arXiv ids from one arm's hits.

    The same chunk-to-paper de-duplication `retrieval_ids` applies to the dense
    arm, applied to any arm. Ground truth is per-paper, so two chunks from one
    paper are one hit — and an arm that returns ten chunks of the same paper must
    not be scored as if it found ten relevant documents.
    """
    ids: list[str] = []
    for h in retriever.search(question, k=k):
        if h.arxiv_id not in ids:
            ids.append(h.arxiv_id)
    return ids


def evaluate_arms(retrievers: list, questions: list[EvalQuestion],
                  k: int = config.EVAL_K, warmup: bool = True) -> list[dict]:
    """Score every arm on every question. One row per (arm, question).

    **Latency.** Measured with `perf_counter` around the search call only, after
    a warm-up query per arm. The warm-up matters more than it sounds: the first
    call to the dense arm pays a one-off sentence-transformer load of tens of
    seconds, and the first call to the reranker pays another. Including those in
    the average would not make the reranker look slow — it would make the
    *measurement* wrong, and the whole quality-per-millisecond claim rests on
    this number being honest.

    Latency is still wall-clock on one laptop CPU, single-threaded, batch size
    one. It is a fair *relative* comparison between arms on identical hardware,
    not a throughput figure anyone should plan capacity from.
    """
    if warmup:
        for r in retrievers:
            r.search("warm up the model weights and any lazy imports", k=k)

    rows: list[dict] = []
    for r in retrievers:
        for q in questions:
            start = time.perf_counter()
            got = _retrieved_paper_ids(r, q.question, k=k)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rows.append(
                {
                    "arm": r.name,
                    "qid": q.qid,
                    "hop": q.hop,
                    "flavour": q.flavour,
                    "ndcg@k": ndcg_at_k(got, q.relevant_ids, k),
                    "ap@k": average_precision(got, q.relevant_ids, k),
                    "recall@k": recall_at_k(got, q.relevant_ids),
                    "mrr": mrr(got, q.relevant_ids),
                    "hit@k": hit_at_k(got, q.relevant_ids),
                    "latency_ms": elapsed_ms,
                }
            )
    return rows


#: The metrics summarised with a bootstrap interval. Latency is summarised too,
#: but as a median rather than a mean — a single scheduler hiccup can double a
#: mean and say nothing about the arm.
ARM_METRICS = ("ndcg@k", "ap@k", "recall@k", "mrr", "hit@k")


# --------------------------------------------------------------------------- #
# The cutoff sweep — is k = 10 (or k = 5) the right place to look?
# --------------------------------------------------------------------------- #
#: The cutoffs swept. 5 is what the system serves (`config.TOP_K`), 10 is what the
#: benchmark reports (`config.EVAL_K`), and 50 is the reranker's shortlist depth
#: (`config.RERANK_CANDIDATES`) — every constant the project already committed to,
#: plus 1 and 3 to see the top of the curve. Sweeping the numbers you already
#: chose is the cheapest way to find out whether they were choices or habits.
SWEEP_KS = (1, 3, 5, 10, 20, 50)


def sweep_k(retrievers: list, questions: list[EvalQuestion],
            ks: tuple[int, ...] = SWEEP_KS, warmup: bool = True) -> list[dict]:
    """nDCG@k and recall@k for every arm at every cutoff. One row per (arm, k).

    **One search, many cutoffs.** Each arm searches once at `max(ks)` and the
    ranked list is then truncated for each smaller k. Searching separately per
    cutoff would be six times the work *and* less correct: a retriever asked for
    50 and a retriever asked for 5 can return different orderings (the reranker
    demonstrably does — it rescores a shortlist), so the curve would mix "quality
    at cutoff k" with "quality of a retriever configured for k". Truncating one
    ranking isolates the cutoff, which is what the sweep is asking about.

    **What it answers.** Two live questions the fixed-k benchmark cannot:
    where recall saturates — i.e. whether serving `TOP_K = 5` chunks discards
    documents the evaluation at 10 was crediting — and how deep a first stage has
    to go before a reranker has everything it needs, which is the assumption
    `RERANK_CANDIDATES = 50` rests on.
    """
    max_k = max(ks)
    if warmup:
        for r in retrievers:
            r.search("warm up the model weights and any lazy imports", k=max_k)

    rows: list[dict] = []
    for r in retrievers:
        per_k: dict[int, list[tuple[float, float]]] = {k: [] for k in ks}
        for q in questions:
            ranked = _retrieved_paper_ids(r, q.question, max_k)
            for k in ks:
                truncated = ranked[:k]
                per_k[k].append((
                    ndcg_at_k(truncated, q.relevant_ids, k),
                    recall_at_k(truncated, q.relevant_ids),
                ))

        for k, pairs in per_k.items():
            ndcgs = [p[0] for p in pairs]
            recalls = [p[1] for p in pairs]
            lo, hi = bootstrap_ci(ndcgs)
            r_lo, r_hi = bootstrap_ci(recalls)
            rows.append({
                "arm": r.name,
                "k": k,
                "n_questions": len(pairs),
                "ndcg@k": float(np.mean(ndcgs)),
                "ndcg@k_lo": lo,
                "ndcg@k_hi": hi,
                "recall@k": float(np.mean(recalls)),
                "recall@k_lo": r_lo,
                "recall@k_hi": r_hi,
            })
    return rows


def summarise_arms(rows: list[dict], k: int = config.EVAL_K) -> list[dict]:
    """Per-arm, per-slice means with bootstrap intervals and median latency.

    Produces one record per (arm, slice), where slice is "all" plus each question
    flavour present. The flavour split is the point: comparing an arm's `title`
    row against its `abstract` row measures how much it leans on literal word
    overlap (see `eval/ADJUDICATION.md`).
    """
    df = pd.DataFrame(rows)
    out: list[dict] = []
    slices = ["all"] + sorted(df["flavour"].unique().tolist())

    for arm in df["arm"].unique():
        for slice_name in slices:
            sub = df[df.arm == arm]
            if slice_name != "all":
                sub = sub[sub.flavour == slice_name]
            if sub.empty:
                continue
            record = {"arm": arm, "slice": slice_name, "n_questions": int(len(sub)),
                      "k": k}
            for metric in ARM_METRICS:
                values = sub[metric].tolist()
                lo, hi = bootstrap_ci(values)
                record[metric] = float(np.mean(values))
                record[f"{metric}_lo"] = lo
                record[f"{metric}_hi"] = hi
            record["latency_ms_median"] = float(sub["latency_ms"].median())
            record["latency_ms_p95"] = float(sub["latency_ms"].quantile(0.95))
            out.append(record)
    return out


# --------------------------------------------------------------------------- #
# Answer quality (LLM-as-judge)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = (
    "You are a strict evaluator of answers to research questions. You are given a "
    "question, a set of source passages, and an answer. Grade two things on a 1-5 "
    "scale:\n"
    "  groundedness: 5 = every claim is directly supported by the passages; "
    "1 = the answer makes claims the passages do not support.\n"
    "  correctness: 5 = fully and accurately answers the question; "
    "1 = wrong or off-topic.\n"
    "Reply with ONLY a JSON object: "
    '{\"groundedness\": <int>, \"correctness\": <int>}.'
)


def judge_answer(question: str, sources_text: str, answer_text: str) -> dict:
    """Grade one answer with the LLM judge; returns groundedness + correctness.

    Cached and key-gated like every other LLM call, so the graded results
    reproduce from the committed cache without an API key.
    """
    user = (
        f"Question: {question}\n\n"
        f"Source passages:\n{sources_text}\n\n"
        f"Answer:\n{answer_text}\n\n"
        "Grade the answer. Reply with only the JSON object."
    )
    result = llm.call(
        messages=[{"role": "user", "content": user}],
        system=JUDGE_SYSTEM,
        model=config.JUDGE_MODEL,
    )
    text = llm.first_text(result)
    try:
        grades = json.loads(text)
        return {
            "groundedness": int(grades["groundedness"]),
            "correctness": int(grades["correctness"]),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        # A judge that returns unparseable output scores 0 — visible and honest,
        # rather than silently dropping the row.
        return {"groundedness": 0, "correctness": 0}


def evaluate_answers(store: VectorStore, questions: list[EvalQuestion]) -> list[dict]:
    """Run naive RAG and the agent on every question and grade both answers.

    Produces one row per (question, system) so the notebook can compare naive vs
    agent, sliced by hop type. Also records the agent's step count — the token /
    latency cost of its extra retrievals, which is the other side of the trade.
    """
    rows = []
    for q in questions:
        naive = rag.answer(store, q.question)
        agentic = agent_mod.answer(store, q.question)

        naive_sources = rag.format_context(naive.sources)
        agent_sources = rag.format_context(agentic.sources)

        rows.append(
            {
                "qid": q.qid, "hop": q.hop, "system": "naive",
                "n_retrievals": 1,
                "n_cited": len(naive.cited_ids),
                **judge_answer(q.question, naive_sources, naive.answer),
            }
        )
        rows.append(
            {
                "qid": q.qid, "hop": q.hop, "system": "agent",
                "n_retrievals": len(agentic.steps),
                "n_cited": len(agentic.cited_ids),
                **judge_answer(q.question, agent_sources, agentic.answer),
            }
        )
    return rows
