# `src/evaluation.py` — the module that carries the project

## What it's for

Answers the question the whole repo exists to answer: does the agent beat naive RAG, and by how
much? Everything else in `src/` is machinery; **this is where the finding comes from.**

It has three deliberately separated parts, and the first two need no API key:

1. **Retrieval metrics** — hit@k, recall@k, MRR. Fully local, reproduce on any machine.
2. **The arm benchmark** — nDCG@10, MAP, bootstrap intervals and latency across the four
   retrievers in `retrievers.py`. Also fully local.
3. **Answer quality** — LLM-as-judge, key-gated, cached and committed.

That split is the single most important design decision in the file. It is why anyone can clone
the repo and reproduce both headline findings offline — the 0.83 → 0.60 multi-hop collapse *and*
the four-arm comparison — with the more expensive answer-quality half layered on top for those
with a key.

## The public surface

| Function | In | Out |
|---|---|---|
| `load_eval_set()` | — | `list[EvalQuestion]` from `eval/qa_set.json` (11 hand-written) |
| `load_known_item_questions()` | — | `list[EvalQuestion]` from the 180 generated queries |
| `hit_at_k(retrieved, relevant)` | two `list[str]` | `float` — 1.0 if any relevant id retrieved |
| `recall_at_k(retrieved, relevant)` | two `list[str]` | `float` — fraction of relevant ids found |
| `mrr(retrieved, relevant)` | two `list[str]` | `float` — 1/rank of first relevant hit |
| `dcg(gains)` / `ndcg_at_k(retrieved, relevant, k)` | | `float` — grades the whole ordering, 0–1 |
| `average_precision(retrieved, relevant, k)` | | `float` — precision at every hit; MAP when averaged |
| `bootstrap_ci(values, ...)` | `list[float]` | `(lo, hi)` — seeded 95% percentile interval |
| `retrieval_ids(store, question, k)` | | ordered distinct arXiv ids |
| `evaluate_retrieval(store, questions, k)` | | `list[dict]`, one row per question |
| `evaluate_arms(retrievers, questions, k)` | | `list[dict]`, one row per (arm, question) + latency |
| `summarise_arms(rows, k)` | | per-(arm, slice) means, intervals, median/p95 latency |
| `sweep_k(retrievers, questions, ks)` | | `list[dict]`, one row per (arm, cutoff) with intervals |
| `judge_answer(...)` / `evaluate_answers(...)` | | LLM-graded rows |

**`summarise_arms` gives each arm its own interval, and that is not enough to compare two of
them.** Two marginal intervals overlapping does not mean the difference is insignificant, and
every arm here answered the identical question list — so the paired difference is available and
is a far tighter measurement. That comparison lives in `significance.py`; this module
deliberately stops at describing each arm on its own.

`EvalQuestion` = `qid`, `question`, `relevant_ids` (ground truth), `hop` ∈ `{"single", "multi"}`,
and `flavour` (defaulted, so the hand-written file loads unchanged).

**`hop` is the field the original hypothesis rests on.** Without it the finding averages away
entirely: overall recall across all 11 questions looks unremarkable, and only splitting by hop
type exposes the 0.83 vs 0.60 collapse.

**`flavour` is the field the benchmark's second finding rests on.** It tags which query style
produced a question — `curated`, `title`, `abstract` or `keyword` — and slicing on it is what
exposes the dense arm collapsing from 0.969 on prose to 0.291 on keyword bags while BM25 barely
moves. Same lesson as `hop`, one layer up: **an average with no slice hides the finding.**

## The walkthrough

### The three metrics, and why all three

They disagree on purpose, and the disagreement is informative:

- **`hit@k`** — did *any* relevant paper show up? A blunt did-it-work-at-all measure.
- **`recall@k`** — what *fraction* of the relevant papers showed up? This is the one that catches
  the multi-hop failure, because a compound question has two relevant papers and finding one
  scores hit@k = 1.0 but recall = 0.5.
- **`mrr`** — 1/rank of the first relevant paper, 1-based. Rewards ranking the right paper *high*,
  not merely including it.

The multi-hop row in the README shows exactly why you need more than one: hit@5 = 0.80 looks
healthy — four of five compound questions surfaced *something* relevant — while recall@5 = 0.60
tells the true story, because on two of them the second required paper never appeared.
**Reporting only hit@k would have hidden the finding.** That is the lesson of this module.

`mrr` returns on the first match and `recall_at_k` guards `if not relevant_ids: return 0.0` —
returning 0 rather than dividing by zero for a question with no ground truth.

### The de-duplication in `retrieval_ids`

```python
for h in store.search(question, k=k):
    if h.arxiv_id not in ids:
        ids.append(h.arxiv_id)
```

Ground truth is **per paper**, but retrieval returns **chunks**. Two chunks from the same paper
are one hit, not two. Without this collapse, recall would be computed against a list containing
duplicates and would be wrong in a direction that flatters the system.

Order is preserved (first appearance wins) because `mrr` depends on rank — using a `set` here
would destroy the ordering the metric needs. This is why it is a list with a membership check
rather than the obvious `set` or `dict.fromkeys` one-liner: the intent is order-preserving
de-duplication, and it is worth the two extra lines to make that legible.

### The judge

`judge_answer` grades groundedness (is every claim supported by the sources?) and correctness
(does it answer the question?) — separated because they fail independently: a perfectly grounded
answer can miss the point, and a correct answer can be ungrounded luck from parametric memory.

The honest detail:

```python
# A judge that returns unparseable output scores 0 — visible and honest,
# rather than silently dropping the row.
```

Dropping unparseable rows would quietly bias the reported average upward, since failures are not
random. Scoring 0 keeps the failure visible in the denominator.

### `sweep_k` — checking the cutoffs the project already committed to

`TOP_K = 5` is what the system feeds the LLM, `EVAL_K = 10` is what the benchmark reports, and
`RERANK_CANDIDATES = 50` is how deep the reranker's shortlist goes. All three were argued in
`config.py` and none was measured. The sweep scores every arm at k ∈ {1, 3, 5, 10, 20, 50} so
those three numbers stop being habits.

**One search, many cutoffs.** Each arm searches once at `max(ks)` and the ranked list is
truncated for each smaller k. This is not only six times cheaper — it is more correct. A
retriever asked for 50 and the same retriever asked for 5 can return *different orderings*; the
reranker demonstrably does, because it rescores a shortlist whose size is the parameter. Search
per cutoff and the curve would confound "quality at cutoff k" with "quality of a retriever
configured for k". Truncating one ranking isolates the cutoff, which is the thing being asked
about. `tests/test_sweep.py` asserts the call count directly, because this property is invisible
in the output.

The two curves answer different questions and are plotted separately for that reason: nDCG@k can
*fall* as k grows (more room for irrelevant documents to sit in the ordering), while recall@k
only climbs and shows where an arm has found everything it is ever going to find.

## The non-obvious bits

- **The comparison holds everything constant except strategy** — same corpus, retriever, judge,
  questions, and `k`. This module is where that discipline is enforced.
- **Grades are cached and committed**, so the headline comparison reproduces from cache with no
  key. The cache is a published artifact, not an optimisation.
- **The judge is a relative comparator, not an absolute score.** Fair for naive-vs-agent because
  it is the same judge and rubric on both sides. The README states this rather than hiding it.
- **11 questions is small and the README says so.** Enough for a decisive effect, not enough for
  tight intervals — which is exactly what the bootstrap made undeniable. BM25's nDCG interval on
  the curated set is [0.611, 0.962], overlapping every other arm. That is the number that
  justified generating 180 more questions (`evalset.py`).
- **The bootstrap resamples *questions*, not documents.** The question set is the sample we
  generalise over: the claim being defended is "on queries like these", not "on this exact
  list". Resampling questions is what makes the interval mean the former.
- **Latency is timed after a per-arm warm-up.** The first dense call pays a one-off model load
  of tens of seconds. Including it would not make an arm look slow — it would make the
  measurement wrong, and the whole quality-per-millisecond claim rests on this being honest.
- **`evaluate_arms` de-duplicates chunks to paper level** via `_retrieved_paper_ids`, the same
  rule `retrieval_ids` applies. Ground truth is per-paper, so an arm returning ten chunks of one
  paper must not be scored as if it found ten relevant documents.

## How it can break

- **Ground truth is hand-curated.** A relevant paper missing from `relevant_ids` scores a correct
  retrieval as a miss. The seeded seminal papers exist so the ground truth is guaranteed present
  in the corpus.
- **`evaluate_retrieval` runs offline; `evaluate_answers` needs a key** and raises `LLMKeyMissing`
  without one.
- **Changing `TOP_K` changes every metric.** They are all "@k" measures; comparing runs at
  different k is meaningless.
- **Judge drift.** A different judge model would move the absolute answer-quality numbers. The
  cached grades pin the published comparison to the model that produced them.
- Tests pin the metric functions against hand-worked examples — including the empty-relevant and
  nothing-retrieved edge cases — and mock the judge.
