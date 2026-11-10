# `src/retrievers.py` — the four arms behind one interface

## What it's for

Wraps four different answers to "which chunks are relevant to this query?" in a single shape, so
the metric code, the figures and the notebook never learn which one they are looking at. Without
this file the benchmark would be four near-duplicate scoring loops, and any difference between
them would be as likely to come from the loop as from the retrieval.

## The public surface

| Class | Adds | Cost on this corpus |
|---|---|---|
| `DenseRetriever(store)` | semantic matching | 17.4 ms · nDCG 0.751 |
| `BM25Retriever(store)` | literal term overlap | 2.0 ms · nDCG 0.978 |
| `HybridRRFRetriever(store)` | fusing the two rankings | 24.5 ms · nDCG 0.874 |
| `RerankRetriever(hybrid)` | joint query-document reading | 2,465 ms · nDCG 0.964 |
| `build_all(store)` | all four, in cost order | — |

Every one has a `name` and a `search(query, k) -> list[Retrieved]`, which is deliberately the
same signature as `VectorStore.search` — so the store itself already satisfies the protocol and
no adapter was needed for the shipped system.

## The invariant this file must not break

`vectorstore.py` calls itself "the single definition of search the corpus", and `rag.py` and
`agent.py` both depend on that. `DenseRetriever` therefore **delegates**:

```python
def search(self, query, k=config.TOP_K):
    return self._store.search(query, k=k)
```

Three lines that could have been "embed the query, call FAISS, map the rows" — and that
temptation is exactly the trap. A re-implementation would drift, and then the benchmark would be
measuring a retriever the shipped system does not use, while every published number claimed
otherwise. `test_retrievers.py::test_dense_arm_is_the_shipped_retriever` asserts the two return
*identical* hits, so this cannot rot quietly.

## Why the arms are ordered this way

Each arm is the previous one **plus exactly one idea**:

1. `dense` — does semantic matching work here?
2. `bm25` — could plain word overlap have done it?
3. `hybrid` — are their mistakes different, or the same mistakes twice?
4. `rerank` — does reading query and document jointly beat comparing two independent summaries,
   and what does it cost?

An arm that added two things at once would be unattributable, which is why there is no "hybrid +
query expansion + reranking" arm. This is the same discipline as an ablation study.

## The walkthrough

**`_row_to_retrieved` is the funnel.** Every arm, however it ranked, ends up building its
results through this one function. That is what guarantees all four return the same citation
fields from the same metadata frame — an arm cannot accidentally cite differently because it
ranked differently.

**`HybridRRFRetriever._fused_rows` is the interesting method.**

```python
for ranking in (dense_rows, bm25_rows):
    for rank, (row, _score) in enumerate(ranking, start=1):
        fused[row] = fused.get(row, 0.0) + 1.0 / (config.RRF_K + rank)
```

Note what is *not* used: `_score`. Reciprocal-rank fusion throws the scores away and keeps only
the positions. This is not laziness — it is the point. A cosine similarity lives in `[-1, 1]`; a
BM25 score is unbounded and depends on the collection's term statistics. They share no scale, so
adding or averaging them is meaningless, and any fixed weighting is a fudge factor. Ranks are
the one thing the two arms produce that *is* comparable.

The behaviour this buys: a document ranked 1st by one arm and unranked by the other still scores
well, but a document ranked 3rd by **both** can beat it. Agreement between two different kinds
of evidence is a stronger signal than one arm's confidence. `RRF_K = 60` (Cormack et al., 2009)
damps the very top ranks so one arm's confident-but-wrong first hit cannot dominate.

**Each arm is asked for a deeper list than the caller wants.** A document must be ranked by at
least one arm to be fusable at all, so fusing two top-10 lists could only ever consider 20
candidates. `depth` defaults to `RERANK_CANDIDATES` (50).

**`RerankRetriever` reranks the hybrid arm, not the dense arm.** A cross-encoder can only
reorder what it is given — it never retrieves. So recall going *into* it should be as high as
possible, and the fused shortlist has better recall than either arm alone.
`test_rerank_cannot_invent_documents_outside_the_shortlist` pins that limitation as a property.

## The non-obvious bits

- **`from .rerank import rerank_scores` is inside `search`,** not at module top. Importing
  `retrievers` must not trigger a 90 MB cross-encoder download — the tests import this module
  constantly and never want the model.
- **`build_all` constructs BM25 twice** (once for the lexical arm, once inside the hybrid arm).
  A couple of seconds, accepted so each arm stays a self-contained object that can be built and
  reasoned about alone. If the corpus grew, sharing one instance would be the first optimisation.
- **`RerankRetriever` reaches into `hybrid._fused_rows` and `hybrid._meta`.** Underscore-private
  access between two classes in the same module, done knowingly: the alternative was widening
  the public surface for one internal collaborator.
- **`np.argsort(-scores, kind="stable")`** — stable so that ties resolve in the shortlist's
  order deterministically. A published number has to reproduce exactly.

## What the benchmark revealed about these arms

Worth recording here because it contradicts the intuition the file was written with:

- **Hybrid is not free insurance.** It scores *below* BM25 alone (0.874 vs 0.978), and on the
  curated `s5` question it inherits BM25's failure rather than dense's success — fusing a
  confident-but-wrong ranking with a correct one pushed the correct document out of the top 10.
  RRF averages toward the weaker arm when the arms are badly mismatched in quality.
- **The reranker cannot rescue a bad first stage,** and on this corpus the first stage was
  already good enough that there was nothing to rescue. It paid 1,200× BM25's latency for a
  difference that does not survive a paired test (+0.014 in BM25's favour, p = 0.10). The cutoff
  sweep shows the mechanism: the dense arm's recall is still only 0.856 at k=50, so what it
  misses is never in the shortlist to be reordered.

## How it can break

- **All arms must index the same collection in the same row order.** They share
  `store.meta`, which is what makes row-id fusion legal. Pass a re-sorted frame to one of them
  and the fusion silently combines different documents.
- **The reranker's shortlist depth is a real quality dial.** Too few and it cannot rescue a
  buried relevant document; too many and latency grows for candidates that were never plausible.
  It is `config.RERANK_CANDIDATES`, not a literal, for that reason.
- **`Retriever` is a `Protocol`, so nothing enforces it at runtime.** A new arm that forgets
  `name` fails at figure-drawing time, not at construction.
