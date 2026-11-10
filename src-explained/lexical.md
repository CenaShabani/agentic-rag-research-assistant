# `src/lexical.py` — BM25, the baseline that won

## What it's for

Ranks documents by **literal term overlap** — no embeddings, no neural network, no model
download. It is the "boring option" the dense retriever has to beat, and on this corpus it is
the option that wins: nDCG@10 of 0.978 at 2.0 ms, against the dense arm's 0.751 at 17.4 ms.

That result is the reason this file matters more than its ~110 lines suggest. If the baseline
were wrong, the headline claim would be wrong in the most embarrassing possible direction.

## The public surface

| Member | In | Out |
|---|---|---|
| `BM25(documents)` | `list[str]` | an index over that fixed collection |
| `.size` | — | `int`, number of documents |
| `.scores(query)` | `str` | `np.ndarray` shape `(n_docs,)`, one score per document |
| `.top_k(query, k)` | `str`, `int` | `list[(row, score)]`, best first, zero-scoring rows dropped |

`top_k` returns **row indices**, not text. That is deliberate: the rows are positions in the
same metadata frame `VectorStore` uses, which is what lets `retrievers.py` fuse a lexical
ranking with a dense one by row id and no join.

## Why it is written out rather than imported

`rank_bm25` is one pip install away. Three reasons it isn't used:

1. **It is the baseline the headline claim is measured against.** An imported black box makes
   the thing your conclusion rests on un-inspectable.
2. **Every parameter changes the answer.** The project's rule (see `config.py`) is that such a
   number must be visible and defensible. `k1` and `b` are named constants with a written why.
3. **It adds no dependency.** The sparse count matrix comes from scikit-learn, already pinned.

It is also about forty lines of actual arithmetic. The cost of writing it is low and the cost of
not understanding it, in an interview about retrieval, is high.

## The formula, and what each piece is doing

```
score(D, Q) = Σ over query terms t of
                  IDF(t) · ( f(t,D) · (k1 + 1) )
                         / ( f(t,D) + k1 · (1 − b + b · |D|/avgdl) )
```

- **IDF(t)** — a term in every document tells you nothing. "the" is worthless, "adstock" is
  decisive. Rare terms get weighted up.
- **k1 (saturation)** — the tenth mention of a word does not make a document ten times as
  relevant. The ratio saturates, giving term frequency diminishing returns.
- **b (length normalisation)** — a long document contains more words and would otherwise win by
  size alone. `b` scales the penalty by how far the document strays from the average length.

Both are at the Lucene/TREC defaults (1.5, 0.75), **deliberately not tuned**. A baseline tuned by
the person hoping to beat it is not a baseline.

## The walkthrough

The constructor does all the work, once, so that query time is cheap.

`CountVectorizer` produces `tf[d, t]` — how often term `t` occurs in document `d` — as a sparse
matrix. No stop-word list and no stemming: BM25's IDF already discounts ubiquitous words, and a
hand-curated stop list would be a tuning knob this baseline is not allowed.

**`doc_freq` uses `(tf > 0).getnnz(axis=0)`.** That counts non-zero entries per column, which is
exactly "in how many documents does this term appear". Written the obvious way — `(tf >
0).sum(axis=0)` on a dense array — it would materialise a 4,298 × ~40,000 boolean matrix.

**The `+1` inside the IDF log is load-bearing.** The Robertson/Sparck-Jones formula without it
goes *negative* for any term appearing in more than half the documents, which means a matching
document gets pushed **down** the ranking for matching. That is not a behaviour anyone wants from
a baseline, and it is silent — no error, just quietly inverted results on common terms.

**The precomputation is the performance trick.** Everything in the formula that depends only on
the document — the whole saturation-and-length denominator — is applied once to the sparse
matrix's stored values at build time:

```python
row_norm = np.repeat(length_norm, np.diff(tf.indptr))   # expand per-row normaliser
weighted = numerator / denominator                       # elementwise, on .data
self._weights = weighted.multiply(self._idf).tocsc()     # fold IDF in as a column scale
```

After this, `weights[d, t]` is the **complete contribution** of term `t` to document `d`'s score.
A query is then just "sum these columns" — a sparse slice and a sum, not a Python loop over
4,298 documents. This is why the arm runs in 2.0 ms.

`np.repeat(length_norm, np.diff(tf.indptr))` is the only genuinely dense line. `tf.indptr` marks
where each row's stored values begin in `.data`, so `np.diff` gives the count of stored values
per row, and `np.repeat` stretches each row's single normaliser across exactly its own entries.

**`.tocsc()` at the end** switches to column-major storage because queries slice by *column*
(term). Leaving it in CSR would make every query slice fight the layout.

## The non-obvious bits

- **`scores` uses `np.unique(term_ids)`.** A repeated query term counts once. BM25 scores
  *document* term frequency; query term frequency only matters for queries long enough that it
  isn't worth modelling here.
- **Unknown query terms just have no column and contribute nothing** — no crash, no special
  case. A query where *no* term is known scores every document 0.0, which the caller sees as a
  tie.
- **`top_k` drops zero-scoring documents rather than padding.** A document scoring exactly 0
  shares no terms with the query at all. Returning ten arbitrary non-matches as filler would
  inflate every recall number the benchmark reports — `test_lexical.py` pins this.
- **`argpartition` before `argsort`.** Finding the top 10 of 4,298 without sorting all of them,
  then sorting only those 10. `kind="stable"` keeps ties in a deterministic order, which matters
  because a published number has to reproduce exactly.

## How it can break

- **The document order must match the vector store's row order.** `BM25Retriever` builds this
  from `store.meta["text"]` for exactly that reason. Build it from a re-sorted frame and the
  hybrid arm will fuse row 7's lexical score with row 7's *different* dense document — silently,
  with plausible-looking output.
- **The index is fixed at construction.** There is no `add`. Adding a document means rebuilding,
  which is fine at this scale (a couple of seconds) and would not be at a million.
- **Memory is the vocabulary, not the corpus.** The weight matrix is `n_docs × n_terms` sparse.
  A corpus with a huge vocabulary of rare terms (identifiers, hashes, OCR noise) grows this fast.
- **It cannot match what is not there.** The `s5` failure in the benchmark — the query says
  "word2vec", the corpus says "word representations in vector space", and the literal token
  appears **zero times in 4,298 chunks** — is not a bug in this file. It is the permanent
  limitation of lexical matching, and it is why the dense arm is kept rather than deleted.

## Where it's used

`retrievers.BM25Retriever` (the lexical arm) and `retrievers.HybridRRFRetriever` (which builds
its own instance for fusion). Nothing else — the shipped RAG and agent paths still go through
`VectorStore`. Swapping the shipped default to BM25 is the benchmark's first recommendation, but
that is a decision for the reader, not a change this file makes on its own.
