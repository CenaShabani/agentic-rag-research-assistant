"""BM25 — the lexical baseline the dense retriever has to beat.

Dense embedding retrieval is the interesting half of this project, but it is only
interesting if it beats the boring option. BM25 is the boring option: a
term-frequency ranking function from the 1990s, no neural network, no GPU, no
model download. It is still the default first-stage retriever in production
search systems, and it is still competitive on a great many benchmarks. A RAG
system that cannot beat BM25 has not earned its embedding model.

**Why implement it rather than import it.** `rank_bm25` is one pip install away.
But BM25 is about forty lines, every one of its parameters changes the answer, and
this project's rule is that a number which changes the answer has to be visible
and defensible (see `config.py`). An imported black box would make the baseline
un-inspectable, and the baseline is the thing the headline claim is measured
against. It also adds no new dependency: the sparse count matrix comes from
scikit-learn, which is already pinned.

**The formula.** For a query Q and document D:

    score(D, Q) = SUM over terms t in Q of
                      IDF(t) * ( f(t,D) * (k1 + 1) )
                             / ( f(t,D) + k1 * (1 - b + b * |D| / avgdl) )

Three ideas, each doing a specific job:

  * **IDF(t)** — a term appearing in every document tells you nothing. "the" is
    worthless; "adstock" is decisive. IDF weights rare terms up.
  * **k1 (saturation)** — the tenth mention of a word does not make a document ten
    times as relevant. The ratio saturates, so term frequency has diminishing
    returns rather than unbounded ones.
  * **b (length normalisation)** — a long document contains more words and would
    otherwise win by sheer size. b scales the penalty by how far the document's
    length strays from the collection average.

Both k1 and b are set to the standard Lucene/TREC defaults in `config.py`. They
are deliberately *not* tuned: a baseline tuned by the person hoping to beat it is
not a baseline.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import config


class BM25:
    """A BM25 index over a fixed collection of documents.

    Built once from the same chunk texts the FAISS index was built from, so the
    lexical and dense arms see exactly the same collection in exactly the same
    row order. That row alignment is what lets `retrievers.py` fuse their two
    ranked lists without a join.
    """

    def __init__(self, documents: list[str]):
        from sklearn.feature_extraction.text import CountVectorizer

        # Lowercase, split on word characters, keep every term. No stop-word list
        # and no stemming: BM25's IDF already discounts ubiquitous words, and
        # hand-curated stop lists are a tuning knob this baseline is not allowed.
        self._vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")

        # tf[d, t] = how many times term t occurs in document d.
        tf = self._vectorizer.fit_transform(documents).astype(np.float32)
        self._n_docs = tf.shape[0]

        doc_lengths = np.asarray(tf.sum(axis=1)).ravel()
        avg_doc_length = float(doc_lengths.mean()) if self._n_docs else 0.0

        # Document frequency: in how many documents does each term appear at all?
        # `.getnnz(axis=0)` counts non-zero entries per column, which is exactly
        # that, and avoids materialising a dense boolean matrix.
        doc_freq = np.asarray((tf > 0).getnnz(axis=0)).ravel().astype(np.float32)

        # Robertson/Sparck-Jones IDF with the +0.5 smoothing terms and a +1 inside
        # the log. The +1 is what keeps the value non-negative: without it, a term
        # occurring in more than half the documents scores negatively and can
        # actively push a matching document *down*, which is not a behaviour
        # anyone wants from a baseline.
        self._idf = np.log(1.0 + (self._n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        # Precompute the whole denominator-dependent part of the score. Every
        # element of the sparse tf matrix is transformed once, here, instead of
        # once per query — which is what makes query time a sparse matrix-vector
        # product rather than a Python loop over documents.
        tf = tf.tocsr()
        length_norm = 1.0 - config.BM25_B + config.BM25_B * (
            doc_lengths / avg_doc_length if avg_doc_length else np.zeros_like(doc_lengths)
        )
        # Expand each row's normaliser across that row's stored values.
        row_norm = np.repeat(length_norm, np.diff(tf.indptr))
        numerator = tf.data * (config.BM25_K1 + 1.0)
        denominator = tf.data + config.BM25_K1 * row_norm
        weighted = sp.csr_matrix(
            (numerator / denominator, tf.indices, tf.indptr), shape=tf.shape
        )

        # Fold IDF in as a column scaling, so a query becomes a plain sum over the
        # matching columns. `weights[d, t]` is now the full contribution of term t
        # to document d's score.
        self._weights = weighted.multiply(self._idf).tocsc().astype(np.float32)

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return self._n_docs

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for `query`, as a dense (n_docs,) array.

        A query term the collection has never seen simply has no column and
        contributes nothing — no crash, no special case. A query where *no* term
        is known scores every document 0.0, which the caller sees as a tie.
        """
        # Reuse the fitted vocabulary: `transform`, not `fit_transform`.
        q = self._vectorizer.transform([query])
        term_ids = q.indices
        if term_ids.size == 0:
            return np.zeros(self._n_docs, dtype=np.float32)

        # Sum the precomputed per-term contribution columns. Repeated query terms
        # count once: BM25 scores document term frequency, not query term
        # frequency, for a query this short.
        cols = self._weights[:, np.unique(term_ids)]
        return np.asarray(cols.sum(axis=1)).ravel()

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """The `k` best-scoring document rows for `query`, best first.

        Uses `argpartition` to find the top k without sorting all ~4,300 scores,
        then sorts only those k. Documents scoring exactly 0 share no terms with
        the query at all and are dropped rather than returned as filler — an
        empty result is a more honest answer than ten arbitrary non-matches.
        """
        scores = self.scores(query)
        k = min(k, scores.size)
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0.0]
