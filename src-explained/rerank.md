# `src/rerank.py` — the cross-encoder, and what it cost

## What it's for

Scores a `(query, document)` pair by reading **both together**, rather than comparing two
vectors that were computed independently. It is the most accurate ranking idea in the project
and by far the most expensive, and measuring that trade is the reason the file exists.

Fifty-odd lines, two functions. Almost all of the value is in understanding *why* it is slow,
because that is the thing an interviewer will ask about.

## Bi-encoder vs cross-encoder — the whole idea

Everything else in this project is a **bi-encoder** design:

```
bi-encoder    : encode(query) · encode(document)      ← one vector each, compared
cross-encoder : model(query + document) → score       ← one forward pass per pair
```

The bi-encoder embeds each document **once, at index time, without knowing the query**. That is
what makes it fast — all the document work happens before the query arrives — and it is also its
ceiling: the document's vector has to be a summary of everything the document might ever be
asked about.

The cross-encoder has no such constraint. Query and document go through a transformer that can
attend across both, so it sees interactions two independent summaries cannot represent. Nothing
can be precomputed, so cost is **per candidate**, not per query.

That is why real systems use both in the arrangement `retrievers.RerankRetriever` implements: a
cheap first stage retrieves a shortlist, the expensive model rescores only that shortlist, and
`config.RERANK_CANDIDATES` is the dial between them.

## The public surface

| Function | In | Out |
|---|---|---|
| `get_cross_encoder()` | — | a cached `CrossEncoder` |
| `rerank_scores(query, documents)` | `str`, `list[str]` | `np.ndarray` of relevance logits |

## The walkthrough

The structure deliberately mirrors `embeddings.py`: `@functools.lru_cache(maxsize=1)` around a
loader whose import sits *inside* the function. Importing this module is therefore free — no
sentence-transformers requirement, no ~90 MB download — and the model becomes a singleton loaded
at most once per process. The same pattern, in the same shape, for the same two reasons.

`ms-marco-MiniLM-L-6-v2` is trained on MS MARCO passage ranking: real search queries against
real passages. That is the task shape this benchmark measures, which is why it was chosen over a
general-purpose model.

**The returned scores are raw logits.** Not probabilities, not bounded to any range, and not
comparable to another arm's scores. They are only ever used to *order* candidates — never
thresholded, never fused with a cosine similarity. Worth being explicit about, because "it
returns a number between 0 and 1" is a natural and wrong assumption that would corrupt any
attempt to combine it with something else.

The empty-list guard returns a correctly-shaped empty array rather than calling the model with
nothing, which some versions handle badly.

## What the benchmark actually found

This is the part worth carrying into an interview.

| | nDCG@10 | median latency |
|---|--:|--:|
| Cross-encoder rerank | 0.964 | 2,465 ms |
| BM25 alone | **0.978** | **2.0 ms** |

**The reranker spent roughly 1,200× BM25's latency to buy nothing measurable.** An earlier
version of this project's README recommended a cross-encoder as "the obvious next iteration"
that "would lift MRR". Building it is what proved that wrong.

Be precise about *how* wrong, because the first attempt at this sentence was also wrong. The
point estimates say the reranker scores lower, and an earlier draft said exactly that — but
paired per question the gap is +0.014 in BM25's favour with 8 wins, 6 losses and 166 ties, and
Holm-corrected p = 0.10. That does not support "scores lower"; it supports **no measurable
difference**. Which is the stronger claim: "slightly worse" invites an argument about whether
the difference matters, while "indistinguishable at a thousand times the price" does not. See
`significance.md`.

The reason is not that the model is bad — it is that a reranker can only reorder what the first
stage handed it, and on this corpus the first stage's ordering was already close to optimal.
There was nothing left to rescue. A reranker earns its cost when first-stage recall is high but
its *ordering* is poor; here the ordering was fine and the ceiling had already been hit.

It is also the arm most sensitive to hardware: 2.6 s is 50 candidate forward passes on a laptop
CPU, single-threaded, batch size one. On a GPU with batching this would fall by orders of
magnitude, and the conclusion could change. That is stated in the README's limitations rather
than left for a reader to discover.

## How it can break

- **First call downloads ~90 MB** and blocks for tens of seconds. This is why
  `evaluation.evaluate_arms` runs a warm-up query per arm before timing anything — without it,
  the first measured query would carry the whole model load and the latency figure would be
  fiction.
- **Latency scales linearly with the shortlist**, so `RERANK_CANDIDATES` is the cost dial and
  belongs in `config.py`, not in a literal.
- **Long documents get truncated** at the model's max sequence length. Our chunks are ~1000
  characters so this does not currently bite; on full-text documents it would, silently.
- **It is never mocked away in the shipped path but always mocked in tests** —
  `test_retrievers.py` patches `rerank_scores` with a length-based stub, so the suite stays
  offline and fast while still pinning that the rerank step genuinely replaces the incoming
  order.
