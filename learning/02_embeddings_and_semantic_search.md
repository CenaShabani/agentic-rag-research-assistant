# 02 · Embeddings and semantic search

Retrieval's job is to find passages *relevant* to a question. The old way was
keyword matching: find documents that contain the same words as the query. That
breaks the moment the document says the same thing in different words — and it
almost always does. "How do transformers attend to tokens?" and "the attention
mechanism computes weighted sums over the sequence" share barely a word, yet one
answers the other. We need to match on **meaning**, not spelling. That is what
embeddings give us.

## The core idea: meaning as location

An **embedding model** takes a piece of text and returns a vector — a list of
numbers, a point in a high-dimensional space. It is trained so that **texts with
similar meaning land near each other**, and unrelated texts land far apart.

Think of it as a map. On a real map, cities that are geographically close are
close on the page. On an embedding map, *ideas* that are close in meaning are
close in the space. "Dog" sits near "puppy" and "canine"; "volcano" sits far
away in another region entirely. The model learned this geometry from vast
amounts of text, without anyone labelling it.

The vectors here are 384 numbers long (that's the "384-dimensional" in the config).
You can't picture 384 dimensions — nobody can — but the math of "near" and "far"
works the same as it does in 2D.

## Searching by meaning

Once every passage in the corpus is a point, search becomes geometry:

1. Embed the **query** into the same space — it becomes a point too.
2. Find the passage points **nearest** to the query point.
3. Those nearest passages are the most semantically relevant. Return them.

No shared keywords required. The query and the passage can use totally different
words; if they mean the same thing, their points are close, and the search finds
them. This is **semantic search**, and it is what makes RAG retrieval work.

## How "near" is measured: cosine similarity

The standard measure of "near" for embeddings is **cosine similarity** — the
cosine of the angle between two vectors. It ranges from -1 (opposite) to 1
(identical direction). Two vectors pointing the same way (similar meaning) score
near 1; perpendicular ones (unrelated) score near 0.

A useful trick: if you **normalize** every vector to length 1 (make it a *unit*
vector), then the plain dot product of two vectors *equals* their cosine
similarity. That's why [`src/embeddings.py`](../src/embeddings.py) passes
`normalize_embeddings=True` — it lets the vector index use a fast dot-product
search that still means "most similar in meaning" (lesson 04).

## Why we run embeddings locally

This project embeds text with `all-MiniLM-L6-v2`, a small
[sentence-transformer](https://www.sbert.net/) model that runs on a CPU in
milliseconds. Two reasons, both about reproducibility: it needs no API key, and
it's deterministic — the same text always produces the same vector, so every
retrieval number in this repo reproduces exactly on any machine. (The *answers*
use a hosted LLM; the *retrieval* stays fully local.)

## Where this shows up in the repo

- [`src/embeddings.py`](../src/embeddings.py) — `embed_texts` / `embed_query`
  wrap the sentence-transformer and normalize the vectors.
- The choice of model and its 384 dimensions are named constants in
  [`src/config.py`](../src/config.py) (`EMBED_MODEL`, `EMBED_DIM`), with the
  rationale written next to them.

Next: **[03 · Chunking strategies](03_chunking_strategies.md)** — why we embed
*pieces* of documents, not whole documents.
