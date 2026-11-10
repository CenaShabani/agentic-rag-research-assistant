# 04 · Vector stores and ANN search

We have chunks (lesson 03) turned into vectors (lesson 02). Now we need a place to
keep thousands — eventually millions — of those vectors and a way to find the
nearest ones to a query *fast*. That's a **vector store**.

## The naive search, and why it's fine here

To find the passages nearest to a query, the honest method is: compare the query
vector against **every** stored vector, sort by similarity, take the top few.
This is **exact** nearest-neighbour search (sometimes "flat" or "brute-force").
Its cost grows linearly with the number of vectors.

For this project — a few thousand chunks — brute force is *instant*, so that's
exactly what we use ([`faiss.IndexFlatIP`](../src/vectorstore.py)). And it buys
something valuable: because the search is exact, retrieval quality reflects the
embeddings and the chunking, with **no approximation error** muddying the picture.
When we measure retrieval in lesson 09, we're measuring the real thing.

## Approximate nearest neighbour (ANN), and when you need it

Brute force stops being instant at scale. Search a *billion* vectors for every
query and linear cost hurts. This is where **Approximate Nearest Neighbour (ANN)**
comes in: clever index structures (HNSW graphs, IVF clustering, product
quantization) that find *almost* the nearest neighbours while looking at only a
small fraction of the vectors. You trade a little recall for a huge speed-up.

The key idea to carry into an interview: **exact vs approximate is a scale
decision.** Small corpus → exact, no downside. Huge corpus → ANN, accept a tiny
recall loss for tractable latency. This project is small, so it stays exact and
keeps the evaluation clean. FAISS supports both; swapping `IndexFlatIP` for an
ANN index is a one-line change if the corpus ever grows.

## Inner product, cosine, and the normalization trick

FAISS's `IndexFlatIP` ranks by **inner product** (dot product). On its own, inner
product favours *longer* vectors, not just similar-direction ones. But recall the
trick from lesson 02: if every vector is **normalized to unit length**, the inner
product *equals* cosine similarity — pure direction, pure meaning, no length bias.

That's the quiet contract between two files:
[`src/embeddings.py`](../src/embeddings.py) normalizes every vector, so
[`src/vectorstore.py`](../src/vectorstore.py) can use the fast inner-product index
and have "highest score" mean "most semantically similar". Break the normalization
and the search silently degrades — which is the kind of subtle coupling worth
knowing about.

## The index is only half of it

A vector index stores vectors and returns *row numbers* — "your nearest neighbours
are rows 42, 7, and 128." That's useless on its own; you need to map those rows
back to the actual chunk text and its citation metadata. So the store keeps two
things **row-aligned**: the FAISS index (vectors) and a metadata table (text +
`arxiv_id` + title). Row *i* in the index is row *i* in the table. Protecting that
alignment is the one invariant `VectorStore` exists to guard — get it wrong and
you return the right ranking of the *wrong* passages.

## No server, no Docker

This vector store is **in-process and file-backed**: build the index, write it to
`data/index/`, load it later. No database server, no container, no network. That's
a deliberate choice — it means the entire retrieval path runs on a laptop and
reproduces top-to-bottom in the capstone notebook. (Production systems often use a
hosted vector database — pgvector, Pinecone, Weaviate — but the *concepts* here
are identical; those are just this index with a network API and an ops team.)

## Where this shows up in the repo

- [`src/vectorstore.py`](../src/vectorstore.py) — `VectorStore.build/save/load`
  and the single `search` method that both RAG and the agent call.
- [`tests/test_vectorstore.py`](../tests/test_vectorstore.py) — pins the round-trip
  (save→load returns the same top hit), descending scores, and dimension handling.

Next: **[05 · The RAG pipeline end to end](05_the_rag_pipeline_end_to_end.md)** —
snapping all these pieces together.
