# 03 · Chunking strategies

Lesson 02 said we embed passages and search by nearness. This lesson answers the
question that immediately follows: **passages of what size?** Do we embed a whole
paper as one vector, one paragraph, one sentence? The answer — chunking — is one
of the highest-leverage decisions in a RAG system, and it's almost pure judgement.

## Why not embed the whole document?

An embedding model compresses its input into **one** vector. That's fine for a
sentence — one vector, one idea. But a whole paper contains many distinct ideas:
the method, the results, the limitations, the related work. Squeeze all of that
into a single vector and it becomes an *average* — a blurry point that's near
nothing in particular. A query about the method won't find it, because the vector
is half about results and half about everything else.

**So we cut documents into chunks and embed each chunk separately.** Now a query
about the method lands on the method chunk. Precision comes from chunks being
small enough that each one means one thing.

## Why not embed single sentences, then?

The opposite failure. A lone sentence often loses the context that makes it
meaningful. "It improves accuracy by 4%." — *what* does, on *what*? Embed that
sentence alone and it's ambiguous; the vector doesn't capture what it's really
about. Chunks that are too small are as bad as chunks that are too large.

## The Goldilocks size

The sweet spot is a few sentences — one coherent idea with enough surrounding
context to stand on its own. This project uses **~1000 characters** (a short
paragraph). That's the `CHUNK_SIZE_CHARS` constant in
[`src/config.py`](../src/config.py), and the reasoning is written right beside it.
There is no universal correct number; it depends on your documents. For dense
scientific abstracts, a paragraph is a good default.

## The overlap trick

Here's a subtle problem: if you cut documents into hard, non-overlapping slices,
a fact that happens to straddle a boundary gets split in half — the first half
ends one chunk, the second half starts the next, and *neither* chunk fully
contains it. A query for that fact might match neither well.

The fix is **overlap**: let each chunk share a little text with the next one
(this project uses 150 characters, ~15%). Now the boundary region appears in
*both* chunks, so a straddling fact survives intact in at least one of them. The
cost is a slightly larger index (some text is stored twice); the benefit is not
losing information at the seams. It's a classic engineering trade-off, and both
knobs are exposed as constants so they can be argued with.

## Chunks must carry their source

A chunk isn't just text — it's text **plus** the metadata needed to trace it back:
which paper it came from, its title, its position. Without that, the model could
retrieve a great passage and have no way to *cite* it, and you'd have no way to
check the answer. In this repo every `Chunk` carries its `arxiv_id` and `title`
(see [`src/chunking.py`](../src/chunking.py)), which is what makes grounded
citations possible in lesson 06.

## Where this shows up in the repo

- [`src/chunking.py`](../src/chunking.py) — `split_text` does the
  overlapping-window slicing (and tries to break on whitespace so it never cuts a
  word in half); `chunk_paper` carries metadata through.
- The size and overlap are `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` in
  [`src/config.py`](../src/config.py).
- [`tests/test_chunking.py`](../tests/test_chunking.py) pins the properties:
  overlap really overlaps, nothing exceeds the size bound, metadata is preserved.

Next: **[04 · Vector stores and ANN search](04_vector_stores_and_ann_search.md)** —
where all these chunk vectors live and how we search them fast.
