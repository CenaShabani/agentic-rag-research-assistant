# `src/vectorstore.py` — the one definition of "search"

## What it's for

Builds, saves, loads and searches the FAISS index. It is **the single definition of retrieval in
the project**: both the naive pipeline (`rag.py`) and the agentic system (`agent.py`) call
`VectorStore.search`, and neither re-implements it.

That is not code hygiene, it is what makes the headline finding trustworthy. The project's whole
claim is a comparison between two strategies. If each had its own retriever, any measured
difference could be an accident of implementation rather than a real difference in strategy, and
no number in the README would mean anything.

## The public surface

| Member | In | Out |
|---|---|---|
| `VectorStore.build(chunks)` | `DataFrame` with `chunk_id, arxiv_id, title, text` | a `VectorStore` — **the expensive call**, embeds everything |
| `.save()` / `VectorStore.load()` | — | writes/reads `data/index/` — load is fast, no embedding |
| `.search(query, k=TOP_K)` | `str`, int | `list[Retrieved]`, best first |
| `.search_rows(query, k=TOP_K)` | `str`, int | `list[(row, score)]` — the raw ranking |
| `.meta` | — | the row-aligned chunk metadata frame |
| `.size` | — | `int`, number of indexed chunks |

`Retrieved` carries `chunk_id`, `arxiv_id`, `title`, `text`, `score` — a cosine similarity in
[-1, 1], higher is more similar. Everything needed to answer *and to cite*, in one object.

**`search_rows` and `meta` exist for the benchmark, and they were added carefully.** The
retrieval arms in `retrievers.py` need two things this class already knew but did not expose:
the ranking as `(row, score)` pairs before metadata is attached (reciprocal-rank fusion needs
positions and row ids, not citation fields), and the metadata frame itself (so the lexical arm
indexes the *same* chunks in the *same* row order, which is what lets two rankings be fused by
row id without a join).

`search` was then rewritten to sit **on top of** `search_rows`, so there is still exactly one
place in this project where FAISS is asked a question. Adding a second search path is precisely
what the module docstring warns against — two paths could disagree, and then no published number
could be trusted. `test_search_rows_and_search_agree` pins that they do not.

## The walkthrough

**The invariant this class exists to protect:** the FAISS index stores only vectors, and the
metadata frame stores the text and citation fields, in **the same row order**. FAISS returning
row `i` means "the chunk at metadata row `i`". Nothing else keeps them aligned — there is no
join key inside FAISS, only integer positions.

That is why `__init__` calls `meta.reset_index(drop=True)`. A metadata frame arriving with a
non-contiguous index — trivially easy to produce by filtering a DataFrame upstream — would make
`.iloc[row]` return the wrong chunk. Not an error: a *wrong citation*, which is the worst failure
mode this project has. One line, and it is the most important line in the file.

`build` embeds every chunk, creates `IndexFlatIP(EMBED_DIM)` and adds the vectors, then keeps
exactly four metadata columns. `IndexFlatIP` does an **exact** search — it compares the query
against every vector. At 4,298 chunks that is instant, and crucially it removes approximate-search
recall as a variable. When the evaluation reports recall@5 = 0.60 on multi-hop questions, that
number is a property of the embeddings and the chunking, not of an ANN index's error. Reaching
for an approximate index here would have been answering a scale problem the corpus doesn't have,
at the cost of muddying the finding.

`search` embeds the query, asks FAISS for top-k, then maps rows back to metadata:

```python
for score, row in zip(scores[0], idxs[0]):
    if row < 0:
        continue
```

**The `row < 0` skip is not paranoia.** FAISS pads its result array with `-1` when `k` exceeds
the number of vectors in the index — a real case on a small or partially-built index. Without
the guard, `.iloc[-1]` would silently return the *last* chunk in the corpus, scored as if it
were a genuine hit. Python's negative indexing turns a sentinel into plausible garbage.

`[0]` on `scores` and `idxs` is because FAISS is batch-oriented: it returns one row of results
per query and we always pass exactly one.

## The non-obvious bits

- **`import faiss` is inside each method**, matching the lazy-import pattern in `embeddings.py`,
  so importing the module is cheap and test collection doesn't need FAISS present.
- **`float(score)` and `int(row)`** convert numpy scalars to Python types, so `Retrieved` objects
  serialise to JSON cleanly — they end up in evaluation output and agent traces.
- **`save`/`load` are separate from `build`** so the expensive step happens once. `load` is what
  the notebook and `ask.py` use; nobody re-embeds a corpus to answer a question.
- **Only four metadata columns are stored.** `position` is dropped after chunking — it is not
  needed to answer or cite, and the index file is smaller without it.

## How it can break

- **Index and metadata written at different times will disagree.** They are saved together by
  `save()` for this reason; writing one by hand is how the alignment invariant dies.
- **A `k` larger than `size`** is handled by the `-1` skip, returning fewer hits rather than
  raising. Callers must not assume `len(hits) == k`.
- **Rebuilding after any change to chunking or the embedding model is mandatory** — old vectors
  and new text are not comparable, and the failure is silent.
- **The whole index lives in memory.** Fine at 4,298 chunks; a corpus two orders of magnitude
  larger needs a different index type, which is the honest scope limit stated in the README.
- Tests cover build → save → load → search round-tripping and that scores come back sorted.
