# 05 · The RAG pipeline end to end

This is the spine of the whole project. Everything in lessons 02–04 was a part;
here they snap together into one pipeline. If you understand this lesson, you
understand the system.

## The six stages

```
  ACQUIRE  ->  CHUNK  ->  EMBED  ->  INDEX   ||   RETRIEVE  ->  GENERATE
  \___________________ build time ___________/     \____ query time ____/
```

There are two phases, and telling them apart is the single most clarifying thing
about RAG:

- **Build time** (run once, ahead of any question): turn a pile of documents into
  a searchable index. Stages 1–4. This is [`run_pipeline.py`](../run_pipeline.py).
- **Query time** (run per question): use that index to answer. Stages 5–6. This is
  [`src/rag.py`](../src/rag.py) / [`src/agent.py`](../src/agent.py).

You pay the expensive work (embedding the whole corpus) *once* at build time, so
each question is cheap: embed one query, search, generate.

## Stage by stage

**1. Acquire** — get the documents. Here, download recent arXiv papers plus a
seeded set of seminal ones, and record *provenance* (what was fetched, when, and a
checksum) so the corpus is traceable. [`src/acquisition.py`](../src/acquisition.py).

**2. Chunk** — cut each document into overlapping, idea-sized pieces, each carrying
its source metadata. [`src/chunking.py`](../src/chunking.py). (Lesson 03.)

**3. Embed** — turn every chunk into a normalized vector with the local
sentence-transformer. [`src/embeddings.py`](../src/embeddings.py). (Lesson 02.)

**4. Index** — load all those vectors into a FAISS index and save it to disk,
row-aligned with the chunk metadata. [`src/vectorstore.py`](../src/vectorstore.py).
(Lesson 04.)

*...index now sits on disk. Questions can start arriving...*

**5. Retrieve** — embed the incoming question with the *same* model, search the
index, get the top-k most similar chunks. (Using the same embedding model at
build and query time is non-negotiable — the query has to live in the same
geometry as the chunks, or "nearest" means nothing.)

**6. Generate** — put those chunks into a grounded prompt and let the LLM write an
answer that cites them. (Lesson 06.)

## Why the split matters

Two practical consequences fall out of the build/query separation:

- **Updating knowledge is cheap.** New paper? Re-run build (or incrementally add
  it). No model retraining — exactly the point from lesson 01.
- **Retrieval is local and reproducible; generation is the only hosted part.**
  Stages 1–5 run on a laptop with no API key. Only stage 6 calls Claude. That's
  why this repo can reproduce every *retrieval* number offline and treats the LLM
  answers as a separately-cached layer.

## The single-source-of-truth rule

Notice that both the plain RAG path and the agentic path use the **same**
`VectorStore.search`. There is exactly one definition of "retrieve" in the repo.
That's deliberate: if there were two, they could drift, and you could never tell
whether a difference between naive and agentic RAG came from the *strategy* or
from an accidental difference in *search*. One retriever, two strategies on top —
that's what makes the lesson-09 comparison fair.

## Where this shows up in the repo

- [`run_pipeline.py`](../run_pipeline.py) is stages 1–4, in order, with progress
  prints. Run it once; retrieval is then ready.
- `ask.py "your question"` runs stages 5–6 for one question.

Next: **[06 · Prompting and grounding](06_prompting_and_grounding.md)** — the
craft of stage 6.
