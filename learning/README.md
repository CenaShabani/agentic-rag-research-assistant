# Learning RAG from zero

This folder is a short course, not documentation. I wrote it while building the
project, for someone (me, six months from now, in an interview) who starts
knowing nothing about retrieval-augmented generation and wants to be able to
*explain* every part of this repo — not just run it.

Read the lessons in order. Each one is a few minutes, builds on the last, and
points at the exact file in `src/` where the idea becomes code, so you can go
from concept → implementation without a gap.

| # | Lesson | The one idea |
|---|--------|--------------|
| 00 | [What is RAG?](00_what_is_rag.md) | Give a language model an open-book exam instead of a closed-book one. |
| 01 | [Why LLMs need retrieval](01_why_llms_need_retrieval.md) | Models are frozen, forgetful, and confidently wrong; retrieval fixes all three. |
| 02 | [Embeddings and semantic search](02_embeddings_and_semantic_search.md) | Turn meaning into geometry so "close" means "related". |
| 03 | [Chunking strategies](03_chunking_strategies.md) | One vector can only mean one thing — so cut documents into idea-sized pieces. |
| 04 | [Vector stores and ANN search](04_vector_stores_and_ann_search.md) | An index that finds nearest neighbours in a sea of vectors, fast. |
| 05 | [The RAG pipeline end to end](05_the_rag_pipeline_end_to_end.md) | Acquire → chunk → embed → index → retrieve → generate. |
| 06 | [Prompting and grounding](06_prompting_and_grounding.md) | How to make the model answer *from the sources* and cite them. |
| 07 | [What makes it agentic](07_what_makes_it_agentic.md) | Let the model *decide* how to retrieve, not just retrieve once. |
| 08 | [Agent tools and loops](08_agent_tools_and_loops.md) | The tool-use loop: search, read, judge, search again, answer. |
| 09 | [Evaluating RAG](09_evaluating_rag.md) | Measure retrieval and answers separately — and compare fairly. |
| 10 | [Failure modes and limits](10_failure_modes_and_limits.md) | Where RAG breaks: bad chunks, lost-in-the-middle, cost, injection. |
| 11 | [Benchmarking retrieval](11_benchmarking_retrieval.md) | nDCG, MAP and a bootstrap interval — how to know whether your retriever is actually better. |

If you only read three, read **00**, **05**, and **07** — they are the spine:
what RAG is, how the pipeline fits together, and what "agentic" adds on top.

Then read **11**. It is the lesson where this project's own conclusion was
overturned by its own measurements — a 1990s lexical baseline beat the embedding
model the system ships, and the cross-encoder an earlier README recommended
turned out to cost 1,200× the latency to buy *nothing measurable*. Lessons 09 and 11 together
are what separate a RAG demo from a defensible one.
