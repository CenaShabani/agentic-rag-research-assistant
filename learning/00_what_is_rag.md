# 00 · What is RAG?

## The one-sentence version

**Retrieval-Augmented Generation (RAG) means: before a language model answers a
question, go fetch the relevant documents and put them in front of it — so it
answers from those documents instead of from memory.**

That's it. Everything else in this project is machinery to do that well.

## The exam analogy

Imagine two students taking the same test about a stack of research papers.

- **Student A** read the papers months ago and now sits the exam **closed-book**.
  They answer from memory. They'll get the gist right, misremember specifics, and
  occasionally state something false with total confidence.
- **Student B** sits the exam **open-book**. For each question, they flip to the
  relevant pages, read them, and answer with the pages open in front of them.
  They can quote, they can cite, and when the book doesn't cover something, they
  can say so.

A plain language model is Student A. **RAG turns it into Student B.** The model's
job shifts from *recall* ("what do I remember about this?") to *reading
comprehension* ("here are the relevant passages — what do they say?"). Reading
comprehension is a much easier, much more reliable task.

## The two halves of the name

- **Retrieval** — finding the right passages for a question. This is a *search*
  problem, and it's most of the work. If retrieval hands the model the wrong
  passages, no amount of clever generation will save the answer.
- **Generation** — the language model reading those passages and writing the
  answer. This is where the LLM earns its keep: synthesizing, summarizing, and
  phrasing — grounded in what retrieval found.

**Augmented** is the glue: the retrieved passages *augment* the prompt. The model
still generates the answer, but its input now includes evidence it didn't have to
remember.

## Why this is the dominant pattern

When you ask a chatbot about your company's documents, a product's manual, or last
week's news, it is almost certainly doing RAG under the hood. RAG is popular
because it fixes the three things a bare LLM is worst at — being out of date,
forgetting details, and making things up — without the enormous cost of
retraining the model. Lesson 01 is exactly those three problems.

## Where this shows up in the repo

The whole project is one RAG system over a corpus of arXiv papers:

- **Retrieval** lives in [`src/vectorstore.py`](../src/vectorstore.py) (finding
  passages) built on [`src/embeddings.py`](../src/embeddings.py) (turning text
  into searchable vectors).
- **Generation** lives in [`src/rag.py`](../src/rag.py) (the plain version) and
  [`src/agent.py`](../src/agent.py) (the agentic version).

Next: **[01 · Why LLMs need retrieval](01_why_llms_need_retrieval.md)** — the
three concrete failures RAG is built to fix.
