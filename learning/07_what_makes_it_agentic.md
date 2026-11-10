# 07 · What makes it agentic

This is the conceptual heart of the project. Everything so far describes *plain*
RAG: retrieve once, generate once. "Agentic" RAG changes one thing — **who decides
how to retrieve** — and that one change is what the whole evaluation is about.

## Plain RAG is a straight line

Recall the plain pipeline's query time:

```
question --> [retrieve top-k, once] --> [generate] --> answer
```

The retrieval is **fixed and up front**. The system embeds the question, grabs the
5 nearest chunks, and hands them over. One shot. If those 5 chunks contain the
answer, great. If they don't, the model is stuck — it never gets to look again. The
control flow is a hard-coded straight line; the LLM only shows up at the very end.

## Where plain RAG breaks: multi-hop questions

Consider: *"How does retrieval-augmented generation combine a dense retriever with
a sequence-to-sequence generator?"* Answering this well needs evidence from **two**
different papers — the RAG paper *and* the Dense Passage Retrieval paper. But a
single search with the whole question as the query tends to pull chunks that are
near the *overall* phrasing — likely several from the RAG paper, and it may never
surface the DPR paper at all. One retrieval, one region of the vector space, one
half of the answer. These are **multi-hop** questions — they need more than one hop
through the corpus — and single-shot retrieval is structurally bad at them.

## Agentic RAG puts the LLM in charge of retrieval

The agentic version flips the control flow. Instead of the code deciding "retrieve
once, then generate," the **LLM decides** how to retrieve, as it goes:

```
question --> LLM: "I'll search for X"  --> [retrieve] --> LLM reads results
         --> LLM: "not enough; search for Y" --> [retrieve] --> LLM reads
         --> LLM: "now I can answer"          --> answer
```

The model can:

- **Decompose** the question — search for "dense passage retriever" and
  "sequence-to-sequence generation" *separately*, each a focused query that
  retrieves better than the mashed-together whole.
- **Read and judge** — look at what came back and decide whether it has enough.
- **Search again** — reformulate and retrieve more if the first pass missed
  something. This second, targeted search is exactly what recovers the DPR paper.
- **Then answer** — grounded, as before, but on a richer, self-assembled evidence
  pool.

The retriever is *the same* `VectorStore.search`. What's new is that the LLM is now
a decision-maker in the loop, not just a text generator at the end. That's the
definition of "agentic": the model uses **tools** in a **loop**, choosing its own
actions toward a goal. (Lesson 08 is the mechanics of that loop.)

## The trade-off — this is the honest part

Agentic RAG is not free. Deciding, reading, and re-searching means **multiple LLM
calls and multiple retrievals** per question, where plain RAG has one of each. More
tokens, more latency, more cost. The entire point of the project's evaluation is to
ask the real question: **is the extra work worth it?** We expect the agent to win
on multi-hop questions (where the second search matters) and roughly tie on
single-hop ones (where one search was already enough) — while costing more either
way. Measuring that trade-off, rather than assuming the fancy version is better, is
the finding this project is built to produce (lesson 09).

## Why we built the loop from scratch

There are frameworks (LangGraph, LlamaIndex) that hand you an agent loop. This
project writes its own, deliberately, because **explainability is the goal**: every
decision the agent makes is a plain message we log, so the notebook can show its
actual reasoning trace — which query it chose, what came back, why it searched
again. A framework would hide that behind abstractions. Here you can read the loop
top to bottom in one file and defend every line.

## Where this shows up in the repo

- [`src/agent.py`](../src/agent.py) — the whole agentic loop, with the search
  trace (`AgentStep`) exposed for inspection.
- Contrast with the straight-line [`src/rag.py`](../src/rag.py).

Next: **[08 · Agent tools and loops](08_agent_tools_and_loops.md)** — how the loop
actually works, mechanically.
