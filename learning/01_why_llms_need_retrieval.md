# 01 · Why LLMs need retrieval

A language model is astonishing at language and unreliable at facts. RAG exists
because of three specific, well-understood weaknesses. Understanding them is the
whole motivation for the project — if you can name these three in an interview,
you can justify every design choice that follows.

## 1. The knowledge is frozen

A model's knowledge is baked in at training time. Ask it about anything that
happened after its training cutoff — a paper published last week, a price that
changed yesterday — and it simply cannot know. Retraining a large model to add
new knowledge costs a fortune and happens rarely.

**Retrieval sidesteps this entirely.** The model doesn't need to *know* the new
paper; it needs to be *handed* the new paper at question time. To update a RAG
system's knowledge, you add documents to the corpus. No retraining. That is a
huge part of why RAG is everywhere.

## 2. The context window is small, but the corpus is large

"Why not just paste all the documents into the prompt?" Because the **context
window** — the amount of text a model can read at once — is finite. Even a large
window can't hold thousands of papers, and stuffing it full is slow and
expensive, and *hurts* quality (the model gets distracted; see the
"lost-in-the-middle" problem in lesson 10).

**Retrieval is the filter.** Instead of showing the model everything, you show it
the handful of passages that actually matter for *this* question. The corpus can
be enormous; the prompt stays small and focused.

## 3. Hallucination — confident, fluent, wrong

This is the big one. When a model doesn't know something, it doesn't stop — it
**hallucinates**: it generates a plausible-sounding answer that is simply false.
The prose is fluent and confident, which makes hallucinations *harder* to catch
than obvious errors. For a research assistant, a confidently invented citation is
worse than useless.

**Retrieval grounds the answer.** When the real passages are in the prompt and
the model is instructed to answer only from them (lesson 06), it has the facts in
front of it — so it doesn't need to invent them. And because it cites the passages
it used, you can *check*. Grounding + citation turns "trust me" into "here's my
source."

> Grounding reduces hallucination; it doesn't eliminate it. A model can still
> misread a passage, or the retrieval can hand it the wrong passage. That's why we
> *measure* groundedness rather than assume it — see lesson 09.

## Putting it together

| Problem | Bare LLM | With RAG |
|---|---|---|
| New information | Can't know it | Add it to the corpus |
| Too much to read | Doesn't fit in context | Retrieve only what's relevant |
| Doesn't know an answer | Invents one | Answers from sources, or says "not in the corpus" |

Every one of these is a *retrieval* fix, not a *model* fix. That's the insight:
you often don't need a bigger or newer model — you need to give the model you have
the right context. Lesson 02 is how retrieval actually finds "the right context".

Next: **[02 · Embeddings and semantic search](02_embeddings_and_semantic_search.md)**.
