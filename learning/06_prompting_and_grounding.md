# 06 · Prompting and grounding

Retrieval found the right passages. Now the LLM has to answer *from* them — not
from its own memory, not from vibes. The mechanism is the **prompt**: how you
assemble the retrieved passages and the question into the text you send the model.
Get this wrong and all the retrieval work is wasted, because the model will happily
ignore the passages and hallucinate anyway.

## What "grounding" means

**Grounding** = binding the answer to the evidence. A grounded answer makes only
claims that the provided passages support. An *un*grounded answer wanders off into
the model's parametric memory — which is exactly the hallucination we ran
retrieval to avoid. Grounding is a *prompting* discipline, enforced by three
instructions.

## The three instructions that do the work

Look at `SYSTEM_PROMPT` in [`src/rag.py`](../src/rag.py). Three sentences carry
almost all the weight:

1. **"Answer using ONLY the provided context passages."** This is the whole ball
   game. It tells the model its job is reading comprehension over the passages,
   not recall. Without this line the model treats the passages as optional hints
   and falls back on memory.

2. **"Cite the arXiv ids of the passages you used."** Citation makes the answer
   *auditable*. A reader — or the evaluation in lesson 09 — can trace each claim to
   a source and check it. It also nudges the model to actually use the passages,
   because it has to name which ones.

3. **"If the passages do not contain enough information, say so plainly."** This is
   the one people forget, and it's crucial. It makes **"I don't know"** a correct,
   allowed answer. Without it, a model handed irrelevant passages will strain to
   produce *some* answer — and that's precisely when it invents things. Permission
   to abstain is a hallucination guardrail.

## Assembling the context

The retrieved chunks are formatted into a labelled block — each passage tagged
with its arXiv id — so the model can cite by id and a reader can map every citation
back to a specific chunk (`format_context` in [`src/rag.py`](../src/rag.py)):

```
[2005.11401] Retrieval-Augmented Generation...
<the chunk text>

[2004.04906] Dense Passage Retrieval...
<the chunk text>
```

Then the user turn is simply: here are the passages, here is the question, answer
from the passages and cite. Structure matters — clearly separating *evidence* from
*question* helps the model keep them straight.

## Verifying the citations are real

The model *says* it cited `[2005.11401]`. Did it cite something that was actually
retrieved, or did it hallucinate a reference? `extract_cited_ids` checks: it only
counts a citation if that id was in the retrieved set. A "citation" to a paper that
was never retrieved is a fabricated reference, not a source — and we don't credit
it. This is a small but important honesty check, and it's reused by both the plain
and agentic paths.

## Grounding reduces hallucination — it doesn't abolish it

Even a perfectly grounded prompt can go wrong: the model can misread a passage, or
retrieval can hand it a plausible-but-wrong passage and it dutifully grounds on the
wrong thing. That's why lesson 09 *measures* groundedness with an independent judge
rather than trusting the instruction to have worked. Prompting stacks the odds;
evaluation checks the outcome.

## Where this shows up in the repo

- [`src/rag.py`](../src/rag.py) — `SYSTEM_PROMPT`, `format_context`,
  `extract_cited_ids`, and the single-shot `answer`.
- The same grounding contract is repeated (with search-strategy guidance added) in
  the agent's system prompt in [`src/agent.py`](../src/agent.py).

Next: **[07 · What makes it agentic](07_what_makes_it_agentic.md)** — the step up
from "retrieve once" to "decide how to retrieve".
