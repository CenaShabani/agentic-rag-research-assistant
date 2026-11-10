# `src/rag.py` — the naive baseline

## What it's for

Classic single-shot RAG: retrieve once, stuff the chunks into a grounded prompt, generate once.
It is the **baseline the agent has to beat**, and its weakness is designed in rather than
accidental — one retrieval, one chance.

This module is not a lesser version of `agent.py`. It is the control condition. The project's
finding only means something because this exists and is implemented honestly, using the same
retriever the agent uses.

## The public surface

| Function | In | Out |
|---|---|---|
| `answer(store, question, k=TOP_K)` | `VectorStore`, `str` | `RAGAnswer` |
| `format_context(hits)` | `list[Retrieved]` | `str` — numbered, id-tagged passage block |
| `extract_cited_ids(answer, hits)` | `str`, `list[Retrieved]` | `list[str]` — ids actually cited |

`RAGAnswer` = `question`, `answer`, `sources` (the retrieved chunks), `cited_ids`.

`extract_cited_ids` is also imported by `agent.py` — the citation rule is defined once and both
paths obey it, the same single-definition discipline `vectorstore.py` applies to search.

## The walkthrough

**`SYSTEM_PROMPT` is the substance of this file.** Three instructions, each doing a specific job:

1. *Answer using ONLY the provided context.* This is what stops the model falling back on its
   parametric memory. Without it the model answers ML questions perfectly well from training
   knowledge — and then the retrieval quality is unmeasurable, because a good answer no longer
   implies a good retrieval. This one line is what makes the whole evaluation valid.
2. *Cite the arXiv ids in square brackets.* Makes the answer auditable, and gives
   `extract_cited_ids` something to parse.
3. *Say so plainly if the passages are insufficient.* Makes "I don't know" an available correct
   answer instead of forcing a confident wrong one. A model with no permission to fail will
   fabricate.

`format_context` renders each hit as `[arxiv_id] title` then the text. The id goes first and in
brackets because that is exactly the token the model is asked to reproduce when citing — the
prompt and the format agree, which is why citation works reliably.

**`extract_cited_ids` only counts ids that were actually retrieved:**

```python
retrieved_ids = {h.arxiv_id for h in hits}
return [aid for aid in retrieved_ids if f"[{aid}]" in answer]
```

A citation of something not in the retrieved set is a *hallucinated reference*, not a source, and
counting it would inflate every groundedness measurement. Filtering against the retrieved set is
what makes the metric honest. Note the direction of the loop: it iterates over retrieved ids and
tests membership in the answer text, rather than regexing brackets out of the answer — so a
fabricated `[9999.99999]` cannot enter the result at all.

`answer` is then four steps with no cleverness: search, format, call, extract. Deliberately
boring. A baseline that has been quietly optimised is not a baseline.

## The non-obvious bits

- **`k` defaults to `config.TOP_K`, and the evaluation passes the same k to both paths.** If the
  agent got a different k, the comparison would measure budget, not strategy.
- **No retry, no reranking, no query rewriting.** All would improve it; all would also stop it
  being the naive baseline the literature means by "RAG".
- **`sources` is the full retrieved set, not just the cited ones.** Evaluation needs to know what
  was *available* to the model, not only what it used — that is how you tell a retrieval failure
  apart from a generation failure.

## How it can break

- **Needs an API key.** `llm.call` raises `LLMKeyMissing` without one; this is the half of the
  project that is key-gated, and the reason the retrieval evaluation was built to run separately
  and offline.
- **A model that ignores the grounding instruction** would silently break the measurement rather
  than error. The `cited_ids` check is the tripwire: an answer with no cited ids is either a
  genuine "insufficient evidence" or an ungrounded answer, and those need to be told apart.
- **Long contexts get truncated by the model, not by us.** With `TOP_K` chunks of ~1000 chars
  this is nowhere near the limit — but raising `TOP_K` far enough would silently drop passages
  from the end of the context, and the "lost in the middle" effect degrades the ones that remain.
- Tests mock the LLM, so the prompt assembly and citation extraction are pinned without a key.
