# LangGraph branch

A branch of the parent **Agentic RAG Research Assistant** that rebuilds the *same
project* — question-answering over the same 2,012 arXiv papers, using the same
`../src/` retriever — but orchestrates the agent as an explicit **LangGraph**
`StateGraph` instead of the hand-written loop in [`../src/agent.py`](../src/agent.py).

It exists to follow the *"LangGraph Complete Course For Beginners (From ZERO To
HERO)"* crash course and apply every concept to our own project rather than a toy.

## What's here

- **[`agentic_rag_langgraph_zero_to_hero.ipynb`](agentic_rag_langgraph_zero_to_hero.ipynb)**
  — one comprehensive, self-contained notebook that teaches LangGraph from zero
  (state → nodes → edges → conditional edges → tools → the ReAct loop) and, with
  those pieces, rebuilds our agentic RAG as a graph you can draw, stream, and
  extend. It runs top-to-bottom **with or without an API key** — retrieval and the
  graph mechanics execute locally; the LLM nodes fall back to a deterministic stub
  when `ANTHROPIC_API_KEY` is unset, so the whole graph still completes.

## How it relates to the parent

Same corpus, same `VectorStore.search`, same finding (single-shot retrieval misses
multi-hop evidence; the iterative agent recovers it). The only thing that changes is
the *spine*: a compiled `StateGraph` with a conditional edge for "search again?"
instead of a Python `while` loop. The notebook's section 8 maps every piece of
`src/agent.py` to its LangGraph equivalent.

## Run it

```bash
cd ..                       # project root
python run_pipeline.py      # build the index once (local, no key)
pip install langgraph langchain-anthropic langchain-core
# open langgraph-branch/agentic_rag_langgraph_zero_to_hero.ipynb and Run All
# (set ANTHROPIC_API_KEY first for real generation instead of the fallback stub)
```

These extra dependencies belong to this branch only; the parent project does not
need LangGraph.
