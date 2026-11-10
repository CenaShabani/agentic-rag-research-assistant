# 08 · Agent tools and loops

Lesson 07 said the agent "uses tools in a loop." This lesson is the mechanics —
what a tool actually is, how the loop turns, and how to read
[`src/agent.py`](../src/agent.py) line by line. This is the part interviewers love,
because it's concrete and it's where "agent" stops being a buzzword.

## What a "tool" is

A **tool** is a function you describe to the model in a structured way — a name, a
description of what it does, and a JSON schema for its inputs — and that the model
can then ask to call. The model doesn't run the tool itself; it *emits a request*
to run it ("call `search_corpus` with query='dense passage retrieval'"), your code
runs it, and you hand the result back. The model reads the result and decides what
to do next.

This project has exactly **one** tool, `search_corpus`, defined in
[`src/agent.py`](../src/agent.py):

```
name: "search_corpus"
description: "Search the arXiv paper corpus... call multiple times with
              different focused queries..."
input: { query: string }
```

One tool, one string argument. Kept deliberately tiny so the agent's choices are
legible: every action it takes is "search for *this*," which is easy to log and
reason about.

## The tool-use loop

Here's the loop in [`src/agent.py`](../src/agent.py)'s `answer`, in plain English:

1. **Send** the conversation so far to Claude, offering the `search_corpus` tool.
2. **Look at the reply.** It's one of two things:
   - A **tool-use** request → the agent wants to search. Go to step 3.
   - **Plain text** → the agent has decided to answer. **Done** — return it.
3. **Run the search** it asked for (`VectorStore.search`), record the query and
   hits in the trace, add the hits to the running evidence pool.
4. **Hand the results back** as a `tool_result` message.
5. **Repeat** from step 1. The conversation now includes the search and its
   results, so the model's next decision is informed by what it just learned.

Every turn — the model's tool request, the search results — gets appended to a
running `messages` list, so the model always sees the full history of what it has
searched and found. That accumulating context is what lets it decide "I already
have the retriever half; now I need the generator half."

## The stopping condition — why bounded loops matter

A loop that lets an LLM decide when to stop *needs a backstop*, or a confused agent
could search forever, burning tokens. This one is bounded by `MAX_AGENT_STEPS`
(4, in [`src/config.py`](../src/config.py)). Four rounds is enough for a two- or
three-hop question (decompose → search → search again → answer) without runaway
loops. If the agent hits the cap without answering, we make one final call with the
tool **removed**, forcing it to commit to an answer using the evidence it has. The
result records `stopped_reason` — `"answered"` (it chose to stop) or `"max_steps"`
(the cap forced it) — so you can see which happened.

## Preserving the trace

The whole reason to build the loop by hand: `AgentAnswer.steps` is a list of every
`(query, hits)` the agent chose, in order. That's the **reasoning trace** — run
`ask.py --mode agentic "..."` and you'll see it printed:

```
step 1: search 'dense passage retriever' -> [2004.04906, ...]
step 2: search 'sequence to sequence generation' -> [2005.11401, ...]
(stopped: answered)
```

You can *watch* the agent decompose the question. That visibility is what makes the
system explainable rather than a black box.

## One subtlety: the assistant turn must be preserved verbatim

When the model emits a tool-use request, we append its response **exactly as
received** before adding the tool result. The API needs the original tool-use block
to match it with the result we send back. Reconstructing or trimming it breaks the
turn. It's a small correctness detail with an obvious symptom if you get it wrong
(the API rejects the next call), and the code comments call it out.

## Where this shows up in the repo

- [`src/agent.py`](../src/agent.py) — `SEARCH_TOOL`, the loop in `answer`,
  `AgentStep` / `AgentAnswer`.
- [`tests/test_agent.py`](../tests/test_agent.py) — the loop tested with a *mocked*
  LLM: tool call parsed and run, evidence accumulated across searches, step cap
  enforced, citations restricted to retrieved ids. All keyless.

Next: **[09 · Evaluating RAG](09_evaluating_rag.md)** — how we measure whether any
of this actually works.
