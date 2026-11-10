# `src/agent.py` — the tool-use loop

## What it's for

The system the project is named after. Instead of retrieving once and hoping, the agent
*decides* how to retrieve: decompose the question, search for a part, read what came back, judge
whether it is enough, search again — then answer.

It is a Claude tool-use loop written by hand. **No framework, deliberately.** Explainability was
a hard requirement: every query the agent chooses is a message we log and can show. LangChain
would have hidden precisely the thing the project is about.

It uses the same `VectorStore` as `rag.py`, so any measured difference between the two is about
the loop, never about a different search.

## The public surface

| Member | In | Out |
|---|---|---|
| `answer(store, question, k=TOP_K)` | `VectorStore`, `str` | `AgentAnswer` |
| `SEARCH_TOOL` | — | the tool schema handed to the model |

`AgentAnswer` = `question`, `answer`, `steps` (`list[AgentStep]` — the trace), `sources`
(de-duplicated evidence pool), `cited_ids`, and `stopped_reason` ∈ `{"answered", "max_steps"}`.

`AgentStep` is just `query` + `hits`: what the agent decided to search for and what came back.
That list *is* the reasoning trace, and it is why `ask.py --mode agentic` can show the
decomposition live.

## The walkthrough

Three pieces of state before the loop:

```python
messages = [{"role": "user", "content": question}]
steps = []
all_hits: dict[str, Retrieved] = {}   # de-dupe by chunk_id
```

`all_hits` being a **dict keyed by `chunk_id`** is the quiet correctness detail. Across several
searches the same chunk will often come back more than once — two focused queries about related
sub-questions land on overlapping text. Collecting into a list would double-count that chunk in
the evidence pool and inflate every downstream count of "how much evidence did the agent have".
The dict makes the union set-like, and the last write wins harmlessly since the value is identical.

Each iteration calls the model **with the tool always offered**. The model, not the code, decides
whether to search again or answer. That is what makes this agentic rather than a scripted
two-pass retrieval — the control flow is the model's to choose, and the step cap is only a
backstop.

```python
messages.append({"role": "assistant", "content": result["content"]})
```

The assistant turn is appended **verbatim**, not summarised or re-rendered. The API requires the
`tool_use` blocks to be present and to match the `tool_result` blocks that answer them by id. Any
tidying here produces an API error on the next turn — this is the single most common way a
hand-written tool loop breaks.

Then the branch: no `tool_use` blocks means the agent chose to answer, and the function returns
with `stopped_reason="answered"`. Otherwise every requested search runs, and all results go back
in **one** user message:

```python
tool_results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": ...})
```

One `tool_result` per `tool_use`, batched into a single message, each matched by `tool_use_id`.
The model can request several searches in one turn, and answering them across separate messages —
or omitting one — is an API error, not a degraded answer.

**The exit path after the loop is the part worth understanding.** If the agent hits
`MAX_AGENT_STEPS` (4) without ever choosing to answer, the code appends a plain instruction to
answer now and calls the model **with `tools=None`**. Removing the tool is what forces commitment:
left available, a model that is stuck in a search loop will simply search again. Setting
`stopped_reason="max_steps"` records that this happened, so the evaluation can tell a confident
answer apart from one squeezed out at the cap.

## The non-obvious bits

- **`SEARCH_TOOL`'s schema takes exactly one string.** A tight schema keeps the agent's choices
  legible and the trace readable; a rich schema (filters, k, date ranges) would give the model
  more ways to be subtly wrong and make the trace hard to compare across questions.
- **The tool *description* is doing prompt engineering.** "Call this multiple times with
  different focused queries" is where the decomposition behaviour is actually encouraged — the
  system prompt repeats it, but the tool description is what the model reads at the decision
  point.
- **`_format_tool_result` returns "No passages found for that query."** for an empty result
  rather than an empty string. An empty tool result reads as a malfunction and derails the model;
  a plain sentence is information it can act on by reformulating.
- **The tool result uses the same `[id] title\ntext` format as `rag.format_context`**, so both
  paths present evidence identically. One more thing held constant so the comparison is about
  strategy.
- **`extract_cited_ids` is imported from `rag.py`**, not redefined — the citation rule is defined
  once for both paths.

## How it can break

- **Needs an API key** (`llm.call`), same as `rag.py`.
- **A model that never stops searching** is bounded by `MAX_AGENT_STEPS`; the cap is the reason
  a confused agent cannot run up an unbounded bill.
- **Cost scales with steps.** Each iteration resends the entire message history including every
  retrieved passage, so token cost grows roughly quadratically in steps. This is exactly why the
  evaluation reports the agent's retrieval count alongside quality — the gain has to be weighed
  against it.
- **Retrieved text is fed to the model as tool results.** The corpus is arXiv abstracts, so this
  is low-risk, but it is an untrusted-input path and the project says so plainly rather than
  claiming injection hardening it does not have.
- Tests mock `llm.call` to script a tool-use turn followed by an answer turn, so the loop's
  message assembly is pinned without a key.
