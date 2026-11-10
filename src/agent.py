"""Agentic RAG — a from-scratch Claude tool-use loop over the retriever.

What makes this *agentic* rather than just RAG? Naive RAG (rag.py) retrieves once,
up front, and hopes that single search grabbed everything. This agent instead
*decides* how to retrieve: it can break a question into parts, search for each,
read what came back, judge whether it has enough, and search again — before it
answers. It runs a real Claude tool-use loop:

    Claude, given a `search_corpus` tool, chooses a query
      -> we run the retrieval and hand back the chunks
      -> Claude reads them, decides: enough? if not, search again (new query)
      -> ... up to MAX_AGENT_STEPS times ...
      -> Claude writes the final grounded, cited answer

We implement the loop by hand — not with a framework — because explainability is
a hard requirement here: every decision the agent makes is a visible message we
log, so the notebook can show its reasoning trace. It uses the same VectorStore
as naive RAG (one definition of retrieval in the repo), so any quality difference
between the two is about the *loop*, not about a different search.

The pay-off this is built to demonstrate: on multi-hop questions — the kind that
need two different pieces of evidence — a single retrieval usually misses one of
them, and the agent's second, reformulated search is what recovers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import config, llm
from .rag import extract_cited_ids
from .vectorstore import Retrieved, VectorStore

# The agent's guidance. The key instructions: decompose multi-part questions and
# search for each part separately (this is what beats single-shot retrieval),
# only answer once you have the evidence, ground every claim, and cite arXiv ids.
SYSTEM_PROMPT = (
    "You are a research assistant answering questions about machine-learning "
    "papers. You have a `search_corpus` tool that returns passages from a corpus "
    "of arXiv papers. Strategy: if a question has multiple parts or needs more "
    "than one fact, search for each part separately rather than in one broad "
    "query — a focused query retrieves better. Read what each search returns "
    "before deciding whether you need another. When you have enough evidence, "
    "write the answer using ONLY the retrieved passages; do not use outside "
    "knowledge. Cite the arXiv ids you used in square brackets, e.g. [2401.01234]. "
    "If the corpus cannot answer the question, say so plainly."
)

# The one tool the agent may call. A tight schema — one string query — keeps the
# agent's choices legible and the loop easy to reason about.
SEARCH_TOOL = {
    "name": "search_corpus",
    "description": (
        "Search the arXiv paper corpus for passages relevant to a query. "
        "Returns the most semantically similar passages with their arXiv ids. "
        "Call this multiple times with different focused queries to gather "
        "evidence for different parts of a question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused search query for one fact or sub-question.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


@dataclass
class AgentStep:
    """One retrieval the agent chose to make — the auditable reasoning trace."""

    query: str
    hits: list[Retrieved]


@dataclass
class AgentAnswer:
    """The agent's final answer plus its full search trace.

    `steps` is what makes the agent explainable: every query it decided to run and
    what came back, in order. `sources` is the de-duplicated union of everything
    retrieved across all steps — the evidence pool the answer was grounded in.
    """

    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    sources: list[Retrieved] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    stopped_reason: str = "answered"  # "answered" or "max_steps"


def _format_tool_result(hits: list[Retrieved]) -> str:
    """Render one search's hits into the text handed back to the agent."""
    if not hits:
        return "No passages found for that query."
    return "\n\n".join(f"[{h.arxiv_id}] {h.title}\n{h.text}" for h in hits)


def answer(store: VectorStore, question: str, k: int = config.TOP_K) -> AgentAnswer:
    """Run the agentic loop and return the answer plus its search trace.

    This is a manual Claude tool-use loop: we call the model, and while it keeps
    asking to use the search tool we run the search and feed the results back,
    appending every turn to the running message history. The loop is bounded by
    MAX_AGENT_STEPS so a confused agent can't retrieve forever — after the cap we
    ask it once more, tools disabled, to answer with what it has.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    steps: list[AgentStep] = []
    all_hits: dict[str, Retrieved] = {}   # de-dupe the evidence pool by chunk_id

    for _ in range(config.MAX_AGENT_STEPS):
        # Inside the loop the search tool is always offered; the model itself
        # decides whether to search again or answer. The step cap below is the
        # backstop if it never chooses to stop searching.
        result = llm.call(messages=messages, system=SYSTEM_PROMPT, tools=[SEARCH_TOOL])

        # Record the assistant turn verbatim so the tool-use blocks are preserved
        # for the next request (the API needs them to match the tool results).
        messages.append({"role": "assistant", "content": result["content"]})

        tool_uses = [b for b in result["content"] if b["type"] == "tool_use"]
        if not tool_uses:
            # No tool call -> the agent chose to answer. We're done.
            text = llm.first_text(result)
            sources = list(all_hits.values())
            return AgentAnswer(
                question=question,
                answer=text,
                steps=steps,
                sources=sources,
                cited_ids=extract_cited_ids(text, sources),
                stopped_reason="answered",
            )

        # Run every requested search and return all results in one user message
        # (the API expects one tool_result per tool_use, batched together).
        tool_results = []
        for tu in tool_uses:
            query = tu["input"]["query"]
            hits = store.search(query, k=k)
            steps.append(AgentStep(query=query, hits=hits))
            for h in hits:
                all_hits[h.chunk_id] = h
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": _format_tool_result(hits),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    # Hit the step cap without answering: ask once more with tools off so the
    # model is forced to commit to an answer using the evidence gathered so far.
    messages.append(
        {
            "role": "user",
            "content": (
                "You have gathered enough evidence. Now write the final answer "
                "using only the passages retrieved so far, citing arXiv ids."
            ),
        }
    )
    result = llm.call(messages=messages, system=SYSTEM_PROMPT, tools=None)
    text = llm.first_text(result)
    sources = list(all_hits.values())
    return AgentAnswer(
        question=question,
        answer=text,
        steps=steps,
        sources=sources,
        cited_ids=extract_cited_ids(text, sources),
        stopped_reason="max_steps",
    )
