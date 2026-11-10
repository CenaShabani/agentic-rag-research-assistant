"""Tests for the agentic loop, with the LLM fully mocked.

The whole point: verify the *loop* — that a tool call is parsed and run, that the
retrieved evidence accumulates, that it cites only retrieved ids, and that the
step cap is enforced — without any API key or network. We script the "LLM" to
return exactly the tool-use / text turns we want and assert the agent reacts
correctly.
"""

from __future__ import annotations

import pandas as pd

from src import agent as agent_mod
from src import config
from src.vectorstore import VectorStore


def _store(toy_embed) -> VectorStore:
    chunks = pd.DataFrame(
        {
            "chunk_id": ["a:0", "b:0"],
            "arxiv_id": ["1111.0001", "2222.0002"],
            "title": ["Transformers", "Retrieval"],
            "text": [
                "the transformer uses self attention over tokens",
                "retrieval augmented generation fetches passages before generating",
            ],
        }
    )
    return VectorStore.build(chunks)


def test_agent_runs_tool_then_answers(monkeypatch, toy_embeddings):
    """One search, then an answer: the happy path."""
    store = _store(toy_embeddings)

    turns = [
        # First call: the agent decides to search.
        {
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "search_corpus",
                 "input": {"query": "self attention transformer tokens"}},
            ],
        },
        # Second call: the agent answers, citing the retrieved id.
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "The transformer uses attention [1111.0001]."}],
        },
    ]
    calls = iter(turns)
    monkeypatch.setattr(agent_mod.llm, "call", lambda **kwargs: next(calls))

    result = agent_mod.answer(store, "How does the transformer work?")

    assert len(result.steps) == 1
    assert result.steps[0].query == "self attention transformer tokens"
    assert result.stopped_reason == "answered"
    assert "1111.0001" in result.cited_ids
    assert result.sources  # evidence pool populated


def test_agent_accumulates_evidence_over_multiple_searches(monkeypatch, toy_embeddings):
    """Two searches for a multi-part question, then answer — evidence unions."""
    store = _store(toy_embeddings)

    turns = [
        {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t1", "name": "search_corpus",
             "input": {"query": "transformer attention"}}]},
        {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t2", "name": "search_corpus",
             "input": {"query": "retrieval augmented generation passages"}}]},
        {"stop_reason": "end_turn", "content": [
            {"type": "text", "text": "Combined answer [1111.0001] [2222.0002]."}]},
    ]
    calls = iter(turns)
    monkeypatch.setattr(agent_mod.llm, "call", lambda **kwargs: next(calls))

    result = agent_mod.answer(store, "How do transformers and RAG relate?")

    assert len(result.steps) == 2
    ids = {s.arxiv_id for s in result.sources}
    assert {"1111.0001", "2222.0002"} <= ids
    assert set(result.cited_ids) == {"1111.0001", "2222.0002"}


def test_agent_respects_step_cap(monkeypatch, toy_embeddings):
    """An agent that never stops searching is forced to answer at the cap."""
    store = _store(toy_embeddings)

    def always_search(**kwargs):
        # If tools are offered, keep searching; once tools are off (the forced
        # final turn), answer.
        if kwargs.get("tools"):
            return {"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "id": "t", "name": "search_corpus",
                 "input": {"query": "again"}}]}
        return {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Forced answer."}]}

    monkeypatch.setattr(agent_mod.llm, "call", always_search)

    result = agent_mod.answer(store, "loops forever?")

    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == config.MAX_AGENT_STEPS


def test_agent_cites_only_retrieved_ids(monkeypatch, toy_embeddings):
    """A hallucinated citation to an unretrieved id is not counted as a source."""
    store = _store(toy_embeddings)
    turns = [
        {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t1", "name": "search_corpus",
             "input": {"query": "transformer"}}]},
        {"stop_reason": "end_turn", "content": [
            {"type": "text", "text": "Answer citing [9999.9999] which was never retrieved."}]},
    ]
    calls = iter(turns)
    monkeypatch.setattr(agent_mod.llm, "call", lambda **kwargs: next(calls))

    result = agent_mod.answer(store, "q")
    assert "9999.9999" not in result.cited_ids
