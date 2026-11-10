"""Naive (single-shot) RAG — the baseline the agent has to beat.

The classic RAG pipeline, in three moves:

    retrieve once  ->  stuff the chunks into a grounded prompt  ->  generate

It retrieves the top-k chunks for the question, hands them to Claude with strict
instructions to answer *only* from those chunks and cite them, and returns the
answer plus the sources. There is no iteration: one retrieval, one generation.

This is the honest baseline. Its weakness is built in — if the single retrieval
misses a fact (common on multi-hop questions that need two different pieces of
evidence), the model never gets a second chance. The agentic version in agent.py
exists precisely to fix that, and the evaluation measures whether it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config, llm
from .vectorstore import Retrieved, VectorStore

# The grounding contract. Two rules do the real work: answer ONLY from the
# provided context (this is what stops the model from falling back on its
# parametric memory and hallucinating), and cite the arXiv ids you used (this is
# what makes the answer auditable). Telling it to say when the context is
# insufficient is what lets "I don't know" be a correct answer instead of a
# confident wrong one.
SYSTEM_PROMPT = (
    "You are a research assistant answering questions about machine-learning "
    "papers. Answer using ONLY the provided context passages. Every claim must be "
    "grounded in the passages — do not use outside knowledge. Cite the arXiv ids "
    "of the passages you used, in square brackets, e.g. [2401.01234]. If the "
    "passages do not contain enough information to answer, say so plainly rather "
    "than guessing."
)


@dataclass
class RAGAnswer:
    """A grounded answer plus everything needed to evaluate and audit it."""

    question: str
    answer: str
    sources: list[Retrieved]
    cited_ids: list[str] = field(default_factory=list)


def format_context(hits: list[Retrieved]) -> str:
    """Render retrieved chunks into a numbered, id-tagged context block.

    Each passage is labelled with its arXiv id so the model can cite it and a
    reader can trace every claim back to a specific chunk.
    """
    return "\n\n".join(
        f"[{h.arxiv_id}] {h.title}\n{h.text}" for h in hits
    )


def extract_cited_ids(answer: str, hits: list[Retrieved]) -> list[str]:
    """Find which retrieved arXiv ids the answer actually cited.

    We only count ids that were in the retrieved set — a "citation" of something
    that wasn't retrieved is a hallucinated reference, not a real source.
    """
    retrieved_ids = {h.arxiv_id for h in hits}
    return [aid for aid in retrieved_ids if f"[{aid}]" in answer]


def answer(store: VectorStore, question: str, k: int = config.TOP_K) -> RAGAnswer:
    """Run naive RAG end to end and return a grounded, cited answer."""
    hits = store.search(question, k=k)
    context = format_context(hits)
    user = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above, and cite the arXiv ids you used."
    )
    result = llm.call(
        messages=[{"role": "user", "content": user}],
        system=SYSTEM_PROMPT,
    )
    text = llm.first_text(result)
    return RAGAnswer(
        question=question,
        answer=text,
        sources=hits,
        cited_ids=extract_cited_ids(text, hits),
    )
