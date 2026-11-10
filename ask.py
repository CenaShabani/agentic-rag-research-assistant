"""Ask the research assistant a question from the command line.

A small interactive demo over the built index. Choose the naive single-shot RAG
path or the agentic path, and (for the agent) see its search trace — the queries
it decided to run — so the "agentic" behaviour is visible, not just asserted.

    python ask.py "How does retrieval-augmented generation reduce hallucination?"
    python ask.py --mode naive "What is a vector database?"

Needs ANTHROPIC_API_KEY for the generation step (retrieval is local). If the exact
question was run before, it replays from the committed cache and needs no key.
"""

from __future__ import annotations

import argparse
import sys

from src import agent as agent_mod
from src import rag
from src.llm import LLMKeyMissing
from src.vectorstore import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the arXiv research assistant.")
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument("--mode", choices=["naive", "agentic"], default="agentic",
                        help="Single-shot RAG (naive) or the iterative agent (default).")
    args = parser.parse_args()

    store = VectorStore.load()

    try:
        if args.mode == "naive":
            result = rag.answer(store, args.question)
            print("\n=== Answer (naive RAG) ===\n")
            print(result.answer)
            print("\nSources:", ", ".join(sorted({s.arxiv_id for s in result.sources})))
        else:
            result = agent_mod.answer(store, args.question)
            print("\n=== Agent search trace ===")
            for i, step in enumerate(result.steps, 1):
                ids = ", ".join(h.arxiv_id for h in step.hits)
                print(f"  step {i}: search {step.query!r} -> [{ids}]")
            print(f"  (stopped: {result.stopped_reason})")
            print("\n=== Answer (agentic RAG) ===\n")
            print(result.answer)
            print("\nSources:", ", ".join(sorted({s.arxiv_id for s in result.sources})))
    except LLMKeyMissing as exc:
        print(f"\nNo answer generated: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
