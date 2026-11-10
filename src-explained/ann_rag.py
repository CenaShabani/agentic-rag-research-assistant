from render_explained import build

build(
    "../src/rag.py",
    "rag.py_explained.pdf",
    "Naive single-shot RAG — the baseline the agent must beat",
    "This module implements <b>classic RAG</b>: retrieve the top chunks once, put them in a "
    "grounded prompt, and generate one answer. It is deliberately the simple baseline; its built-in "
    "weakness on multi-hop questions is what motivates the agent in agent.py.",
    [
        (15, "Lines 1 to 15 are the module docstring. They summarise RAG as three moves — retrieve "
             "once, stuff the chunks into a grounded prompt, generate — and name the built-in "
             "weakness: a single retrieval that misses a fact gives the model no second chance, "
             "which is common on multi-hop questions and is exactly what the agent fixes."),
        (7, "Lines 16 to 22 are imports. Line 17 enables postponed annotation evaluation. Line 19 "
            "imports dataclass and field for the small result type defined below. Line 21 imports "
            "the project config and the llm wrapper. Line 22 imports Retrieved (one search hit) and "
            "VectorStore (the shared retriever) from the vector-store module."),
        (15, "Lines 23 to 37 define SYSTEM_PROMPT. The comment on lines 24 to 29 explains the "
             "grounding contract; the string on lines 30 to 37 is the actual instruction sent to "
             "Claude. Its three working rules: answer ONLY from the provided passages (blocks the "
             "model from using memory and hallucinating), cite the arXiv ids used in square brackets "
             "(makes the answer auditable), and say plainly when the passages are insufficient "
             "(makes I-don't-know a valid answer)."),
        (10, "Lines 38 to 47 define the RAGAnswer dataclass. The @dataclass decorator (line 40) "
             "auto-generates the constructor. The fields are the question asked (line 44), the "
             "generated answer text (line 45), the list of Retrieved source chunks (line 46), and "
             "cited_ids (line 47) — which defaults to an empty list via field(default_factory=list) "
             "so each instance gets its own list rather than sharing one."),
        (11, "Lines 48 to 58 define format_context, which renders the retrieved chunks into the text "
             "block placed in the prompt. The docstring (lines 51 to 55) notes each passage is "
             "tagged with its arXiv id so the model can cite it. The return (lines 56 to 58) joins "
             "the hits with blank lines, each formatted as its id in brackets, its title, then its "
             "text — a generator expression iterating over the hits."),
        (10, "Lines 59 to 68 define extract_cited_ids, the citation-honesty check. Line 67 builds a "
             "set of the ids that were actually retrieved. Line 68 returns only those retrieved ids "
             "whose bracketed form appears in the answer text. The docstring (lines 62 to 66) "
             "explains why: a citation to something never retrieved is a hallucinated reference and "
             "must not be counted as a real source."),
        (None, "Lines 69 to 90 define answer(), the end-to-end naive pipeline. It takes the shared "
                "store, the question, and k (defaulting to config.TOP_K). Line 73 retrieves the "
                "top-k chunks. Line 74 formats them into context. Lines 75 to 79 build the user "
                "message: the context, then the question, then a restatement of the instruction. "
                "Lines 80 to 83 call the LLM with that message and the system prompt. Line 84 pulls "
                "the plain text out of the response. Lines 85 to 90 package everything into a "
                "RAGAnswer, computing cited_ids from the final text so the record is self-contained "
                "and auditable."),
    ],
)
