from render_explained import build

build(
    "../src/agent.py",
    "agent.py_explained.pdf",
    "Agentic RAG — a from-scratch Claude tool-use loop over the retriever",
    "This is the methodology track's core. Unlike naive RAG, the agent decides how to retrieve: it "
    "decomposes a question, searches for each part, judges whether it has enough, and searches "
    "again — up to a cap — before answering. The loop is hand-written so every decision is a "
    "visible, logged message.",
    [
        (24, "Lines 1 to 24 are the module docstring. They define what makes this agentic (the model "
             "decides how to retrieve rather than retrieving once), sketch the tool-use loop, "
             "explain why it is built by hand rather than with a framework (explainability), and "
             "state the pay-off it demonstrates: on multi-hop questions a single retrieval misses "
             "one piece of evidence and the agent's second search recovers it."),
        (9, "Lines 25 to 33 are imports: postponed annotations (line 26), dataclass and field for "
            "the result types (line 28), Any for message typing (line 29), config and the llm "
            "wrapper (line 31), extract_cited_ids reused from rag.py (line 32), and Retrieved plus "
            "VectorStore from the shared vector store (line 33) — the same retriever naive RAG uses."),
        (15, "Lines 34 to 48 define SYSTEM_PROMPT, the agent's guidance. The comment (lines 35 to "
             "37) highlights the key instruction; the string (lines 38 to 48) tells the model it has "
             "a search_corpus tool and to decompose multi-part questions into separate focused "
             "searches (which beats one broad query), to answer only from retrieved passages, to "
             "cite arXiv ids, and to say plainly when the corpus cannot answer."),
        (23, "Lines 49 to 71 define SEARCH_TOOL, the single tool the agent may call, as the JSON "
             "schema Claude's tool-use API expects. It has a name (line 53), a description telling "
             "the model to call it multiple times with focused queries (lines 54 to 59), and an "
             "input_schema (lines 60 to 70) declaring one required string parameter, query, with "
             "additionalProperties false so the model cannot invent extra fields. A tight schema "
             "keeps the agent's choices legible."),
        (9, "Lines 72 to 80 define the AgentStep dataclass — one retrieval the agent chose to make. "
            "It holds the query it decided to run (line 78) and the hits that came back (line 79). "
            "The list of these steps is the auditable reasoning trace the notebook can display."),
        (17, "Lines 81 to 97 define the AgentAnswer dataclass — the final result plus the full "
             "trace. Fields: the question and answer (lines 91 to 92), the ordered list of steps "
             "(line 93), the de-duplicated union of all retrieved sources (line 94), the cited ids "
             "(line 95), and stopped_reason (line 96, either answered or max_steps). The three list "
             "fields use default_factory=list so each instance gets its own list."),
        (7, "Lines 98 to 104 define _format_tool_result, which renders one search's hits into the "
            "text handed back to the model. Lines 101 to 102 return a clear no-passages message when "
            "a search finds nothing; line 103 otherwise joins each hit as its bracketed id, title, "
            "and text — the same citable format the prompt uses."),
        (11, "Lines 105 to 115 begin answer(), the agentic loop. The docstring (lines 107 to 114) "
             "describes it as a manual tool-use loop bounded by MAX_AGENT_STEPS with a forced-answer "
             "turn at the cap. Line 115 seeds the message history with the user's question."),
        (3, "Lines 116 to 118 initialise two accumulators: steps (the trace, line 116) and all_hits, "
            "a dict keyed by chunk_id (line 117) that de-duplicates the evidence pool so the same "
            "chunk retrieved by two different searches is stored once."),
        (5, "Lines 119 to 123 open the bounded loop (up to MAX_AGENT_STEPS iterations). The comment "
            "(lines 120 to 122) notes the search tool is always offered inside the loop and the "
            "model itself decides whether to search or answer. Line 123 calls Claude with the "
            "messages, system prompt, and the search tool."),
        (4, "Lines 124 to 127 append the assistant's turn to the history verbatim. The comment "
            "(lines 125 to 126) explains why this matters: the API needs the original tool-use "
            "blocks preserved so the tool results sent next can be matched to them."),
        (2, "Lines 128 to 129 extract any tool_use blocks from the model's response — the searches "
            "it is requesting this turn."),
        (12, "Lines 130 to 141 handle the answer case: if there were no tool calls (line 130) the "
             "agent chose to answer, so it pulls the text (line 132), snapshots the de-duped sources "
             "(line 133), and returns an AgentAnswer with cited_ids computed from the retrieved "
             "sources and stopped_reason set to answered (lines 134 to 141)."),
        (18, "Lines 142 to 159 handle the search case: for each requested search (line 146) it reads "
             "the query (line 147), runs the shared retriever (line 148), records the step (line "
             "149), folds the hits into the de-duped pool (lines 150 to 151), and builds a "
             "tool_result carrying the matching tool_use_id and the formatted hits (lines 152 to "
             "158). Line 159 appends all results as one user message — the API expects one "
             "tool_result per tool_use, batched together."),
        (None, "Lines 160 to 182 are the step-cap backstop, reached only if the model never chose to "
                "answer within the loop. Lines 163 to 171 append a final user message forcing an "
                "answer from the evidence gathered. Line 172 calls Claude once more with tools set "
                "to None (so it must answer, not search). Lines 173 to 181 return an AgentAnswer "
                "just like the normal path, but with stopped_reason set to max_steps so callers can "
                "see the cap was hit rather than a natural stop."),
    ],
)
