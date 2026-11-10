from render_explained import build

build(
    "../src/evaluation.py",
    "evaluation.py_explained.pdf",
    "Does the agent beat naive RAG — and by how much?",
    "This module is the methodology that carries the project. It measures retrieval and answers "
    "separately: retrieval metrics run fully locally, and answer quality is graded by an LLM judge "
    "(key-gated and cached), comparing naive RAG against the agent, sliced by single-hop vs "
    "multi-hop.",
    [
        (21, "Lines 1 to 21 are the module docstring. They lay out the two halves — local retrieval "
             "metrics (hit@k, recall@k, MRR against known-relevant papers) and LLM-as-judge answer "
             "grading (groundedness and correctness) — and state the design goal: a fair head-to-"
             "head where the only thing that differs between the two systems is single-shot versus "
             "iterative retrieval."),
        (9, "Lines 22 to 30 are imports: json for reading the question set and parsing judge output "
            "(line 25), dataclass (line 26), the agent module aliased as agent_mod (line 28), config "
            "plus the llm wrapper and rag module (line 29), and VectorStore (line 30)."),
        (21, "Lines 31 to 51 define the EvalQuestion dataclass — one graded question. The docstring "
             "(lines 38 to 44) explains relevant_ids is the ground truth (the arXiv ids that "
             "genuinely answer it) and hop tags it single or multi, with the hypothesis that the "
             "agent's edge appears on multi-hop rows. Fields are qid, question, relevant_ids, and "
             "hop (lines 46 to 49)."),
        (5, "Lines 52 to 56 define load_eval_set, which reads eval/qa_set.json (line 54), JSON-loads "
            "it (line 55), and constructs one EvalQuestion per entry by unpacking each dict (line "
            "56)."),
        (9, "Lines 57 to 65 define hit_at_k. Line 64 returns 1.0 if the intersection of retrieved "
            "and relevant ids is non-empty, else 0.0 — a blunt did-retrieval-find-anything measure. "
            "Lines 59 to 61 are the section header for the local retrieval metrics."),
        (8, "Lines 66 to 73 define recall_at_k. Line 69 to 70 guard against an empty relevant set "
            "(returning 0.0). Line 71 computes the intersection; line 72 returns its size over the "
            "number of distinct relevant ids — the fraction of needed papers that were retrieved, "
            "the metric that exposes multi-hop weakness."),
        (15, "Lines 74 to 88 define mrr (mean reciprocal rank). The docstring (lines 76 to 81) "
             "explains ranks are 1-based so the top slot scores 1.0, the second 0.5, and so on. "
             "Lines 82 to 85 walk the retrieved ids in order and return one over the rank of the "
             "first relevant one; line 86 returns 0.0 if none is relevant. It rewards ranking the "
             "right paper high."),
        (11, "Lines 89 to 99 define retrieval_ids, which turns a search into an ordered list of "
             "distinct paper ids. The docstring (lines 90 to 94) explains the de-duplication to "
             "paper level: the ground truth is per-paper, so two chunks from one paper count as one "
             "hit. Lines 95 to 99 iterate the search hits and append each arxiv_id only the first "
             "time it appears, preserving rank order."),
        (18, "Lines 100 to 117 define evaluate_retrieval, which computes all three metrics for every "
             "question — fully local, no key. For each question (line 106) it gets the retrieved "
             "paper ids (line 107) and appends a row with the qid, hop, and the three metric values "
             "(lines 108 to 116). Line 117 returns the rows, ready to aggregate or plot."),
        (16, "Lines 118 to 133 define JUDGE_SYSTEM, the system prompt for the LLM judge. It "
             "instructs the model to grade two axes on a 1-to-5 scale — groundedness (are all claims "
             "supported by the passages) and correctness (does it accurately answer) — and to reply "
             "with ONLY a JSON object of the two integer scores, which keeps the output "
             "machine-parseable."),
        (30, "Lines 134 to 163 define judge_answer, which grades one answer. Lines 142 to 147 build "
             "the user message containing the question, the source passages, and the answer. Lines "
             "148 to 152 call the LLM with the judge system prompt and judge model (cached and "
             "key-gated like every call). Line 153 pulls the text; lines 154 to 159 parse the JSON "
             "and return the two integer scores. Lines 160 to 163 catch a malformed judge reply and "
             "score it 0 — visible and honest, rather than silently dropping the row."),
        (None, "Lines 164 to 197 define evaluate_answers, the head-to-head harness. For each "
                "question (line 174) it runs naive RAG (line 175) and the agent (line 176) on the "
                "same store, formats each system's sources (lines 178 to 179), and appends two rows "
                "— one per system. The naive row records one retrieval (lines 181 to 188); the agent "
                "row records its actual step count as n_retrievals (lines 189 to 195), which is the "
                "cost side of the trade. Each row merges in the judge's scores via dict unpacking. "
                "Line 197 returns all rows, giving one row per question-and-system for slicing by "
                "hop type in the notebook."),
    ],
)
