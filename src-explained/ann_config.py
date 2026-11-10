from render_explained import build

build(
    "../src/config.py",
    "config.py_explained.pdf",
    "Every tuned constant in one place, each with a reason",
    "This module is the project's control panel. Every judgement parameter — a path, a model name, "
    "a chunk size, a loop cap — is a named constant here with a prose justification, so no "
    "answer-changing number is ever a magic value buried in a function.",
    [
        (9, "Lines 1 to 9 are the module docstring, stating the governing rule: if a number changes "
            "the answer it must be visible and defensible here, and every entry point (pipeline, "
            "eval, notebooks, tests) reads its settings from this one module."),
        (5, "Lines 10 to 14 import os (to read the API-key environment variable) and Path (for "
            "building filesystem paths), after enabling postponed annotations on line 11."),
        (8, "Lines 15 to 22 define SEED = 42. The comment explains it makes every stochastic step — "
            "embedding, sampling, shuffling — reproducible, matching the portfolio-wide seed "
            "convention."),
        (14, "Lines 23 to 36 define the directory paths. Line 28 anchors PROJECT_ROOT to the folder "
             "two levels above this file, so paths resolve identically from a notebook, a test, or "
             "the command line. Lines 29 to 36 derive the data, raw, processed, index, outputs, "
             "figures, results, and eval directories from that root using the path-join operator."),
        (8, "Lines 37 to 44 name the individual artifact files — provenance JSON, the papers JSONL, "
            "the chunks parquet, the FAISS index, and the row-aligned chunk-metadata parquet — so a "
            "rename happens in exactly one place. The inline comments document each file's shape."),
        (9, "Lines 45 to 53 set ARXIV_CATEGORIES to cs.CL and cs.LG. The comment explains these are "
            "computational linguistics and machine learning — the categories most likely to contain "
            "the retrieval, embedding, and agent literature the evaluation asks about, so the corpus "
            "can actually answer the questions."),
        (6, "Lines 54 to 59 set MAX_PAPERS = 2000. The comment justifies the bound: the full "
            "pipeline must finish in minutes on a CPU so the capstone can run top-to-bottom, and a "
            "few thousand abstracts is enough to make retrieval quality visible."),
        (6, "Lines 60 to 65 set ARXIV_PAGE_SIZE = 100 and ARXIV_REQUEST_DELAY_SECONDS = 3.0. The "
            "comment notes these match arXiv's published rate guidance; being greedier risks a "
            "temporary IP block that would hurt reproducibility."),
        (24, "Lines 66 to 89 define SEED_PAPER_IDS, a tuple of twelve seminal papers fetched by id "
             "regardless of the recent slice. The comment (lines 67 to 75) gives two reasons: the "
             "evaluation's ground-truth answers point at these papers so they must always be "
             "present (making the metrics deterministic), and they are the foundational works the "
             "field is about. Each id (lines 77 to 88) carries a comment naming the paper — the "
             "Transformer, RAG, BERT, GPT-3, Sentence-BERT, DPR, ReAct, chain-of-thought, word2vec, "
             "Bahdanau attention, InstructGPT, and LLaMA."),
        (10, "Lines 90 to 99 set CHUNK_SIZE_CHARS = 1000. The comment explains the trade-off: "
             "too-large chunks dilute the embedding across several ideas, too-small ones lose "
             "context, and about 1000 characters is one coherent idea for scientific text."),
        (6, "Lines 100 to 105 set CHUNK_OVERLAP_CHARS = 150. The comment explains overlap (~15%) "
            "carries context across a boundary so a fact split between two chunks survives in at "
            "least one, without ballooning the index with near-duplicate text."),
        (11, "Lines 106 to 116 configure embeddings: EMBED_MODEL is all-MiniLM-L6-v2 (line 114), "
             "EMBED_DIM is 384 (line 115), and EMBED_BATCH_SIZE is 64 (line 116). The comment "
             "justifies the model as a fast, CPU-friendly, short-text sentence-transformer that "
             "keeps the retrieval path reproducible without a GPU."),
        (9, "Lines 117 to 125 set TOP_K = 5. The comment explains 5 is enough context for the LLM "
            "to synthesize an answer but few enough that an irrelevant chunk cannot crowd out the "
            "relevant ones (the lost-in-the-middle problem); the agent can retrieve multiple times, "
            "so this is the width of a single retrieval."),
        (9, "Lines 126 to 134 set GEN_MODEL to claude-sonnet-5. The comment frames it as the "
            "cost/quality sweet spot for grounded question-answering and notes it can be swapped to "
            "Opus for maximum quality or Haiku to cut cost, in this one place."),
        (5, "Lines 135 to 139 set JUDGE_MODEL, the LLM-as-judge model for answer evaluation, to the "
            "same model as generation so the grader is at least as capable as what it grades."),
        (4, "Lines 140 to 143 set MAX_ANSWER_TOKENS = 1024 — generous enough for a cited "
            "multi-paragraph answer but bounded so a runaway generation cannot rack up cost."),
        (7, "Lines 144 to 150 set MAX_AGENT_STEPS = 4, the agent loop's stopping condition. The "
            "comment explains 4 rounds is enough for a two- or three-hop question (decompose, "
            "search, search again, answer) without letting a confused agent loop forever."),
        (7, "Lines 151 to 157 set LLM_CACHE_FILE. The comment explains every Claude call is cached "
            "by an input hash so the notebook and tests reproduce published numbers without a key, "
            "and re-runs never pay twice for identical calls."),
        (5, "Lines 158 to 162 set ANTHROPIC_API_KEY_ENV to the name of the environment variable "
            "holding the key. The comment stresses never hard-coding or committing a key, and that "
            "an unset key makes the project fall back to the cache."),
        (10, "Lines 163 to 172 define api_key_available(), returning True when the key environment "
             "variable is set (line 172 wraps os.environ.get in bool, so an empty value counts as "
             "absent). The docstring explains this gate is what lets live calls be optional and the "
             "project reproduce on a machine with no key."),
        (None, "Lines 173 to 182 define ensure_dirs(), which creates every directory the project "
                "writes to. The loop (lines 181 to 182) calls mkdir with parents=True and "
                "exist_ok=True on the raw, processed, index, figures, and results directories, so a "
                "fresh checkout (where data/ and outputs/ are git-ignored and absent) just works."),
    ],
)
