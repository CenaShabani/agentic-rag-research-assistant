"""Central configuration for the agentic RAG research assistant.

Every judgement parameter in this project is a named constant here, each with a
prose "why". The rule (borrowed from the rest of this portfolio): if a number
changes the answer, it must be visible and defensible — never a magic number
buried in a function. `run_pipeline.py`, `run_eval.py`, the notebooks, and the
tests all read their settings from this one module, so there is a single source
of truth for how the system is tuned.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
# One seed for the whole project. Embedding a query, sampling papers, shuffling
# an evaluation set — all of it is reproducible so every published number can be
# regenerated exactly. Matches the SEED = 42 convention used across the portfolio.
SEED = 42

# --------------------------------------------------------------------------- #
# Paths — everything is resolved relative to the project root, so the code runs
# the same whether invoked from a notebook, a test, or the command line.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"
EVAL_DIR = PROJECT_ROOT / "eval"

# Named artifact files. Referenced by the pipeline and every consumer so a
# rename happens in exactly one place.
PROVENANCE_FILE = RAW_DIR / "provenance.json"
PAPERS_FILE = RAW_DIR / "papers.jsonl"          # one arXiv paper per line
CHUNKS_FILE = PROCESSED_DIR / "chunks.parquet"  # one chunk per row
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"    # the serialized vector index
CHUNK_META_FILE = INDEX_DIR / "chunk_meta.parquet"  # row-aligned with the index

# The retrieval benchmark's own artifacts. The generated question set is
# committed (it is small, and every published number depends on it); the drift
# slice is raw downloaded data and is git-ignored like the rest of data/.
KNOWN_ITEM_FILE = EVAL_DIR / "known_item_set.json"
DRIFT_PAPERS_FILE = RAW_DIR / "papers_drift.jsonl"
DRIFT_PROVENANCE_FILE = RAW_DIR / "provenance_drift.json"
ARM_ROWS_FILE = RESULTS_DIR / "arm_comparison.json"   # one row per (arm, query)
ARM_SUMMARY_FILE = RESULTS_DIR / "arm_summary.json"   # per-arm means + intervals
SIGNIFICANCE_FILE = RESULTS_DIR / "arm_significance.json"  # pairwise paired tests
K_SWEEP_FILE = RESULTS_DIR / "k_sweep.json"           # nDCG/recall vs cutoff
CHUNK_ABLATION_FILE = RESULTS_DIR / "chunk_ablation.json"  # quality vs chunk size
ABLATION_SIGNIFICANCE_FILE = RESULTS_DIR / "chunk_ablation_significance.json"
DRIFT_RESULTS_FILE = RESULTS_DIR / "drift.json"

# --------------------------------------------------------------------------- #
# Corpus acquisition (arXiv)
# --------------------------------------------------------------------------- #
# We retrieve recent NLP / ML papers. cs.CL is computational linguistics and
# cs.LG is machine learning — the two categories most likely to contain the
# retrieval, embedding, and agent literature our evaluation questions ask about,
# so the corpus actually *can* answer them.
ARXIV_CATEGORIES = ("cs.CL", "cs.LG")

# How many papers to pull. Bounded deliberately: the whole pipeline (download →
# chunk → embed → index) must finish in a few minutes on a CPU laptop so the
# capstone notebook can run top-to-bottom. A few thousand abstracts is plenty to
# make retrieval quality visible without a GPU or an overnight job.
MAX_PAPERS = 2000

# arXiv asks clients to page politely and sleep between requests. These match
# their published guidance; raising the page size or dropping the delay risks a
# temporary IP block, which would make the pipeline *less* reproducible.
ARXIV_PAGE_SIZE = 100
ARXIV_REQUEST_DELAY_SECONDS = 3.0

# Seminal papers, fetched by id and merged into the corpus regardless of the
# recent slice. Two reasons this matters:
#   1. The evaluation set's ground-truth answers point at these papers, so they
#      must be present for the metrics to be meaningful. Pinning them by id makes
#      the evaluation deterministic — it does not depend on which recent papers
#      happened to be returned that day.
#   2. They are the papers the retrieval/embedding/agent literature is *about*,
#      so a corpus that contains them can actually answer questions about the
#      field's foundations.
SEED_PAPER_IDS = (
    "1706.03762",  # Attention Is All You Need (the Transformer)
    "2005.11401",  # Retrieval-Augmented Generation (the original RAG paper)
    "1810.04805",  # BERT
    "2005.14165",  # GPT-3 (Language Models are Few-Shot Learners)
    "1908.10084",  # Sentence-BERT (the embedding model family we use)
    "2004.04906",  # Dense Passage Retrieval (DPR)
    "2210.03629",  # ReAct (reason + act — the agentic-loop idea)
    "2201.11903",  # Chain-of-Thought prompting
    "1301.3781",   # word2vec (Efficient Estimation of Word Representations)
    "1409.0473",   # Bahdanau attention (align-and-translate)
    "2203.02155",  # InstructGPT (training LMs to follow instructions with RLHF)
    "2302.13971",  # LLaMA
)

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
# Retrieval works on *chunks*, not whole documents. Too-large chunks dilute the
# embedding (one vector trying to represent several ideas); too-small chunks lose
# the context a sentence needs to be meaningful. ~1000 characters is a few
# sentences — one coherent idea — which is a well-trodden default for abstract-
# and-paragraph-length scientific text.
CHUNK_SIZE_CHARS = 1000

# Overlap carries a little context across the boundary so a fact split across two
# chunks is still retrievable from at least one of them. 150 chars (~15%) is the
# usual trade-off: enough to avoid orphaning a sentence, small enough not to
# balloon the index with near-duplicate text.
CHUNK_OVERLAP_CHARS = 150

# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
# all-MiniLM-L6-v2 is the workhorse sentence-transformer: 384-dimensional, fast
# on CPU, and strong on short-text semantic similarity — exactly this task. It is
# small enough to download and run in the capstone without a GPU, which keeps the
# whole retrieval path reproducible on any machine.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
EMBED_BATCH_SIZE = 64

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
# How many chunks a single retrieval call returns. 5 is a common default: enough
# context for the LLM to synthesize an answer, few enough that an irrelevant
# chunk can't crowd out the relevant ones ("lost in the middle"). The agent can
# retrieve multiple *times*; this is the width of one retrieval.
TOP_K = 5

# --------------------------------------------------------------------------- #
# Retrieval benchmark — the four arms
# --------------------------------------------------------------------------- #
# The cutoff the benchmark reports at. TOP_K = 5 is what the *system* feeds the
# LLM; EVAL_K = 10 is what the *evaluation* measures over. They are deliberately
# different numbers: nDCG@10 is the field's convention (BEIR, MS MARCO, TREC all
# report at 10), and measuring deeper than you serve shows whether a relevant
# document was merely ranked 7th rather than missing altogether — which is the
# difference between "retrieval failed" and "the cutoff is too tight".
EVAL_K = 10

# --- BM25 (the lexical arm) ------------------------------------------------ #
# BM25's term-frequency saturation. Raising k1 lets repeated terms keep adding
# score; lowering it makes one occurrence nearly as good as five. 1.5 is the
# long-standing default from the TREC literature and the value Lucene and
# Elasticsearch ship, so the lexical baseline is the one a reviewer expects
# rather than a tuned-to-flatter-itself variant.
BM25_K1 = 1.5

# BM25's length normalisation, in [0, 1]. b = 1 fully penalises long documents,
# b = 0 ignores length entirely. 0.75 is again the Lucene/TREC default. Our
# chunks are near-uniform in length by construction (CHUNK_SIZE_CHARS), so this
# parameter does comparatively little here — worth knowing before claiming BM25
# was carefully tuned. It was not; it is the honest out-of-the-box baseline.
BM25_B = 0.75

# --- Hybrid (reciprocal-rank fusion) --------------------------------------- #
# RRF combines two ranked lists by summing 1 / (RRF_K + rank) across them. The
# constant damps the influence of the very top ranks so one arm's confident-but-
# wrong first hit cannot dominate the fusion. 60 is the value from Cormack et al.
# (2009), where RRF was introduced, and it is what makes fusion work without
# needing the two arms' scores to be on a comparable scale — which dense cosine
# similarity and BM25 emphatically are not.
RRF_K = 60

# --- Cross-encoder reranker ------------------------------------------------ #
# A *cross*-encoder reads the query and the document together and scores the
# pair, instead of embedding each separately and comparing vectors. That joint
# read is why it ranks better — and why it cannot be precomputed into an index,
# so it must run at query time over a candidate shortlist. This is the accuracy/
# latency trade the benchmark exists to measure.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates the reranker rescores. Every candidate is one forward pass,
# so this is the arm's cost dial: too few and the reranker cannot rescue a
# relevant document the first-stage retriever buried; too many and latency grows
# for candidates that were never plausible. 50 is a common first-stage depth and
# is 5x the reported cutoff, so a document has to be badly misranked to be lost
# before reranking gets a chance at it.
RERANK_CANDIDATES = 50

# --- Uncertainty ----------------------------------------------------------- #
# Resamples for the percentile bootstrap. The bootstrap resamples *questions*,
# not documents, because the question set is the sample we generalise over — the
# claim being defended is "on a query like these", not "on this exact list".
# 2000 is enough for a stable 95% interval; more mostly costs time.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CI = 0.95

# --- The known-item question set ------------------------------------------- #
# How many papers to sample for generated queries. Each sampled paper yields
# three queries — one per query style (title, abstract sentence, keyword bag) —
# so 60 papers gives 180 graded queries. That is enough for a bootstrap interval
# to be worth printing, which the original 11 hand-written questions were not.
# Sampling is seeded, so the set is identical on every machine.
KNOWN_ITEM_PAPERS = 60

# Length of a keyword query, in content words. Real search queries are short,
# unordered word bags, not grammatical sentences — and a bi-encoder trained on
# sentences has no reason to handle them equally well. Five words is short enough
# to be a realistic search box entry and long enough to identify one paper out of
# 2,012. This is the harshest query style in the set, deliberately.
KEYWORD_QUERY_WORDS = 5

# The shortest abstract sentence accepted as a query. Very short sentences
# ("We release our code.") identify no particular paper, so they would add noise
# to the ground truth rather than difficulty.
MIN_QUERY_CHARS = 40

# --- Drift probe ----------------------------------------------------------- #
# How many newer papers to pull for the staleness check. These are papers the
# index has never seen; the probe asks how well the existing index covers the
# topics they talk about. Kept small because it is one polite arXiv page-walk.
DRIFT_PAPERS = 200

# --- The re-indexing trigger, and the mistake it took a re-index to find ----- #
#
# Drift is measured as the drop in mean top-1 similarity between queries drawn
# from *indexed* papers and queries drawn from *newer, unseen* ones. The original
# trigger fired on a raw 15% relative drop, and on 2026-07-27 the probe read
# 27.2% — so the project recommended re-indexing.
#
# The index was rebuilt on 2026-08-12, from a corpus with **zero** paper overlap
# with the old one (see outputs/results/corpus_turnover.json). The same probe on
# that one-hour-old index read **26.7%**. Rebuilding — the largest freshness
# intervention available — moved the number by half a percentage point.
#
# That is the metric failing, not the index. The drop has a large structural
# floor that has nothing to do with staleness: a baseline query is a sentence
# lifted verbatim out of a document sitting in the index, so its top-1 match is
# near-perfect by construction, while a probe query comes from a document that is
# not in the index and never can be. The gap between "verbatim self-match" and
# "closest topical neighbour" is a property of the measurement, and it is there on
# day zero. A threshold on the raw drop therefore fires on *every* index, always,
# and says nothing.
#
# The floor below is that day-zero reading — measured, not assumed, on the
# freshly-built 2026-08-12 index. It is specific to this corpus, this embedding
# model and this window size, and must be re-measured if any of them change; that
# is the cost of the metric being interpretable at all.
DRIFT_FLOOR_RELATIVE_DROP = 0.267

# What the trigger actually tests: how far the current drop exceeds the day-zero
# floor. This is the "gap widening over successive probes" that drift.py's own
# docstring always described as the signal — the original constant just never
# implemented it. 10 excess points is a judgement call about cost, not a property
# of the data: re-embedding this corpus is a few minutes on a laptop, so it can be
# tight; on an overnight GPU job it would be looser.
DRIFT_TRIGGER_EXCESS_DROP = 0.10

# --------------------------------------------------------------------------- #
# Generation + agent (Claude via the Anthropic API)
# --------------------------------------------------------------------------- #
# The generation model. Sonnet 5 is the cost/quality sweet spot for grounded
# question-answering over retrieved context — near-Opus quality at Sonnet price.
# Swap to "claude-opus-5" for maximum quality or "claude-haiku-4-5" to cut cost;
# every call reads this constant, so it changes in one place.
GEN_MODEL = "claude-sonnet-5"

# The judge model for the answer-quality evaluation (LLM-as-judge). Kept the same
# as the generation model by default so the grader is at least as capable as the
# thing it grades.
JUDGE_MODEL = "claude-sonnet-5"

# Hard ceiling on answer length. Generous enough for a cited multi-paragraph
# answer, bounded so a runaway generation can't rack up cost.
MAX_ANSWER_TOKENS = 1024

# The agentic loop's stopping condition. The agent may call the retrieval tool,
# read what it got, decide it needs more, and search again — but not forever.
# After this many tool-use rounds it must answer with what it has. 4 is enough
# for a two- or three-hop question (decompose → search → search again → answer)
# without letting a confused agent loop indefinitely and burn tokens.
MAX_AGENT_STEPS = 4

# Where the on-disk LLM response cache lives. Every Claude call is cached by a
# hash of its inputs so (a) the notebook and tests reproduce published numbers
# WITHOUT an API key, and (b) re-running the pipeline doesn't pay for identical
# calls twice. This is the same "reference-only, load the saved result"
# convention the portfolio's Kaggle-GPU notebooks use.
LLM_CACHE_FILE = RESULTS_DIR / "llm_cache.json"

# The environment variable holding the Anthropic API key. NEVER hard-code a key;
# NEVER commit one. If it is unset, live LLM calls are skipped and the cache is
# used instead — so the project degrades gracefully to "reproduce from cache".
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"


def api_key_available() -> bool:
    """True when an Anthropic API key is present in the environment.

    Used everywhere a live call is optional: if this is False we fall back to the
    committed response cache instead of erroring, which is what lets the whole
    project reproduce on a machine with no key.
    """
    return bool(os.environ.get(ANTHROPIC_API_KEY_ENV))


def ensure_dirs() -> None:
    """Create every output/data directory this project writes to.

    Called at the top of the pipeline and eval entry points so a fresh checkout
    (where data/ and outputs/ are git-ignored and therefore absent) just works.
    """
    for d in (RAW_DIR, PROCESSED_DIR, INDEX_DIR, FIGURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
