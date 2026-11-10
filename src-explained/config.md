# `src/config.py` — every knob, argued, in one place

## What it's for

The single place where every path, constant and model id lives. Nothing else in the project
hard-codes a number that someone might want to change.

It is imported by every other module and imports none of them — the root of the dependency graph,
which is what lets the Kaggle notebook generator inline modules in a safe order.

The reason it is worth reading rather than skimming: **most constants here carry their rationale
in a comment.** The file is the project's argued-decisions record, not a settings dump.

## The public surface

Constants, grouped:

| Group | Key values |
|---|---|
| Reproducibility | `SEED = 42` |
| Paths | `PROJECT_ROOT` … `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `INDEX_DIR`, `OUTPUTS_DIR`, `FIGURES_DIR`, `RESULTS_DIR`, `EVAL_DIR` |
| Files | `PROVENANCE_FILE`, `PAPERS_FILE` (jsonl), `CHUNKS_FILE` (parquet), `FAISS_INDEX_FILE`, `CHUNK_META_FILE` |
| Acquisition | `ARXIV_CATEGORIES = ("cs.CL", "cs.LG")`, `MAX_PAPERS = 2000`, `ARXIV_PAGE_SIZE = 100`, `ARXIV_REQUEST_DELAY_SECONDS = 3.0`, `SEED_PAPER_IDS` |
| Chunking | `CHUNK_SIZE_CHARS = 1000`, `CHUNK_OVERLAP_CHARS = 150` |
| Embedding | `EMBED_MODEL`, `EMBED_DIM = 384`, `EMBED_BATCH_SIZE = 64` |
| Retrieval | `TOP_K = 5` |
| Generation | `GEN_MODEL`, `JUDGE_MODEL`, `MAX_ANSWER_TOKENS`, `MAX_AGENT_STEPS` |

Plus `ensure_dirs()`, called by anything that writes.

## The walkthrough

**`PROJECT_ROOT = Path(__file__).resolve().parent.parent`** is the line everything else hangs off.
Deriving paths from the module's own location — rather than from the current working directory —
is why the pipeline behaves identically whether it is run from the repo root, from `notebooks/`,
or by pytest.

It is also the one line that needs special handling when the Kaggle notebook is generated, since
an inlined module has no real `__file__`. The generator sets a synthetic one pointing at
`/kaggle/working/src/config.py`, so this expression still resolves sensibly. Worth knowing that
the two files are coupled in that specific way.

**The constants that are genuinely decisions, not defaults:**

- `ARXIV_REQUEST_DELAY_SECONDS = 3.0` — arXiv's stated rate limit. Politeness to a free API that
  the project depends on, and the reason acquisition takes minutes rather than seconds.
- `SEED_PAPER_IDS` — the 12 seminal papers (Transformer, BERT, RAG, DPR, ReAct, …) fetched
  explicitly. **This is an evaluation-integrity decision, not a curation preference:** the eval
  questions have known-correct answers, and those answers must be *in* the corpus or retrieval
  metrics measure nothing. Without seeding, a recent-papers-only crawl might not contain them.
- `CHUNK_SIZE_CHARS = 1000` / `CHUNK_OVERLAP_CHARS = 150` — the two knobs that shape retrieval
  precision. 15% overlap is enough that a fact spanning a boundary survives whole in one chunk,
  without inflating the index much. Note 1000 characters sits comfortably inside MiniLM's
  256-word-piece limit, which is what stops silent truncation.
- `EMBED_DIM = 384` — a property of the chosen model, stated once so the FAISS index and the model
  cannot disagree.
- `TOP_K = 5` — used by naive RAG, the agent and the evaluation. One value, so the comparison is
  fair by construction rather than by remembering.
- `MAX_AGENT_STEPS` — the agent's retrieval budget and the backstop against an unbounded loop.
- `SEED = 42` — the reproducibility contract.

### `DRIFT_FLOOR_RELATIVE_DROP` — the one constant that is a measurement

Every other number here is a judgement someone argued for. This one is **an observation**: it is
what the drift probe reads on an index built an hour earlier, and it exists because the previous
version of the trigger — a flat 15% on the raw drop — fired on a brand-new index. See
`drift.md` for the full account; the part that belongs in a discussion of *config* is the
category difference.

A threshold you can defend from first principles (`ARXIV_REQUEST_DELAY_SECONDS`, `RRF_K`) stays
valid when the corpus changes. A threshold calibrated against a measured floor does **not**:
change the embedding model, the chunk size or `MAX_PAPERS`, and the floor moves while the
constant does not, so the trigger keeps returning verdicts that no longer mean anything. That
makes it the most fragile line in this file, and the comment above it says so. The project's rule
is that a number which changes the answer must be visible and defensible — this one is visible,
defensible, and has an expiry condition, which is a third thing the rule did not originally
anticipate.

## The non-obvious bits

- **`GEN_MODEL` and `JUDGE_MODEL` are separate constants** even when set to the same value. They
  are different roles, and a future run might want a stronger judge than generator. Collapsing
  them would make that a code change rather than a config change.
- **`ensure_dirs()` exists so no writer has to think about it.** Every save path calls it first;
  a missing directory never surfaces as an error mid-pipeline.
- **`PAPERS_FILE` is `.jsonl`, `CHUNKS_FILE` is `.parquet`.** Deliberate: papers are appended and
  human-inspectable line by line; chunks are a columnar table read whole by the vector store.
- **No secrets here.** The API key comes from the environment in `llm.py`, never from config.

## How it can break

- **Changing `CHUNK_SIZE_CHARS`, `CHUNK_OVERLAP_CHARS` or `EMBED_MODEL` invalidates the index.**
  Chunks, vectors and FAISS row ids are positional; changing one without rebuilding all of them
  produces citations that point at the wrong text — a silent, plausible-looking failure.
- **Changing `TOP_K` changes every reported metric**, since they are all "@k" measures. Numbers
  from different k values are not comparable.
- **Raising `MAX_PAPERS` a long way** makes the exact FAISS index slower linearly and the
  acquisition much slower (rate limit × pages).
- **Changing `GEN_MODEL` or a prompt invalidates the committed LLM cache**, because the model id is
  part of the cache key. Cached numbers then need regenerating with a key.
