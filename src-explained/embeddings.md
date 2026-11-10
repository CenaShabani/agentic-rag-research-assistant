# `src/embeddings.py` — text to vectors

## What it's for

Maps text to points in a 384-dimensional space where "near" means "about the same thing". It is
the hinge the whole project turns on: it is what lets a query find a chunk by *meaning* rather
than by shared words.

Sits between `chunking` (which produces the text) and `vectorstore` (which indexes the vectors).
Sixty-one lines, and two of the decisions in it shape everything downstream.

## The public surface

| Function | In | Out |
|---|---|---|
| `get_model()` | — | a cached `SentenceTransformer` |
| `embed_texts(texts)` | `list[str]` of length n | `np.ndarray`, shape `(n, 384)`, dtype `float32`, **L2-normalised** |
| `embed_query(query)` | one `str` | `np.ndarray`, shape `(1, 384)` — the shape FAISS wants |

`embed_query` exists purely so callers don't have to remember to wrap a single string in a list.
Passing a bare string to a sentence-transformer silently embeds it *per character* in some
versions, which is the kind of bug that produces plausible-looking nonsense.

## The walkthrough

**The lazy import is a deliberate structural choice, not tidiness.** The `from
sentence_transformers import SentenceTransformer` sits *inside* `get_model()`, so importing this
module costs nothing. The tests and any config-reading tool can import it without
sentence-transformers being installed and without triggering a ~90 MB model download. If that
import sat at module top level, every test run would pay for it.

`@functools.lru_cache(maxsize=1)` on `get_model` makes the model a singleton. Loading it takes
seconds; doing that once per batch instead of once per call is the difference between a pipeline
that finishes and one you abandon.

**`normalize_embeddings=True` is the load-bearing line in the file.** It makes every vector unit
length, which makes the inner product between two vectors exactly equal to their cosine
similarity. That equivalence is what lets `vectorstore` use a fast inner-product index and have
the result mean "most semantically similar". Turn this off and the index still runs, still
returns results, and quietly ranks by a quantity that rewards long chunks — a silent quality
regression with no error message.

`.astype(np.float32)` at the end is not cosmetic either: FAISS requires float32 and will reject
or misread float64 input.

## The non-obvious bits

- **`show_progress_bar=False`** — keeps notebook output clean, so a re-run diff shows real
  changes rather than progress bars.
- **`batch_size` comes from config**, not a literal, because it is the memory/throughput knob
  someone will actually want to turn on a different machine.
- **`EMBED_DIM` is asserted in config, not derived here.** The dimension is a property of the
  chosen model; hard-coding 384 in two places would be how a model swap breaks the index.
- **Determinism.** The model is fixed and runs locally with no sampling, so the same text embeds
  to the same vector on every machine — this is what keeps the retrieval half of the evaluation
  reproducible with no API key, which is the project's headline reproducibility claim.

## How it can break

- **First call downloads ~90 MB.** Offline with a cold cache, `get_model()` raises. Everything
  after it fails too, which is the correct behaviour — a half-embedded corpus is worse than none.
- **Changing `EMBED_MODEL` invalidates the index.** A different model means a different vector
  space and possibly a different dimension; old vectors and new queries are not comparable. Both
  the index and the chunk metadata must be rebuilt together.
- **Very long chunks are silently truncated** at the model's max sequence length (256 word-pieces
  for MiniLM). Text past that limit contributes nothing to the vector. Chunk size is set well
  inside that limit in `config`, which is why this doesn't bite here — but it is exactly what
  breaks when someone raises `CHUNK_SIZE_CHARS` without checking.
- Tests cover shape, dtype and unit-norm of the output. They do not assert specific vector
  values, which would pin the test to a model version.
