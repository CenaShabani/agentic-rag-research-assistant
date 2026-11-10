# `src/chunking.py` — cutting papers into retrievable pieces

## What it's for

Turns each paper into overlapping windows of text small enough that one embedding vector means
one specific thing. It sits third in the pipeline: `acquisition` produces `Paper` objects,
**`chunking` slices them**, `embeddings` turns each slice into a vector.

Its second job matters as much as the first: every chunk carries its parent paper's id and
title, which is the only reason an answer can cite anything.

## The public surface

| Function | In | Out |
|---|---|---|
| `split_text(text, size, overlap)` | `str`, two ints (default from `config`) | `list[str]` — overlapping windows |
| `chunk_paper(paper, size, overlap)` | one `Paper`, two ints | `list[Chunk]` |
| `chunk_papers(papers, size, overlap)` | `list[Paper]`, two ints | `list[Chunk]` — the whole corpus |
| `chunks_to_frame(chunks)` | `list[Chunk]` | `DataFrame[chunk_id, arxiv_id, title, text, position]` |
| `save_chunks` / `load_chunks` | — | writes/reads the parquet the vector store consumes |

`Chunk` is a dataclass: `chunk_id` (`"2401.01234:2"` — paper id, colon, position), `arxiv_id`,
`title`, `text`, `position`.

The `chunk_id` format is doing real work: it is unique, sorts sensibly, and can be parsed back
to its paper without a lookup table. That is why citations survive the round trip through FAISS,
which only stores integer row indices.

**`size` and `overlap` are threaded all the way down on purpose.** They default to the shipped
constants, and `run_pipeline.py` never passes them — there is still exactly one chunking
configuration in the shipped system, and it lives in `config.py`. They are parameters so
`run_ablation.py` can rebuild the corpus at other settings and *measure* what the constant is
worth, without monkeypatching a module-level value. A knob you can only change by mutating
global state is a knob nobody tests.

## The walkthrough

`split_text` is the whole module. Everything else is plumbing around it.

The loop advances a window of `size` characters, stepping forward by `size - overlap` each
iteration so consecutive windows share `overlap` characters of text. That shared margin is the
point of the design: a fact sitting across a boundary survives whole in at least one chunk.
Without it, the sentence that answers a question can be severed in two and match nothing.

Inside the loop, the interesting decision is the whitespace snap:

```python
last_space = text.rfind(" ", start, end)
if last_space > start + size // 2:
    end = last_space
```

Ending a window mid-word corrupts the embedding — `transform` and `transformer` are different
tokens and the model has no idea a word was guillotined. So the window prefers to end at the
last space. **The `> start + size // 2` guard is the non-obvious half.** On text with no spaces
for a long stretch — a URL, a base64 blob, a long equation — `rfind` would return a position
near the start of the window and every chunk would collapse to a sliver. The guard says: only
snap backwards if it costs less than half the window. Otherwise take the hard cut.

The advance is `start = end - overlap`. Note it is computed from `end`, not from
`start + size` — so after a whitespace snap the overlap is still measured from where the chunk
actually ended. Getting this wrong is how overlap silently drifts.

## The non-obvious bits

- **`if overlap >= size: raise`** — not defensive padding. If overlap ever equalled size, the
  advance would be zero and the loop would spin forever appending the same window. It is a
  guard against an infinite loop, so it raises rather than clamps.
- **Character-based, not token-based.** A tokenizer would be more principled — models think in
  tokens, not characters — but it would put a heavyweight dependency in the retrieval path and
  make chunking non-reproducible across tokenizer versions. The trade is stated in the module
  docstring and it is the right one for a corpus of abstracts.
- **`text.strip()` before and after.** Leading whitespace shifts every subsequent boundary;
  trailing whitespace produces empty chunks that embed to noise.
- **The final `[c for c in chunks if c]`** drops empties, which can appear when a snap lands on
  a run of spaces.
- **Size and overlap live in `config.py`, not here.** They are argued constants, and the
  argument belongs where every other tunable is.

## How it can break

- **Very short papers** return a single chunk — handled explicitly by the early return, so
  `len(text) <= size` never enters the loop.
- **An empty abstract** returns `[]`, and that paper simply contributes no chunks. It does not
  raise. Worth knowing: a paper can be in the corpus and unreachable by retrieval.
- **Pathological text with no whitespace** falls back to hard cuts, by design.
- **Changing `CHUNK_SIZE_CHARS` invalidates the index.** Chunks, vectors and FAISS ids are
  positional — re-chunking without re-embedding produces citations that point at the wrong text.
  Re-run the whole pipeline, never just this step.
- The tests pin the boundary behaviour: overlap is respected, no word is split when a space is
  available, and the overlap-≥-size guard raises. They do not pin behaviour on multi-byte
  characters — character offsets are Python string indices, so combining characters could in
  principle split a grapheme.
