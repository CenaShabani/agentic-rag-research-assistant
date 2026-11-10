# `src/acquisition.py` — getting the corpus, and proving where it came from

## What it's for

Downloads papers from the arXiv API, parses the Atom feed into `Paper` objects, and writes them
to disk **with a provenance record**. First stage of the pipeline: `acquisition` → `chunking` →
`embeddings` → `vectorstore`.

Two things distinguish it from a download script: it records licence per paper, and it fingerprints
what it wrote.

## The public surface

| Member | In | Out |
|---|---|---|
| `Paper` | dataclass | `arxiv_id, title, abstract, authors, categories, published, license` + `.text`, `.redistributable` |
| `fetch_papers(max_papers)` | int | `list[Paper]` — paged, rate-limited |
| `fetch_by_ids(arxiv_ids)` | tuple of ids | `list[Paper]` — the seeded seminal papers |
| `save_papers(papers, path=None, provenance_path=None)` | `list[Paper]` | writes JSONL + provenance |
| `load_papers(path=None)` | — | `list[Paper]` |
| `acquire()` | — | the whole thing: fetch + seed + save |

`Paper.text` is `title + "\n\n" + abstract` — **this is the field retrieval actually runs over**,
and it is the reason the README says the corpus is abstracts, not full text.

## The walkthrough

### The licence field is the interesting design decision

`Paper` records `license`, and `redistributable` checks it against a whitelist. Neither is used to
filter anything today, because the project only ever stores title + abstract, which is
licence-safe for every paper.

So why record it? Because it makes a future decision *defensible rather than retroactive*. If the
corpus were extended to full text, the per-paper licence is already there and the filter is a
one-line change. Deciding to store full text first and check licences afterwards is how a project
ends up with data it cannot publish. The provenance record even counts `redistributable_count`,
so the proportion is visible before anyone tries.

This is the `02-data-acquisition` discipline showing up in code rather than in a README promise.

### Parsing defensively

```python
arxiv_id = raw_id.rsplit("/", 1)[-1].split("v")[0] if raw_id else ""
```

The `<id>` element is a URL like `http://arxiv.org/abs/2401.01234v2`. Two operations: take the
tail, then strip the version suffix. **Stripping the version matters for evaluation** — ground
truth is written as `2401.01234`, and a corpus storing `2401.01234v2` would score every correct
retrieval as a miss. A one-character difference that would silently invalidate the headline number.

```python
if arxiv_id and title and abstract:
    yield Paper(...)
```

Entries missing any of the three are skipped rather than yielded with empty fields. A paper with
no abstract contributes nothing retrievable, and letting it through would produce empty chunks
that embed to noise.

```python
title=" ".join(title.split())
```

arXiv wraps titles and abstracts with newlines and runs of spaces. Collapsing whitespace at parse
time means every downstream consumer — chunker, embedder, prompt formatter — sees clean text, and
chunk boundaries aren't thrown off by stray formatting.

`license_el is not None` rather than a truthiness test: an `Element` with no children is falsy in
ElementTree, so `if license_el:` would treat a present-but-empty element as absent. A classic
ElementTree trap, correctly avoided here.

### Provenance

`_sha256` streams the file in 1 MB blocks rather than reading it whole — the pattern that keeps
memory flat regardless of corpus size.

**The paths on `save_papers` / `load_papers` are parameters, defaulting to the main corpus.**
That exists so any *other* slice of arXiv the project downloads — currently the drift probe's
newer slice (`src/drift.py`) — gets the same provenance treatment: same fields, same SHA-256
fingerprint, same code. A second downloader with its own weaker provenance is exactly how a
corpus quietly becomes untraceable.

`save_papers` writes `provenance.json` alongside the data: source URL, categories, UTC timestamp,
paper count, redistributable count, filename and **SHA-256 of the file**. That hash is what makes
"2,012 papers" a checkable claim rather than a number in a README. Anyone can re-hash the file and
confirm the analysis ran on the data being described.

## The non-obvious bits

- **JSONL, not JSON.** One paper per line means the file is appendable, streamable, and readable
  with `head` — and a single corrupt line doesn't take down the whole parse.
- **`asdict(p)`** serialises the dataclass; `Paper(**json.loads(line))` reverses it. The round trip
  works only because every field is a JSON-native type, which is why `published` is a string rather
  than a `datetime`.
- **Rate limiting is a config constant** (`ARXIV_REQUEST_DELAY_SECONDS = 3.0`), matching arXiv's
  stated guidance. This is why acquisition takes minutes.
- **`fetch_by_ids` is separate from `fetch_papers`** because the seeded seminal papers are fetched
  by explicit id, not by category query — a different API call shape for a different purpose.

## How it can break

- **arXiv is a live API.** A schema change, an outage or a rate-limit tightening breaks fetching.
  The saved JSONL plus provenance file is the mitigation: the corpus on disk is what the numbers
  describe, and re-running is not required to reproduce them.
- **`_parse_feed` silently drops malformed entries.** Correct, but it means a partial feed yields a
  smaller corpus without an error. The provenance `paper_count` is what makes that visible.
- **Re-running `acquire()` refetches "recent" papers**, which drift over time — a later run
  produces a *different* corpus with a different hash. That is why the hash is recorded rather than
  assumed stable.
- **No retry on a failed page.** A network blip mid-crawl loses that page's papers; the count in
  provenance is the tell.
