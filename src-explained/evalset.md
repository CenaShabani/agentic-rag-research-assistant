# `src/evalset.py` — manufacturing 180 graded questions without an API key

## What it's for

Generates the evaluation set the benchmark measures on. The project started with **11
hand-written questions**, which is too few to put a confidence interval on: one question
changing its mind moves the mean by nine percentage points, so "arm A beats arm B" over 11 rows
is noise wearing a decimal point.

This file produces 180 more, deterministically, with no human adjudicator and no API key.

## The construction

```
take a paper → derive a query from it → the right answer is that paper
```

This is **known-item retrieval**, a standard IR evaluation design. Its advantage is that the
relevance judgement is not a matter of opinion: the query was manufactured *from* the document
it points to, so there is no adjudicator to disagree with. 180 unambiguous labels for free.

Every sampled paper yields three queries, one per **query style**:

| Style | Query | Character |
|---|---|---|
| `abstract` | longest non-boilerplate sentence in the abstract | fluent prose, verbatim substring of its target |
| `title` | the paper's title | natural language, short, content-dense |
| `keyword` | 5 content words from across the abstract, stop-words dropped, order shuffled | no grammar, not a substring of anything |

## The public surface

| Function | In | Out |
|---|---|---|
| `build_known_item_set(papers=None)` | `list[Paper]` or `None` (loads the corpus) | `list[dict]` — 3 per eligible sampled paper |
| `save_known_item_set(rows)` | the rows | writes `eval/known_item_set.json` |
| `load_known_item_set()` | — | the committed set, generating it if absent |

Each row is `{qid, question, relevant_ids, hop, flavour}` — the same shape as the hand-written
`eval/qa_set.json`, so `EvalQuestion(**row)` consumes both without a special case. `flavour`
defaults to `"curated"` on the dataclass, which is why the hand-written file still loads
unchanged.

## The mistake this file made first, and the fix

**The first version had only the `abstract` style, and it was useless.** Every arm scored ~1.00
and three of the four tied at exactly 1.000.

The reason is worth internalising: an abstract in this corpus is ~1,000 characters, chunking is
~1,000 characters, so most abstracts are a single chunk. A sentence lifted verbatim from the
abstract is therefore a **literal substring of the very passage it targets**. That is a
degenerate retrieval task — it asks "can you find the document containing this exact text?",
which any method solves. A benchmark where every arm ties has measured nothing.

The `keyword` style is the fix. Five shuffled content words drawn from across the *whole*
abstract are not a substring of any single passage, and there is no grammar left for a
sentence-embedding model to lean on. It separated the arms immediately — dense fell from 0.969
to 0.291 while BM25 barely moved.

**And the fix turned out to be more interesting than the problem.** The three styles form a
gradient from fluent prose to a word bag, so comparing one arm *across* them measures how much
that arm depends on the user phrasing their query nicely. That is a real property of a real
system, and no single averaged score can show it.

## The walkthrough

**`_sentences` splits on punctuation-then-capital and sorts longest-first.** Crude, but
abstracts are well-punctuated prose and this avoids a heavyweight NLP dependency for a job that
does not need one. Over-splitting is self-correcting, since fragments below `MIN_QUERY_CHARS`
are dropped anyway. Longest-first because length correlates with specificity, and a query has to
be specific to identify one paper out of 2,012.

**`_BOILERPLATE_PREFIXES` drops housekeeping sentences.** "We release our code", "In this
paper…" — every paper has them, so they identify no particular document. Including them would
not make the benchmark harder, it would make it *impossible for every arm equally*, which is
noise, not difficulty.

**`_STOPWORDS` includes academic filler** — "propose", "novel", "results", "approach" — because
those appear in nearly every abstract in this corpus and would waste keyword slots. Critically,
this list is used **only to build queries, never inside a retriever**: BM25's IDF already
discounts ubiquitous terms, and a hand-curated stop list inside the baseline would be an
untunable thumb on the scale.

**The sort before the sample is not decoration:**

```python
eligible = sorted((p for p in papers if _eligible(p)), key=lambda p: p.arxiv_id)
rng = random.Random(config.SEED)
sampled = rng.sample(eligible, n)
```

`load_papers` returns whatever order the JSONL happened to be written in. A seeded sample over
an unsorted list is only reproducible if the input order is too. Without that `sorted`, the
"deterministic" set would silently depend on download order — and
`test_input_order_does_not_change_the_sample` pins it.

**`_eligible` requires a paper to support *all three* styles.** That keeps the slices perfectly
paired: same papers, same targets, only the phrasing differs. If a paper could contribute to one
style and not another, comparing an arm across styles would compare two different samples of
papers and the slope in figure 06 would be uninterpretable.

## The non-obvious bits

- **`hop` is `"single"` for every generated query.** A known-item query is answered by exactly
  one paper, by construction. Multi-hop questions cannot be manufactured this way — which is
  precisely why the 11 hand-written ones remain the harder slice and are reported separately.
- **The keyword seed is `config.SEED + i`**, per paper. A single shared seed would give every
  paper the same shuffle pattern.
- **`eval/known_item_set.json` is committed**, unlike the corpus. It is small, and every
  published number depends on it — a reviewer checking a claim needs the exact queries it was
  measured on, not a promise that they were regenerable.

## How it can break

- **The committed set and the index can drift apart.** If the corpus is re-downloaded, sampled
  papers may no longer be indexed and every arm scores 0 on those rows.
  `test_the_committed_set_matches_the_real_corpus` cross-checks every label against
  `chunk_meta.parquet` for exactly this.
- **The labels are correct by definition, which is not the same as realistic.** "Correct by
  construction" and "a query a human would ever type" are different claims, and only the first
  is being made. That is what `eval/ADJUDICATION.md` is for, and why the hand-written set is
  kept as the control.
- **Every query is built from its target's own words**, which structurally favours lexical
  matching. Stated prominently rather than buried — it is the single biggest caveat on the
  headline number.
