# `src/drift.py` — when does this index need rebuilding?

## What it's for

Every number in this project describes an index built on a particular day, and arXiv publishes
hundreds of cs.CL / cs.LG papers a week. Measured quality starts decaying the moment the build
finishes, and nothing in a notebook notices.

This module measures that decay and converts it into a **re-indexing trigger** — the one
operational question a benchmark like this should be able to answer. It is also the only part of
the evaluation that touches the network.

**Read the correction below before quoting any number from this module.** The first version of
the trigger was wrong, the project acted on its recommendation, and acting on it is what exposed
the error. That sequence is the most useful thing in this file.

## The obvious probe, and why it does not work

The natural design is: take queries about newer papers, run them, watch the scores fall.

That measures nothing. Those papers **are not in the index**, so no retriever can possibly
return them. Every arm scores exactly zero, and zero is not a measurement of drift — it is a
measurement of the setup. Worse, it would look like a dramatic result.

## What is measured instead

The question that *can* be answered is whether the index still contains anything close to the
new material. So the probe compares two similarity distributions:

```
baseline : queries drawn from *indexed* papers        → mean top-1 cosine
probe    : queries drawn from *newer, unseen* papers  → mean top-1 cosine
```

The baseline is a ceiling — those queries came from documents sitting in the index, so their
best match should be near-perfect. The probe asks how well the existing corpus covers what the
field is talking about *now*.

**Some gap is normal and permanent.** An index cannot contain the future; this is not a bug to
be tuned away. The signal is the gap widening across successive probes.

The dense arm is used because it is the only one with a calibrated, comparable notion of
distance. A BM25 score is unbounded and collection-dependent, so "BM25 score went down" would
not mean anything across two different sets of queries.

## The trigger, and the version of it that was wrong

```
relative_drop = (baseline_mean − probe_mean) / baseline_mean
```

The original rule fired when this exceeded a flat 15%. On **2026-07-27** it read
**0.730 vs 0.532 — a 27.2% drop**, comfortably past the threshold, so the README recommended
re-indexing and the project tracked it as an action.

On **2026-08-12** the index was rebuilt, against a corpus sharing **not one paper** with the old
one (`outputs/results/corpus_turnover.json`). The same probe, run on an index one hour old, read
**0.725 vs 0.531 — a 26.7% drop.**

The largest freshness intervention available moved the metric by half a percentage point.

### Why the level cannot mean what it was taken to mean

A baseline query is a sentence lifted verbatim out of a document that is *in the index*, so its
top-1 match is a near-self-match by construction. A probe query comes from a document that is
absent by definition, so its top-1 match is the closest topical neighbour. The gap between
"verbatim self-match" and "closest neighbour" is a property of **the measurement**, not of the
index's age. It is there in full on day zero.

A threshold on the raw drop therefore fires on every index, always. It is the kind of metric
that produces a confident recommendation from a quantity that cannot vary in the direction being
claimed — and note that the module's own prose always said the *widening* was the signal. The
constant simply never implemented what the docstring described. The prose was right and unread;
the code was wrong and trusted.

### What it does now

```
excess = relative_drop − config.DRIFT_FLOOR_RELATIVE_DROP
```

Re-index when `excess` exceeds `config.DRIFT_TRIGGER_EXCESS_DROP` (0.10). The floor
(`0.267`) is the day-zero reading, **measured on the freshly-built index rather than assumed**,
and it is specific to this corpus, this embedding model and this window size — change any of
them and it has to be measured again. That is the price of the metric being interpretable at
all, and it is cheaper than the alternative, which is a threshold that means nothing.

On the current index: `excess = +0.0005`, so no re-index. Stating a threshold still means
stating a cost — re-embedding this corpus is a few minutes on a laptop, so the tolerance can be
tight; on an overnight GPU job it would be looser. **Both numbers are business decisions
informed by a metric, not properties of the data.**

## The public surface

| Function | Does |
|---|---|
| `fetch_drift_slice(known_ids, n)` | downloads recent arXiv papers not already in the corpus |
| `load_or_fetch_drift_slice(known_ids)` | cached-first, but re-checked against the index; raises `DriftSliceUnavailable` |
| `measure_drift(store)` | the full probe → one record |
| `MIN_USABLE_SLICE_FRACTION` | `0.5` — how much of a cached slice must still be unseen for it to be usable |

## The walkthrough

**The slice is "newest first, minus what we already have".** `acquisition.fetch_papers` already
pages politely and sorts by submission date descending, so it is reused verbatim. It
**over-fetches by 4×** because the newest page overlaps heavily with a corpus downloaded days
ago, and only the non-overlap is usable.

**The cache is re-checked on load, not just on fetch.** This is the second bug the re-index
exposed, and it is worth being precise about it because an earlier version of this very file
listed the behaviour as a *known limitation* rather than a defect:

> "It needs the network once. After that the cached slice is used forever, which means a stale
> cache reports stale drift."

That reads like a caveat. It is a correctness bug, and re-indexing is exactly the event that
detonates it. A probe slice is "newer, unseen papers" only relative to the index it was fetched
against; rebuild the index and yesterday's slice is mostly *inside* the new corpus. Reusing it
then compares the index against its own contents — top-1 similarity near 1.0 on both sides, the
drop collapsing toward zero, and a stale index pronounced healthy. Silent and
self-congratulatory, which is the worst failure mode a monitoring metric can have.

So `load_or_fetch_drift_slice` filters the cached slice by `known_ids` and re-fetches when fewer
than half of `config.DRIFT_PAPERS` survive. Below that floor the remaining papers are too few
for the mean to be anything but noise. `tests/test_drift.py` pins all three cases: fresh cache
reused, absorbed cache replaced, and the re-fetch being told what the index already holds.

**The slice is cached to `data/raw/papers_drift.jsonl`** through `acquisition.save_papers` — with
a path argument added for this purpose — so it gets the same provenance treatment as the main
corpus: same fields, same SHA-256 fingerprint, same code. A second downloader with its own
weaker provenance is how a corpus quietly becomes untraceable.

**Both query sets use abstract sentences, not titles.** A title is short and unusually
distinctive; a mid-abstract sentence is a fairer probe of whether the index covers the *content*
of the new literature. Both sides use the same construction, which is what makes them
comparable — this is a controlled comparison, and the query style is the thing held constant.

**Top-1, not mean-over-top-k.** The question is "does the index contain anything genuinely close
to this?", and averaging in the 5th-best match blunts exactly that signal.

**Network failure is soft.** `DriftSliceUnavailable` is raised rather than crashing, and
`run_retrieval_bench.py` catches it, prints why, falls back to a previously-committed
`drift.json` for the figure, and carries on. This mirrors how the API-key half of `run_eval.py`
already degrades — the same convention, so a reader learns it once.

## The non-obvious bits

- **The record carries `floor_relative_drop` and `trigger_excess`**, not just the verdict. A
  saved `drift.json` has to be readable without also knowing which version of `config.py`
  produced it — otherwise the correction described above would be invisible in the artifact it
  most needs to be visible in.
- **`_sentences` and `_eligible` are imported from `evalset`**, underscore-private names and
  all. Deliberate: the drift queries must be built exactly the way the benchmark queries are, or
  the baseline is not a baseline. Duplicating the logic would be the bug.
- **Both sides are sampled with the same seed and the same size** (`n = min(DRIFT_PAPERS,
  len(fresh))`), so neither side gets an accidental sample-size advantage.
- **`except Exception`** on the fetch is intentionally broad — network, timeout, arXiv
  rate-limit, malformed feed all mean the same thing to the caller: no slice today.

## How it can break

- **The floor is a constant someone has to maintain.** Change the embedding model, the chunk
  size or `MAX_PAPERS` and `DRIFT_FLOOR_RELATIVE_DROP` is silently wrong — the trigger keeps
  returning a verdict, just not a meaningful one. It is the same class of failure as the original
  bug, moved one level up, and the only defence is the comment in `config.py` saying so.
- **Two readings is not a trend.** The excess-over-floor rule is the right *shape* of test, but
  it has been evaluated at exactly two points in time. What would actually justify the tolerance
  is a series of probes across several weeks, which the project does not have and does not claim.
- **A corpus downloaded today has no drift to measure.** If arXiv returns nothing outside the
  existing corpus, `DriftSliceUnavailable` says so explicitly rather than reporting a
  meaningless 0%.
- **It measures topical coverage, not answer quality.** A similarity gap is evidence about how
  well the index covers current literature; it is not a direct measurement of how much worse
  users' answers have become. Closing that gap would need queries about the new papers *and* the
  new papers indexed.
