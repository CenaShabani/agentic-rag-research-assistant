"""The drift probe — when does this index need rebuilding?

Every retrieval number in this project describes an index built on 2026-07-27.
An index is a photograph of a corpus, and arXiv publishes hundreds of cs.CL and
cs.LG papers a week. The measured quality starts decaying the moment the build
finishes, and nothing in a notebook notices.

This module measures that decay and turns it into a **re-indexing trigger** — the
one operational question a benchmark like this should be able to answer.

## What is measured, and why not the obvious thing

The obvious probe is: take queries about newer papers, run them, and watch the
scores fall. That does not work, and it is worth being clear why. Those papers
are not in the index, so no retriever can possibly return them — every arm scores
zero, and zero is not a measurement of drift, it is a measurement of the setup.

What *can* be measured is whether the index still has anything close to the new
material. So the probe compares two similarity distributions, using the dense arm
because it is the only one with a calibrated, comparable notion of distance:

    baseline : queries drawn from *indexed* papers  -> mean top-1 cosine
    probe    : queries drawn from *newer, unseen* papers -> mean top-1 cosine

The baseline is a ceiling — those queries were extracted from documents sitting
in the index, so their best match should be near-perfect. The probe asks how well
the existing corpus covers what the field is talking about now. Some gap is
normal and permanent: an index cannot contain the future, and this is not a bug
to be tuned away. The gap *widening* over successive probes is the signal.

## The trigger — and the version of it that was wrong

    relative_drop = (baseline_mean - probe_mean) / baseline_mean

The first version of this module triggered a re-index when `relative_drop`
exceeded a flat 15%. It read 27.2% on 2026-07-27 and duly recommended rebuilding.

**Acting on that recommendation is what showed the trigger was broken.** The
index was rebuilt on 2026-08-12 against a corpus sharing not one paper with the
old one, and the same probe on the one-hour-old index read **26.7%**. The largest
possible freshness intervention moved the metric by half a point.

The reason is in the paragraph above, and this module always half-said it: some
gap is normal and permanent, and *the widening is the signal*. The flat threshold
tested the level anyway. The level is dominated by a structural floor — a
baseline query is a sentence taken verbatim from a document in the index, so its
top-1 match is near-perfect by construction; a probe query's document is absent
by definition. That gap exists on day zero and is not staleness.

So the trigger now tests the **excess over a measured day-zero floor**:

    excess = relative_drop - config.DRIFT_FLOOR_RELATIVE_DROP

Re-index when `excess` exceeds `config.DRIFT_TRIGGER_EXCESS_DROP`. The floor is a
number someone had to go and measure on a fresh index, and it is specific to this
corpus, embedder and window — which is the honest cost of the metric meaning
anything. Both thresholds are business decisions informed by a metric, not
properties of the data.

## Network

This is the one part of the project that reaches the internet at evaluation time.
It degrades exactly the way the API-key half does: if the fetch fails, fall back
to a previously-downloaded slice; if there is none, skip and say so. Nothing else
in the benchmark depends on it.
"""

from __future__ import annotations

import numpy as np

from . import config
from .acquisition import Paper, load_papers, save_papers
from .evalset import _sentences, _eligible
from .vectorstore import VectorStore


class DriftSliceUnavailable(RuntimeError):
    """No newer arXiv slice could be fetched or loaded from cache."""


def fetch_drift_slice(known_ids: set[str],
                      n_papers: int = config.DRIFT_PAPERS) -> list[Paper]:
    """Download recent arXiv papers the index has never seen.

    Reuses `acquisition.fetch_papers`, which already pages politely and sorts by
    submission date descending — so "newest first, minus everything we already
    have" is the natural slice. Over-fetches, because the newest page still
    overlaps heavily with a corpus downloaded days ago and only the non-overlap
    is usable.
    """
    from .acquisition import fetch_papers

    # Ask for enough that, after removing papers already in the corpus, there is
    # still a usable sample left.
    candidates = fetch_papers(max_papers=n_papers * 4)
    fresh = [p for p in candidates if p.arxiv_id not in known_ids and _eligible(p)]
    return fresh[:n_papers]


#: A cached slice is only usable if enough of it is still outside the index. Below
#: this fraction of `config.DRIFT_PAPERS` the cache is discarded and re-fetched:
#: too few probe queries and the mean top-1 similarity is noise, and the trigger
#: would fire (or fail to) on a handful of papers.
MIN_USABLE_SLICE_FRACTION = 0.5


def load_or_fetch_drift_slice(known_ids: set[str]) -> list[Paper]:
    """Return a drift slice that is genuinely *outside* the index, fetching if not.

    Cached deliberately: the probe should not re-download on every run, and a
    committed run's numbers should be re-derivable from the same bytes that
    produced them. Network failure falls back to the cache and only raises when
    there is nothing to fall back to.

    **The cache must be re-checked against the index, not just reused.** A slice
    is "newer, unseen papers" only relative to the index it was fetched against.
    Re-index — which is exactly what this probe exists to trigger — and yesterday's
    probe slice is mostly *inside* the new corpus. Reusing it then compares the
    index against its own contents: top-1 similarity is near 1.0 on both sides,
    the drop collapses toward zero, and the probe reports a healthy index for the
    same reason a thermometer in your pocket reports a fever. The failure is
    silent and self-congratulatory, which is the worst kind, so the filter is
    applied on the *load* path and not only on the fetch path.
    """
    if config.DRIFT_PAPERS_FILE.exists():
        cached = [p for p in load_papers(config.DRIFT_PAPERS_FILE)
                  if p.arxiv_id not in known_ids]
        if len(cached) >= config.DRIFT_PAPERS * MIN_USABLE_SLICE_FRACTION:
            return cached[:config.DRIFT_PAPERS]
        # Otherwise fall through and re-fetch: the index has caught up with this
        # slice, so it can no longer measure anything.

    try:
        fresh = fetch_drift_slice(known_ids)
    except Exception as exc:  # network, timeout, arXiv rate-limit
        raise DriftSliceUnavailable(f"could not fetch a newer arXiv slice: {exc}") from exc

    if not fresh:
        raise DriftSliceUnavailable(
            "arXiv returned no papers outside the existing corpus — "
            "the corpus is probably too recent for a drift probe to say anything."
        )
    save_papers(fresh, config.DRIFT_PAPERS_FILE, config.DRIFT_PROVENANCE_FILE)
    return fresh


def _top1_similarities(store: VectorStore, queries: list[str]) -> np.ndarray:
    """Best cosine similarity in the index for each query.

    Top-1 rather than a mean over top-k: the question is "does the index contain
    anything genuinely close to this?", and averaging in the 5th-best match
    blunts exactly the signal being looked for.
    """
    sims = []
    for q in queries:
        rows = store.search_rows(q, k=1)
        sims.append(rows[0][1] if rows else 0.0)
    return np.asarray(sims, dtype=float)


def _queries_from(papers: list[Paper], n: int, seed: int) -> list[str]:
    """One abstract-sentence query per paper, sampled deterministically.

    Abstract sentences rather than titles, because a title is short and unusually
    distinctive; a mid-abstract sentence is a fairer probe of whether the index
    covers the *content* of the new literature.
    """
    import random

    rng = random.Random(seed)
    pool = sorted((p for p in papers if _eligible(p)), key=lambda p: p.arxiv_id)
    sampled = rng.sample(pool, min(n, len(pool)))
    return [_sentences(p.abstract)[0] for p in sampled]


def measure_drift(store: VectorStore) -> dict:
    """Compare index coverage of indexed vs newer papers. Returns one record.

    Raises `DriftSliceUnavailable` when there is no newer slice to compare
    against — the caller decides whether that is fatal (it is not; `run_retrieval_bench.py`
    reports it and carries on).
    """
    indexed = load_papers()
    known_ids = {p.arxiv_id for p in indexed}
    fresh = load_or_fetch_drift_slice(known_ids)

    n = min(config.DRIFT_PAPERS, len(fresh))
    baseline_queries = _queries_from(indexed, n, seed=config.SEED)
    probe_queries = _queries_from(fresh, n, seed=config.SEED)

    baseline = _top1_similarities(store, baseline_queries)
    probe = _top1_similarities(store, probe_queries)

    baseline_mean = float(baseline.mean())
    probe_mean = float(probe.mean())
    relative_drop = (
        (baseline_mean - probe_mean) / baseline_mean if baseline_mean else 0.0
    )

    excess = relative_drop - config.DRIFT_FLOOR_RELATIVE_DROP

    return {
        "n_baseline_queries": int(baseline.size),
        "n_probe_queries": int(probe.size),
        "baseline_top1_mean": baseline_mean,
        "baseline_top1_median": float(np.median(baseline)),
        "probe_top1_mean": probe_mean,
        "probe_top1_median": float(np.median(probe)),
        "relative_drop": float(relative_drop),
        # The day-zero reading this drop is interpreted against, carried into the
        # record so a saved result is readable without also knowing which version
        # of config.py produced it.
        "floor_relative_drop": config.DRIFT_FLOOR_RELATIVE_DROP,
        "excess_over_floor": float(excess),
        "trigger_excess": config.DRIFT_TRIGGER_EXCESS_DROP,
        "reindex_now": bool(excess > config.DRIFT_TRIGGER_EXCESS_DROP),
    }
