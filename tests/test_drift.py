"""The drift probe's one job is to compare the index against papers it has *not* seen.

Everything else in the probe is arithmetic. The failure mode worth pinning is the
one that is silent: reusing a cached probe slice after a re-index, when that slice
has since been absorbed into the corpus. The probe would then measure the index
against its own contents, report a near-zero drop, and pronounce a stale index
healthy — the same shape of bug as a test that passes because it asserts nothing.
"""

from __future__ import annotations

import pytest

from src import config, drift
from src.acquisition import Paper


def _paper(arxiv_id: str) -> Paper:
    """A minimally-populated Paper — only the id matters to the slice filter."""
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="A sentence long enough to survive the eligibility filter used "
                 "when sampling probe queries from abstracts.",
        authors=["A Author"],
        categories=["cs.CL"],
        published="2026-08-01T00:00:00Z",
        license=None,
    )


@pytest.fixture
def cached_slice(monkeypatch, tmp_path):
    """Point the drift module at a temporary cached slice, and record re-fetches.

    Returns a `calls` list the test can inspect: a re-fetch appends to it, so
    "did the loader go back to arXiv?" is directly observable rather than
    inferred from the returned papers.
    """
    path = tmp_path / "papers_drift.jsonl"
    monkeypatch.setattr(config, "DRIFT_PAPERS_FILE", path)
    monkeypatch.setattr(config, "DRIFT_PROVENANCE_FILE", tmp_path / "provenance_drift.json")
    monkeypatch.setattr(config, "DRIFT_PAPERS", 10)

    calls: list[set[str]] = []

    def fake_fetch(known_ids, n_papers=config.DRIFT_PAPERS):
        calls.append(set(known_ids))
        return [_paper(f"9999.{i:04d}") for i in range(10)]

    monkeypatch.setattr(drift, "fetch_drift_slice", fake_fetch)
    return path, calls


def _write_slice(path, ids: list[str]) -> None:
    from src.acquisition import save_papers

    save_papers([_paper(i) for i in ids], path, path.parent / "provenance_drift.json")


def test_fresh_cached_slice_is_reused(cached_slice):
    """A slice still outside the index is served from cache — no network call."""
    path, calls = cached_slice
    _write_slice(path, [f"2608.{i:04d}" for i in range(10)])

    got = drift.load_or_fetch_drift_slice(known_ids={"1706.03762"})

    assert len(got) == 10
    assert calls == [], "a usable cached slice must not trigger a re-fetch"


def test_absorbed_cached_slice_triggers_refetch(cached_slice):
    """After a re-index the cache is inside the corpus, so it must be replaced.

    This is the regression the re-index of 2026-08-12 exposed: the loader
    returned the cached slice whenever the file existed, without ever re-checking
    it against the index it was about to be compared with.
    """
    path, calls = cached_slice
    absorbed = [f"2608.{i:04d}" for i in range(10)]
    _write_slice(path, absorbed)

    got = drift.load_or_fetch_drift_slice(known_ids=set(absorbed))

    assert calls, "a cached slice fully contained in the index must be re-fetched"
    assert all(p.arxiv_id not in absorbed for p in got)


def test_partially_absorbed_slice_refetches_below_the_floor(cached_slice):
    """Half the slice absorbed is still usable; nearly all of it is not."""
    path, calls = cached_slice
    ids = [f"2608.{i:04d}" for i in range(10)]
    _write_slice(path, ids)

    # 5 of 10 survive — exactly at the floor, so the cache is kept.
    drift.load_or_fetch_drift_slice(known_ids=set(ids[:5]))
    assert calls == []

    # 8 of 10 absorbed — below the floor, so it is replaced.
    drift.load_or_fetch_drift_slice(known_ids=set(ids[:8]))
    assert len(calls) == 1


def test_refetch_is_told_what_the_index_already_has(cached_slice):
    """The re-fetch must pass the index's ids through, or it re-downloads them."""
    path, calls = cached_slice
    ids = [f"2608.{i:04d}" for i in range(10)]
    _write_slice(path, ids)

    drift.load_or_fetch_drift_slice(known_ids=set(ids))

    assert calls[0] == set(ids)


# --------------------------------------------------------------------------- #
# The trigger: excess over the day-zero floor, not the raw drop
# --------------------------------------------------------------------------- #
@pytest.fixture
def probe_at(monkeypatch):
    """Run `measure_drift` with a chosen baseline/probe similarity pair.

    Everything upstream of the arithmetic — the corpus, the index, the sampled
    queries — is replaced, because the thing under test is the decision rule.
    """
    import numpy as np

    def run(baseline: float, probe: float) -> dict:
        monkeypatch.setattr(drift, "load_papers", lambda *a, **k: [_paper("1111.0001")])
        monkeypatch.setattr(drift, "load_or_fetch_drift_slice",
                            lambda known_ids: [_paper("9999.0001")])
        monkeypatch.setattr(drift, "_queries_from", lambda papers, n, seed: ["q"] * 3)

        # measure_drift scores the baseline queries first, then the probe queries.
        sims = iter([baseline, probe])
        monkeypatch.setattr(
            drift, "_top1_similarities",
            lambda store, queries: np.full(len(queries), next(sims)),
        )
        return drift.measure_drift(store=None)

    return run


def test_a_drop_at_the_floor_does_not_trigger(probe_at):
    """The reading that exposed the bug: a fresh index sits at the floor.

    On 2026-08-12 a one-hour-old index read a 26.7% drop. Under the original flat
    15% rule that recommended an immediate re-index of an index built that hour.
    """
    rec = probe_at(baseline=1.0, probe=1.0 - config.DRIFT_FLOOR_RELATIVE_DROP)

    assert rec["relative_drop"] == pytest.approx(config.DRIFT_FLOOR_RELATIVE_DROP)
    assert rec["excess_over_floor"] == pytest.approx(0.0, abs=1e-9)
    assert rec["reindex_now"] is False


def test_a_drop_well_past_the_floor_triggers(probe_at):
    """Genuine widening — floor plus more than the tolerance — still fires."""
    drop = config.DRIFT_FLOOR_RELATIVE_DROP + config.DRIFT_TRIGGER_EXCESS_DROP + 0.05
    rec = probe_at(baseline=1.0, probe=1.0 - drop)

    assert rec["excess_over_floor"] > config.DRIFT_TRIGGER_EXCESS_DROP
    assert rec["reindex_now"] is True


def test_record_carries_the_floor_it_was_judged_against(probe_at):
    """A saved drift.json must be readable without the config that produced it."""
    rec = probe_at(baseline=1.0, probe=0.5)

    assert rec["floor_relative_drop"] == config.DRIFT_FLOOR_RELATIVE_DROP
    assert rec["trigger_excess"] == config.DRIFT_TRIGGER_EXCESS_DROP
