# Snapshot — the index built 2026-07-27

Every number the project published before the first re-index, frozen here so the before/after is
a **file comparison rather than a claim**.

## Why it exists

The drift probe on this index read a 27.2% top-1 similarity drop against a stated 15%
re-indexing trigger, and the README's own recommendation was "re-index now". Acting on that
rebuilds the corpus from a fresh arXiv pull, which moves every published number — so the old
ones had to be preserved as data or the "we acted on the trigger" story would be unfalsifiable.

## What is here

| File | What it was |
|---|---|
| `provenance.json` | The acquisition record for the old corpus: 2,012 papers, downloaded 2026-07-27, SHA-256 of `papers.jsonl` |
| `known_item_set.json` | The 180 generated queries, sampled from the **old** corpus — regenerated after the re-index because their target papers changed |
| `arm_summary.json` | Per-arm, per-slice means with bootstrap intervals and latency. The source of the old headline: BM25 nDCG@10 0.987 @ 2.9 ms vs dense 0.762 @ 22.8 ms |
| `arm_comparison.json` | One row per (arm, question) behind that summary |
| `arm_summary_curated.json`, `arm_comparison_curated.json` | The same on the 11 hand-written questions, reported separately and never pooled |
| `drift.json` | The probe that fired: baseline 0.730 vs probe 0.532, 27.2% drop, `reindex_now: true` |
| `retrieval_metrics.json` | The original single-hop vs multi-hop finding (recall 1.00 → 0.60) |

## How to read it against the current results

The current run lives one directory up in `outputs/results/`. The comparison worth making is
**directional, not absolute**: the two runs index different draws of recent arXiv papers, so a
number moving is expected and a *conclusion* moving is the finding. The 12 seeded seminal papers
(`config.SEED_PAPER_IDS`) are pinned by id and present in both corpora, which is what keeps the
11 curated questions comparable across the rebuild.
