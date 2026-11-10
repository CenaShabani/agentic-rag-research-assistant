"""Benchmark the four retrieval arms and write the results the README reports.

    python run_retrieval_bench.py

Everything here is local: no API key, no paid call. The only network access is
the optional drift probe, which fails soft.

Three stages:

  1. **Arm comparison.** Dense, BM25, hybrid RRF and cross-encoder rerank, scored
     on the 120 generated known-item queries plus the 11 hand-written ones, with
     nDCG@10 / MAP / recall / MRR, bootstrap intervals, and per-arm latency.
  2. **The curated slice.** The same four arms on the 11 hand-written questions,
     reported separately and never pooled with the generated ones — they are
     different tasks with different label provenance (see eval/ADJUDICATION.md),
     and averaging them would produce a number that describes neither.
  3. **Drift probe.** How well the index still covers newer arXiv papers, and
     whether that crosses the re-indexing trigger. Skipped with a message if
     arXiv is unreachable and nothing is cached.

Results land in outputs/results/ and every figure is regenerated from them.
This is the counterpart to `run_eval.py`, which owns the naive-vs-agent
comparison; neither touches the other's outputs.
"""

from __future__ import annotations

import json

from src import config, drift, evaluation, figures, retrievers, significance
from src.vectorstore import VectorStore


def main() -> None:
    config.ensure_dirs()

    store = VectorStore.load()
    known_item = evaluation.load_known_item_questions()
    curated = evaluation.load_eval_set()

    print(f"Index: {store.size} chunks.")
    print(f"Questions: {len(known_item)} known-item + {len(curated)} curated.")
    print(f"Reporting at k={config.EVAL_K}.\n")

    arms = retrievers.build_all(store)
    print("Arms: " + ", ".join(a.name for a in arms))

    # --- 1. the generated known-item benchmark ----------------------------- #
    print("\nScoring arms on the known-item set (warming up first) ...")
    rows = evaluation.evaluate_arms(arms, known_item, k=config.EVAL_K)
    summary = evaluation.summarise_arms(rows, k=config.EVAL_K)

    with open(config.ARM_ROWS_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(config.ARM_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  {'arm':<8} {'nDCG@10':>18}  {'MAP':>8}  {'recall':>7}  {'median ms':>9}")
    for rec in summary:
        if rec["slice"] != "all":
            continue
        ci = f"[{rec['ndcg@k_lo']:.3f}, {rec['ndcg@k_hi']:.3f}]"
        print(f"  {rec['arm']:<8} {rec['ndcg@k']:>6.3f} {ci:>11}  "
              f"{rec['ap@k']:>8.3f}  {rec['recall@k']:>7.3f}  "
              f"{rec['latency_ms_median']:>9.1f}")

    # --- 2. the hand-written curated slice --------------------------------- #
    # Warm-up is off: the arms were warmed in stage 1 and warming again would
    # only cost time.
    print("\nScoring arms on the 11 curated questions (reported separately) ...")
    curated_rows = evaluation.evaluate_arms(arms, curated, k=config.EVAL_K,
                                            warmup=False)
    curated_summary = evaluation.summarise_arms(curated_rows, k=config.EVAL_K)
    with open(config.RESULTS_DIR / "arm_comparison_curated.json", "w",
              encoding="utf-8") as f:
        json.dump(curated_rows, f, indent=2)
    with open(config.RESULTS_DIR / "arm_summary_curated.json", "w",
              encoding="utf-8") as f:
        json.dump(curated_summary, f, indent=2)

    for rec in curated_summary:
        if rec["slice"] != "all":
            continue
        print(f"  {rec['arm']:<8} nDCG@10 {rec['ndcg@k']:.3f}  "
              f"recall {rec['recall@k']:.3f}  (n={rec['n_questions']})")

    # --- 3. is any of that gap real? --------------------------------------- #
    # Runs on the rows already scored above — no arm is searched twice. The
    # marginal intervals printed in stage 1 cannot answer "is A better than B",
    # because the arms share a question set and the paired difference is the
    # measurement that exists.
    print("\nPaired significance tests (sign-flip randomization, Holm-corrected) ...")
    sig = significance.compare_arms(rows)
    with open(config.SIGNIFICANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2)
    for rec in sig:
        mark = "*" if rec["significant"] else " "
        print(f"  {mark} {rec['arm_a']:>6} - {rec['arm_b']:<6} "
              f"{rec['mean_diff']:+.3f} "
              f"[{rec['mean_diff_lo']:+.3f}, {rec['mean_diff_hi']:+.3f}]  "
              f"p(Holm)={rec['p_value_holm']:.4f}  "
              f"({rec['n_wins_a']}W/{rec['n_wins_b']}L/{rec['n_ties']}T)")

    # --- 4. where should the cutoff be? ------------------------------------ #
    print("\nCutoff sweep (one search at k=50 per arm, truncated) ...")
    sweep = evaluation.sweep_k(arms, known_item, warmup=False)
    with open(config.K_SWEEP_FILE, "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)
    ks = sorted({r["k"] for r in sweep})
    print("    arm     " + "".join(f"{'r@'+str(k):>9}" for k in ks))
    for arm in dict.fromkeys(r["arm"] for r in sweep):
        by_k = {r["k"]: r["recall@k"] for r in sweep if r["arm"] == arm}
        print(f"    {arm:<8}" + "".join(f"{by_k[k]:>9.3f}" for k in ks))

    # --- 5. drift ---------------------------------------------------------- #
    drift_record: dict | None = None
    print("\nDrift probe (needs arXiv, or a cached slice) ...")
    try:
        drift_record = drift.measure_drift(store)
        with open(config.DRIFT_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(drift_record, f, indent=2)
        verdict = ("RE-INDEX" if drift_record["reindex_now"]
                   else "at the day-zero floor — nothing to re-index for")
        print(f"  top-1 similarity: indexed {drift_record['baseline_top1_mean']:.3f} "
              f"vs newer {drift_record['probe_top1_mean']:.3f} "
              f"({drift_record['relative_drop']:.1%} drop, "
              f"{drift_record['excess_over_floor']:+.1%} vs the "
              f"{drift_record['floor_relative_drop']:.1%} floor) -> {verdict}")
    except drift.DriftSliceUnavailable as exc:
        print(f"  Skipped drift probe: {exc}")
        prev = config.DRIFT_RESULTS_FILE
        if prev.exists():
            with open(prev, encoding="utf-8") as f:
                drift_record = json.load(f)
            print("  Loaded previously-committed drift.json for figures.")

    # --- figures ----------------------------------------------------------- #
    print("\nRegenerating benchmark figures ...")
    figures.benchmark_figures(summary, rows, drift_record,
                              significance_records=sig, k_sweep_rows=sweep)
    print(f"Done. Results in {config.RESULTS_DIR}, figures in {config.FIGURES_DIR}.")


if __name__ == "__main__":
    main()
