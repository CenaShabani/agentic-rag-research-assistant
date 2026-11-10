"""Is the chunk size this project argued for worth anything?

    python run_ablation.py

Local, keyless, ~10 minutes on a CPU laptop (it re-embeds the whole corpus once
per setting).

## Why this one constant

`config.CHUNK_SIZE_CHARS = 1000` carries a paragraph of justification — too large
and one vector has to average several ideas, too small and a sentence loses the
context that made it meaningful — and not one measurement. It is the only
build-time decision that changes *both* retrieval arms at once, and for different
reasons:

  * **dense** — chunk size sets how much meaning one vector has to carry. This is
    the dilution argument the constant was justified with.
  * **BM25** — chunk size *is* document length, which is the denominator of the
    length normalisation (`config.BM25_B`). Longer chunks match more query terms
    and are penalised for it. The constant's own comment notes that b "does
    comparatively little here" because chunks are near-uniform in length; that
    claim is only true at one setting, and this is the run that tests it.

If the two arms peak at different sizes, then 1000 is a retriever-specific choice
the project made once for both — a more useful finding than "1000 was fine".

## What is deliberately left out

The cross-encoder. It costs ~2.6 s a query, so a third arm would add roughly 4
hours to a run whose whole point is the *first-stage* representation. The
reranker rescores whatever the first stage hands it; it is not the thing chunk
size acts on. Stated here rather than left to be noticed.

The overlap is held at 15% of the chunk size rather than fixed in characters, so
each setting keeps the same proportional margin. Varying both would confound
them: a change in score could be the window or the margin and nothing would say
which.
"""

from __future__ import annotations

import json
import time

import numpy as np

from src import acquisition, chunking, config, evaluation, figures, significance
from src.lexical import BM25
from src.retrievers import BM25Retriever, DenseRetriever
from src.vectorstore import VectorStore

#: The sizes swept, in characters. 1000 is the shipped constant and sits in the
#: middle by design — a sweep that only looks upward from the current value can
#: only ever confirm it. 400 is roughly two sentences of an abstract; 2000 is most
#: of a short one, which is where the dilution argument should start to bite.
CHUNK_SIZES = (400, 700, 1000, 1400, 2000)

#: Overlap as a fraction of chunk size, matching the shipped 150/1000.
OVERLAP_FRACTION = 0.15


def build_store(papers: list, size: int) -> tuple[VectorStore, int]:
    """Chunk at `size`, embed, and build an in-memory index. Returns (store, n).

    Nothing is written to `data/` — an ablation must not leave the shipped index
    replaced by whichever setting happened to run last. Every store here lives
    and dies inside this process.
    """
    overlap = int(round(size * OVERLAP_FRACTION))
    chunks = chunking.chunk_papers(papers, size=size, overlap=overlap)
    frame = chunking.chunks_to_frame(chunks)
    return VectorStore.build(frame), len(chunks)


def main() -> None:
    config.ensure_dirs()
    papers = acquisition.load_papers()
    questions = evaluation.load_known_item_questions()
    print(f"{len(papers)} papers, {len(questions)} known-item queries, "
          f"reporting at k={config.EVAL_K}.")
    print(f"Sweeping chunk sizes {CHUNK_SIZES} at {OVERLAP_FRACTION:.0%} overlap.\n")

    records: list[dict] = []
    per_question: dict[int, list[dict]] = {}
    for size in CHUNK_SIZES:
        t0 = time.time()
        store, n_chunks = build_store(papers, size)
        arms = [DenseRetriever(store), BM25Retriever(store)]

        rows = evaluation.evaluate_arms(arms, questions, k=config.EVAL_K)
        per_question[size] = rows
        for arm in arms:
            scores = [r["ndcg@k"] for r in rows if r["arm"] == arm.name]
            recalls = [r["recall@k"] for r in rows if r["arm"] == arm.name]
            lo, hi = evaluation.bootstrap_ci(scores)
            r_lo, r_hi = evaluation.bootstrap_ci(recalls)
            records.append({
                "chunk_size": size,
                "overlap": int(round(size * OVERLAP_FRACTION)),
                "arm": arm.name,
                "n_chunks": n_chunks,
                "mean_chunk_chars": float(np.mean(
                    [len(t) for t in store.meta["text"]])),
                "n_questions": len(scores),
                "k": config.EVAL_K,
                "ndcg@k": float(np.mean(scores)),
                "ndcg@k_lo": lo,
                "ndcg@k_hi": hi,
                "recall@k": float(np.mean(recalls)),
                "recall@k_lo": r_lo,
                "recall@k_hi": r_hi,
            })

        marks = "  ".join(
            f"{r['arm']} {r['ndcg@k']:.3f}" for r in records[-2:]
        )
        print(f"  {size:>5} chars -> {n_chunks:>5} chunks  {marks}  "
              f"({time.time() - t0:.0f}s)")

    with open(config.CHUNK_ABLATION_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    figures.chunk_ablation_figure(records)

    # --- is any of that movement real? ------------------------------------- #
    # Same discipline the arm comparison uses: every setting was scored on the
    # same questions, so the per-question difference is the measurement, and
    # reading two overlapping intervals is not. Each setting is compared against
    # the shipped one, with the family being "all sizes vs the default, per arm".
    print("\nPaired against the shipped "
          f"{config.CHUNK_SIZE_CHARS}-char setting (sign-flip, Holm-corrected):")
    tests: list[dict] = []
    for arm in ("dense", "bm25"):
        shipped_rows = {r["qid"]: r["ndcg@k"]
                        for r in per_question[config.CHUNK_SIZE_CHARS]
                        if r["arm"] == arm}
        arm_tests = []
        for size in CHUNK_SIZES:
            if size == config.CHUNK_SIZE_CHARS:
                continue
            other = {r["qid"]: r["ndcg@k"] for r in per_question[size]
                     if r["arm"] == arm}
            qids = sorted(shipped_rows)
            diffs = np.array([other[q] - shipped_rows[q] for q in qids])
            lo, hi = evaluation.bootstrap_ci(diffs.tolist())
            arm_tests.append({
                "arm": arm,
                "chunk_size": size,
                "vs_chunk_size": config.CHUNK_SIZE_CHARS,
                "n_questions": int(diffs.size),
                "mean_diff": float(diffs.mean()),
                "mean_diff_lo": lo,
                "mean_diff_hi": hi,
                "n_better": int((diffs > 0).sum()),
                "n_worse": int((diffs < 0).sum()),
                "n_ties": int((diffs == 0).sum()),
                "p_value": significance.randomization_p_value(diffs),
            })
        for test, p_adj in zip(arm_tests,
                               significance.holm_adjust(
                                   [t["p_value"] for t in arm_tests])):
            test["p_value_holm"] = p_adj
            test["significant"] = bool(p_adj < 0.05)
            mark = "*" if test["significant"] else " "
            print(f"  {mark} {arm:<6} {test['chunk_size']:>5} vs "
                  f"{config.CHUNK_SIZE_CHARS}: {test['mean_diff']:+.3f} "
                  f"[{test['mean_diff_lo']:+.3f}, {test['mean_diff_hi']:+.3f}]  "
                  f"p(Holm)={test['p_value_holm']:.4f}  "
                  f"({test['n_better']}B/{test['n_worse']}W/{test['n_ties']}T)")
        tests.extend(arm_tests)

    with open(config.ABLATION_SIGNIFICANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2)

    # The comparison the run exists to make: does each arm want the same size?
    print("\nBest chunk size by arm (nDCG@10):")
    for arm in ("dense", "bm25"):
        best = max((r for r in records if r["arm"] == arm),
                   key=lambda r: r["ndcg@k"])
        shipped = next(r for r in records
                       if r["arm"] == arm and r["chunk_size"] == config.CHUNK_SIZE_CHARS)
        print(f"  {arm:<6} best {best['chunk_size']:>5} chars "
              f"({best['ndcg@k']:.3f})   shipped {shipped['chunk_size']} chars "
              f"({shipped['ndcg@k']:.3f})   "
              f"cost of the default: {shipped['ndcg@k'] - best['ndcg@k']:+.3f}")

    print(f"\nDone. Results in {config.CHUNK_ABLATION_FILE.name}, "
          f"figure 10_chunk_ablation.png.")


if __name__ == "__main__":
    main()
