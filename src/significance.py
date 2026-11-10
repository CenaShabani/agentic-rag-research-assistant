"""Is the gap between two arms real, or is it the question set?

`evaluation.summarise_arms` gives every arm a bootstrap interval of its own, and
the README then prints four of them in a table. That invites a comparison the
intervals cannot support. **Two marginal intervals overlapping does not mean the
difference is not significant** — it is a well-known trap, and here it would be a
self-inflicted one, because the arms all answered the *identical* question list.
When the same questions go to every arm, the per-question *difference* is
available, and it is a far tighter measurement than either arm's spread:

    arm A on q17: 0.63     arm B on q17: 0.50     difference: +0.13
    arm A on q18: 0.99     arm B on q18: 0.86     difference: +0.13

Both arms vary hugely across questions (0.63 vs 0.99 — some questions are simply
harder), and that variance is what widens the marginal intervals. It cancels
exactly in the difference. Pairing is not a refinement here; it is the difference
between a test that can see the effect and one that cannot.

## What this module reports, per pair of arms

1. **Mean difference in nDCG@10**, with a percentile bootstrap interval over the
   *differences* — the effect size, in the metric's own units. This is the number
   worth quoting: "BM25 beats dense by 0.23 [0.17, 0.28] nDCG".
2. **A two-sided paired randomization test.** Under the null "the two arms are
   interchangeable", the sign of each question's difference is arbitrary, so
   flipping signs at random generates the null distribution directly. It is the
   standard significance test in IR (Smucker et al., 2007, found it and the
   bootstrap agree closely and both beat the t-test's assumptions), and it needs
   nothing from the data but exchangeability — no normality, no equal variance.
3. **A Holm-corrected p-value.** Four arms is six pairs; five would be ten. Ten
   independent tests at 0.05 give a ~40% chance of at least one false positive,
   so an uncorrected table of pairwise wins is a table of some wins and some
   noise. Holm is used rather than Bonferroni because it is uniformly more
   powerful at the same guarantee, and the family is stated explicitly: all
   pairwise comparisons within one run, on one metric.

## What it deliberately does not do

No test is run *across* the two question sets. The generated known-item queries
and the 11 hand-written ones are different tasks with different label provenance
(`eval/ADJUDICATION.md`), and pooling them to grow n would buy significance by
mixing populations — the exact move this module exists to make unnecessary.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import config


#: Sign-flip resamples for the randomization test. With 180 questions the exact
#: test would need 2**180 flips, so it is sampled; 10,000 gives a p-value stable
#: to about ±0.004 at p = 0.05, which is finer than any decision made from it.
N_PERMUTATIONS = 10_000


def paired_differences(rows: list[dict], arm_a: str, arm_b: str,
                       metric: str = "ndcg@k") -> tuple[np.ndarray, list[str]]:
    """Per-question `metric` differences (arm_a − arm_b), aligned by question id.

    Returns the difference vector and the qids it came from, in a fixed order.
    Alignment is by qid rather than by position: the benchmark writes rows arm by
    arm and nothing guarantees the two blocks are in the same order, and a paired
    test on mis-paired rows is worse than no test — it looks fine and means
    nothing.
    """
    df = pd.DataFrame(rows)
    a = df[df.arm == arm_a].set_index("qid")[metric]
    b = df[df.arm == arm_b].set_index("qid")[metric]

    common = sorted(set(a.index) & set(b.index))
    if len(common) != len(a) or len(common) != len(b):
        raise ValueError(
            f"{arm_a} and {arm_b} were not scored on the same questions "
            f"({len(a)} vs {len(b)}, {len(common)} shared) — the pairing that "
            "makes this test valid does not hold."
        )
    diffs = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
    return diffs, common


def randomization_p_value(diffs: np.ndarray,
                          n_permutations: int = N_PERMUTATIONS,
                          seed: int = config.SEED) -> float:
    """Two-sided paired randomization (sign-flip) test p-value.

    The null is that the two arms are interchangeable, so for any question the
    observed difference was as likely to have come out with the opposite sign.
    Flipping signs uniformly at random therefore samples the null distribution of
    the mean difference, with no distributional assumption at all.

    The +1 in both numerator and denominator is the standard correction: the
    observed arrangement is itself one of the possible sign assignments, and
    excluding it can return p = 0, which is a claim no finite resample supports.
    """
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        return 1.0
    observed = abs(float(diffs.mean()))

    rng = np.random.default_rng(seed)
    # One (n_permutations, n) matrix of ±1, then a row-wise mean: the whole test
    # is two array operations.
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, diffs.size))
    null_means = np.abs((signs * diffs).mean(axis=1))
    n_at_least = int((null_means >= observed - 1e-12).sum())
    return (n_at_least + 1) / (n_permutations + 1)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, returned in the input order.

    Sort ascending, multiply the i-th smallest by (m − i), then enforce
    monotonicity so an adjusted p-value can never fall below one that preceded
    it. Controls the family-wise error rate at the same level as Bonferroni while
    rejecting at least as much — there is no reason to prefer Bonferroni here.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)   # step-down monotonicity
        adjusted[idx] = running
    return adjusted


def compare_arms(rows: list[dict], metric: str = "ndcg@k",
                 alpha: float = 0.05) -> list[dict]:
    """Every pairwise arm comparison, with effect size, interval and p-value.

    One record per unordered pair, ordered so `mean_diff` is positive and `arm_a`
    is the winner — a table where every row reads "A beat B by X" needs no sign
    convention explained to it.
    """
    from .evaluation import bootstrap_ci

    df = pd.DataFrame(rows)
    arms = [a for a in df["arm"].unique()]

    records: list[dict] = []
    for arm_a, arm_b in combinations(arms, 2):
        diffs, qids = paired_differences(rows, arm_a, arm_b, metric)
        if diffs.mean() < 0:                      # orient so arm_a wins
            arm_a, arm_b, diffs = arm_b, arm_a, -diffs

        lo, hi = bootstrap_ci(diffs.tolist())
        records.append({
            "arm_a": arm_a,
            "arm_b": arm_b,
            "metric": metric,
            "n_questions": int(diffs.size),
            "mean_diff": float(diffs.mean()),
            "mean_diff_lo": lo,
            "mean_diff_hi": hi,
            "n_wins_a": int((diffs > 0).sum()),
            "n_wins_b": int((diffs < 0).sum()),
            "n_ties": int((diffs == 0).sum()),
            "p_value": randomization_p_value(diffs),
        })

    adjusted = holm_adjust([r["p_value"] for r in records])
    for record, p_adj in zip(records, adjusted):
        record["p_value_holm"] = p_adj
        record["significant"] = bool(p_adj < alpha)
        # An interval on the difference that excludes zero and a significant test
        # are two routes to the same conclusion. They are reported together
        # because when they disagree - which happens near the boundary - that
        # disagreement is information, not something to resolve by picking one.
        record["ci_excludes_zero"] = bool(record["mean_diff_lo"] > 0)

    records.sort(key=lambda r: -r["mean_diff"])
    return records


def significance_table(records: list[dict]) -> pd.DataFrame:
    """The comparison records as a printable frame, for the notebook.

    Column labels are ASCII on purpose. A Windows console defaults to cp1252 and
    raises on a delta or a minus sign, so a table that renders in Jupyter would
    crash the same code at a terminal — and this project's numbers are meant to
    be checkable from either.
    """
    return pd.DataFrame([
        {
            "comparison": f"{r['arm_a']} vs {r['arm_b']}",
            "diff nDCG@10": round(r["mean_diff"], 3),
            "95% CI": f"[{r['mean_diff_lo']:.3f}, {r['mean_diff_hi']:.3f}]",
            "wins/losses/ties": f"{r['n_wins_a']}/{r['n_wins_b']}/{r['n_ties']}",
            "p (Holm)": f"{r['p_value_holm']:.4f}",
            "significant": "yes" if r["significant"] else "no",
        }
        for r in records
    ])
