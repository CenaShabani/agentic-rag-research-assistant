# `src/significance.py` — is the gap between two arms real?

## What it's for

Takes the per-question scores the benchmark already produced and answers the one question the
summary table cannot: **did arm A actually beat arm B, or did this question set happen to
favour it?** It sits downstream of `evaluation.evaluate_arms` — it runs no searches of its own
and adds no cost to a benchmark run — and its output feeds `figures.significance_figure` and
the README's comparison table.

It exists because the project was one step away from a mistake it criticises elsewhere: printing
four confidence intervals side by side and letting the reader eyeball the overlap.

## The public surface

| Member | In | Out |
|---|---|---|
| `paired_differences(rows, arm_a, arm_b, metric)` | benchmark rows, two arm names | `(np.ndarray` of per-question differences, `list[str]` of qids in the same order`)` |
| `randomization_p_value(diffs, n_permutations, seed)` | `np.ndarray`, `int`, `int` | `float` in `(0, 1]` |
| `holm_adjust(p_values)` | `list[float]` | `list[float]`, same order, family-wise corrected |
| `compare_arms(rows, metric, alpha)` | benchmark rows | `list[dict]`, one per unordered pair, sorted by effect size |
| `significance_table(records)` | the records | `pd.DataFrame` for display |
| `N_PERMUTATIONS` | — | `10_000`, the sampled sign-flip count |

`compare_arms` is the entry point; the other three are the pieces, exposed so the tests can
check each one against a case worked out by hand.

## The idea: why pairing changes everything

Every arm answered the **same** question list. That is not a convenience, it is the whole
statistical opportunity. Consider two arms on two questions:

| | arm A | arm B | difference |
|---|--:|--:|--:|
| q17 | 0.63 | 0.50 | +0.13 |
| q18 | 0.99 | 0.86 | +0.13 |

Each arm's *own* scores swing from 0.63 to 0.99, because some questions are simply harder than
others. That question-difficulty variance is what makes each marginal confidence interval wide,
and it is entirely shared between the arms — so it **cancels exactly** in the difference column.
The unpaired view sees two noisy averages that might be the same. The paired view sees an arm
that won by 0.13 both times.

This is why "the intervals overlap, so the difference isn't significant" is wrong as a general
rule. Overlapping marginal intervals are consistent with a large, perfectly consistent paired
effect. The reverse implication does hold — non-overlapping intervals do imply a difference —
which is what makes the error easy to commit and hard to notice.

## The walkthrough

### `paired_differences` — align by question id, never by position

The benchmark writes its rows arm by arm, and nothing in the format guarantees the blocks are in
the same question order. Subtracting positionally would produce a difference vector that looks
completely normal and means nothing. So both arms are indexed by `qid`, intersected, and sorted.

The `ValueError` when the sets differ in size is deliberate and unforgiving. A quiet
inner-join here would silently answer a different question — "how do these arms compare on the
questions they happen to share" — while the caller believes it asked about all of them.

### `randomization_p_value` — the sign-flip test

The null hypothesis is that the two arms are **interchangeable**: on any given question, the
difference was as likely to come out negative as positive. If that is true, then flipping the
sign of each question's difference at random produces an equally likely arrangement of the same
data. Do that ten thousand times, record the mean each time, and you have built the null
distribution of the mean difference **from the data itself** — no normal distribution assumed,
no equal variances, no minimum sample size.

The p-value is then the fraction of those arrangements whose mean is at least as extreme as the
one actually observed.

Two implementation notes:

- **The whole test is two array operations.** A `(n_permutations, n)` matrix of ±1, multiplied
  elementwise by the differences, then a row-wise mean. No Python loop over resamples, which is
  what makes ten thousand permutations cost milliseconds.
- **`(n_at_least + 1) / (n_permutations + 1)`.** The observed arrangement *is* one of the
  possible sign assignments, so it belongs in the count. Without the correction the function can
  return exactly 0, which is a claim no finite number of resamples can support — the honest
  statement is "smaller than 1/10001", and the +1 makes the returned number say that.

This is the standard test in the IR literature (Smucker, Allan & Carterette, 2007, compared it
against the bootstrap, the t-test, the sign test and the Wilcoxon; the randomization test and
the bootstrap agreed closely, and the sign and Wilcoxon tests were found to be too weak because
they discard the *magnitude* of each difference).

### `holm_adjust` — paying for asking many questions

Four arms make six pairs; five arms make ten. Each test at α = 0.05 has its own 5% chance of a
false positive, so a table of ten uncorrected comparisons has roughly a 40% chance of containing
at least one arm pair that looks different and is not.

Holm sorts the p-values ascending and multiplies the i-th smallest by `(m − i)` — the largest
multiplier goes to the smallest p-value, and the penalty relaxes as it steps down. Bonferroni
would multiply *every* p-value by `m`; Holm controls the same family-wise error rate and rejects
at least as many hypotheses, so there is no reason to prefer Bonferroni.

**The monotonicity pass (`running = max(running, value)`) is not cosmetic.** Because the
multipliers shrink faster than the raw p-values grow, the step-down arithmetic can produce an
adjusted value *lower* than the one before it — a table where a weaker result appears more
significant than a stronger one. Dragging each value up to the running maximum is part of the
procedure, not a tidy-up. `test_holm_matches_the_hand_computation` pins the exact case where it
bites.

### `compare_arms` — assembling the table

Pairs are **oriented so `arm_a` is the winner**: if the mean difference comes out negative, the
arms and the sign are both flipped. Every row then reads "A beat B by X", and no reader has to
hold a sign convention in their head.

Each record carries both `significant` (the Holm-corrected test) and `ci_excludes_zero` (the
bootstrap interval on the difference). These are two routes to the same conclusion and they
usually agree. When they disagree — which happens near the boundary — that is information about
how marginal the result is, so both are reported rather than one being chosen.

## The non-obvious bits

- **`bootstrap_ci` is imported inside the function**, not at module scope. `evaluation` imports
  `agent`, which imports the LLM client; a top-level import here would drag that whole chain into
  a module that needs nothing but arithmetic.
- **The bootstrap resamples the *differences*, not each arm separately.** Resampling the arms
  independently would destroy the pairing and reproduce exactly the too-wide interval this module
  exists to avoid.
- **`n_wins / n_losses / n_ties` are reported alongside the mean.** They catch the case a mean
  hides: an arm that loses most questions narrowly and wins two enormously has a positive mean
  difference and is not, in any useful sense, better.
- **Ties are common here and are not a defect.** On a known-item set both arms frequently return
  the target at rank 1, giving a difference of exactly 0. The randomization test handles them
  correctly — a zero difference contributes nothing under any sign flip.
- **10,000 permutations, not exhaustive.** With 180 questions the exact test would need 2^180
  sign assignments. The sampling error at p ≈ 0.05 is about ±0.004, finer than any decision made
  from the number.

## How it can break

- **Pass rows from two different runs and the pairing is a fiction.** The qids would match, the
  arithmetic would succeed, and the result would compare arms across different indexes. Nothing
  in the data marks a run, so this is a caller's responsibility.
- **`compare_arms` on the curated 11 questions is technically valid and practically useless.**
  Eleven paired differences give the randomization test very little to work with; the minimum
  attainable p-value from an exhaustive sign-flip on 11 items is 2/2048 ≈ 0.001, but the
  *effect* has to be perfectly consistent to get there. The module does not stop you; the
  write-up reports the generated set and says why.
- **Never pool the two question sets to grow n.** They are different tasks with different label
  provenance (`eval/ADJUDICATION.md`), and combining them would buy significance by mixing
  populations.
- **`significant` is a decision, not a fact.** It depends on α, on the family the correction was
  applied over, and on the metric chosen. The record keeps `p_value`, `p_value_holm` and the
  interval so a reader can apply their own threshold.

## Where it's used

`run_retrieval_bench.py` stage 3, writing `outputs/results/arm_significance.json`;
`figures.significance_figure` draws it; the capstone notebook and README read the saved JSON.
Nothing in the *serving* path touches this module — it is measurement about the system, not part
of it.
