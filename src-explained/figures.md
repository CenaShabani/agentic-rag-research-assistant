# `src/figures.py` — every figure, regenerated from results

## What it's for

Draws every figure the README and the deck use, from the saved evaluation results. **No figure in
this project is drawn by hand or lives only in a notebook cell.** `run_eval.py` calls these
straight after computing results, so a figure can never drift from the numbers it claims to show.

This is the same one-source-per-number rule that governs the Kaggle notebook (generated, not
copied) and the deck (images exported here, never redrawn in PowerPoint).

## The public surface

| Function | In | Writes |
|---|---|---|
| `retrieval_metrics_figure(retrieval_rows)` | `list[dict]` from `evaluate_retrieval` | `outputs/figures/01_retrieval_metrics.png` |
| `answer_quality_figure(answer_rows)` | `list[dict]` from `evaluate_answers` | groundedness/correctness comparison |
| `cost_tradeoff_figure(answer_rows)` | same | quality gain vs retrieval count |
| `all_figures(retrieval_rows, answer_rows)` | both | all of the above |
| `arm_comparison_figure(summary)` | `summarise_arms` output | `04_arm_comparison.png` — nDCG per arm with bootstrap error bars |
| `quality_vs_latency_figure(summary)` | same | `05_quality_vs_latency.png` — **the headline** |
| `query_style_figure(summary)` | same | `06_query_style_robustness.png` |
| `drift_figure(drift_record)` | `measure_drift` output | `07_drift.png` |
| `significance_figure(records)` | `significance.compare_arms` output | `08_arm_significance.png` |
| `k_sweep_figure(rows)` | `evaluation.sweep_k` output | `09_k_sweep.png` |
| `chunk_ablation_figure(rows)` | `run_ablation.py` output | `10_chunk_ablation.png` |
| `benchmark_figures(summary, rows, drift, significance, sweep)` | | all of the above except 10 |

Every function takes **rows of results**, not a store or a model. They cannot compute anything —
only render what evaluation already produced. That constraint is what makes drift impossible.

### The three benchmark figures, and what each was chosen to show

- **04** puts **error bars on every bar**. Without them it is four bars and a story; with them a
  reader can see for themselves whether the gaps survive a 180-question sample.
- **05 uses a log x-axis**, because the arms span three orders of magnitude in latency. On a
  linear axis the three cheap arms collapse into one another and the chart says only "the
  reranker is slow", which is not the finding. The finding is that BM25 sits alone in the
  top-left — best quality, lowest cost, dominating everything else.
- **06 is a line chart, not a grouped bar chart**, because the *slope* is the measurement.
  Grouped bars would show three heights per arm; lines show that dense falls off a cliff
  (−0.68) while BM25 stays flat (−0.04), which is the whole point.
- **07 draws two reference lines, and the *lower* one is the important one.** An earlier version
  drew only the trigger and read as a verdict on the index. It now draws the **day-zero floor** —
  what this probe reads on an index built an hour ago — because the distance from the floor is
  the only part of the drop that carries information. The figure had to carry the correction,
  not just the corrected number; see `drift.md`.
- **08 is a forest plot, not a bar chart.** The quantity is a *difference with an interval*, and
  the only question a reader has is whether each interval clears zero — so zero is the loudest
  line on the chart and the pairs Holm rejects are drawn hollow. Significance is a property of
  the comparison, not a footnote under it.
- **09 is two panels rather than two lines on one axis.** nDCG@k and recall@k answer different
  questions and share no useful scale: nDCG@k can fall as k grows, recall@k only climbs. The
  served cutoff (`TOP_K`) and the reported one (`EVAL_K`) are drawn in, since checking those two
  numbers is the reason the sweep exists.
- **10 marks the shipped chunk size.** The reader's question is "was the default any good", so
  the default has to be visible on the axis rather than inferred from a caption.

## The walkthrough

### `_lazy_plt` does three jobs in six lines

```python
matplotlib.use("Agg")
```

Sets the headless backend **before** `pyplot` is imported — order matters, after the import it has
no effect. "Agg" means figures are written to files and no window is ever opened, which is what
lets the pipeline run in CI, over SSH, or inside a scheduled job without hanging on a display that
isn't there.

The lazy import matches `embeddings.get_model` and `llm._get_client`: importing `src.figures`
costs nothing, so tests and tooling can import the package without matplotlib installed.

The `rcParams` block sets shared style — 120 dpi, 10pt, top and right spines off — in one place, so
all figures look like one set rather than three separately-styled charts.

### `retrieval_metrics_figure` is where the finding becomes visible

The grouping is the substance:

```python
groups = {"all": df, "single-hop": df[df.hop == "single"], "multi-hop": df[df.hop == "multi"]}
```

Plotting all three side by side is what makes the collapse legible: the "all" bars look
respectable, and only the single-hop vs multi-hop split shows recall falling from 1.00 to 0.60.
**A figure showing only the overall average would hide the entire finding** — the same trap the
metrics themselves guard against by reporting hit@k, recall@k and MRR together.

`sub[m].mean() if len(sub) else 0.0` guards an empty group, so a question set with no multi-hop
rows renders zeros rather than raising on an empty mean.

`ax.set_ylim(0, 1)` fixes the axis to the metric's true range. Letting matplotlib autoscale would
make a 0.60 bar look near-maximal whenever nothing scored higher — an honesty detail, not a
cosmetic one. Bounded metrics get bounded axes.

The bar offsets (`xi + i * width` with `width = 0.25`) are manual grouped-bar placement, with
`set_xticks` shifted by one width to centre the labels under each trio.

### Colours

`_NAIVE_COLOR` (grey) and `_AGENT_COLOR` (green) are module constants so naive and agentic are the
same colours in every figure. A reader learns the mapping once.

## The non-obvious bits

- **`plt.close(fig)`** after every save. Without it, figures accumulate in memory across a run and
  matplotlib eventually warns; in a long pipeline it is a real leak.
- **`fig.tight_layout()`** before saving stops labels being clipped at the figure edge — the
  usual cause of a chart that looks fine on screen and truncated in the PNG.
- **Filenames are numbered** (`01_…`) so the figures directory sorts in narrative order.
- **The `deck/` subdirectory** under `outputs/figures/` holds deck-sized variants, produced the
  same way. Deck images being exported files rather than PowerPoint drawings is what the deck
  standard's verification checks.

## How it can break

- **Passing rows from a stale results file** would draw a figure that disagrees with the current
  numbers. The functions take results as arguments precisely so the caller controls freshness —
  `run_eval.py` computes then draws in one pass.
- **Missing `answer_rows`** (no API key, so no answer evaluation) means `answer_quality_figure` and
  `cost_tradeoff_figure` have nothing to draw. This is the authored-but-not-executed half the
  README is explicit about.
- **A changed metric name** in evaluation output breaks the `metrics` list here silently — the
  column lookup would raise, which is the loud failure, but a *renamed* hop value would quietly
  produce an empty group and a zero bar.
- Figures are not covered by unit tests beyond smoke checks; the guarantee they rest on is that
  they can only render what evaluation computed.
