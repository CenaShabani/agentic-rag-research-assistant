"""Every figure in the README, regenerated from the evaluation results.

No figure in this project is drawn by hand or lives only in a notebook cell — each
one is produced here from the saved results, so it is always reproducible and can
never drift from the numbers it claims to show. `run_eval.py` calls these after
computing the results.
"""

from __future__ import annotations

import pandas as pd

from . import config

# A calm, consistent palette so the project's figures read as one set.
_NAIVE_COLOR = "#9aa5b1"
_AGENT_COLOR = "#2f6f4f"


def _lazy_plt():
    """Import matplotlib lazily and apply a clean default style."""
    import matplotlib

    matplotlib.use("Agg")  # headless: write files, never open a window
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def retrieval_metrics_figure(retrieval_rows: list[dict]) -> None:
    """Bar chart of hit@k / recall@k / MRR, overall and by hop type."""
    plt = _lazy_plt()
    df = pd.DataFrame(retrieval_rows)
    metrics = ["hit@k", "recall@k", "mrr"]

    groups = {"all": df, "single-hop": df[df.hop == "single"], "multi-hop": df[df.hop == "multi"]}
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(metrics))
    width = 0.25
    for i, (label, sub) in enumerate(groups.items()):
        vals = [sub[m].mean() if len(sub) else 0.0 for m in metrics]
        ax.bar([xi + i * width for xi in x], vals, width, label=label)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Retrieval quality (higher is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "01_retrieval_metrics.png")
    plt.close(fig)


def answer_quality_figure(answer_rows: list[dict]) -> None:
    """Naive vs agent, groundedness + correctness, split by hop type."""
    plt = _lazy_plt()
    df = pd.DataFrame(answer_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric in zip(axes, ["groundedness", "correctness"]):
        pivot = df.groupby(["hop", "system"])[metric].mean().unstack("system")
        pivot = pivot.reindex(columns=["naive", "agent"])
        pivot.plot(kind="bar", ax=ax, color=[_NAIVE_COLOR, _AGENT_COLOR], rot=0,
                   legend=(metric == "groundedness"))
        ax.set_title(metric)
        ax.set_xlabel("question type")
        ax.set_ylim(0, 5)
    axes[0].set_ylabel("judge score (1-5)")
    fig.suptitle("Answer quality: naive RAG vs agentic RAG")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "02_answer_quality.png")
    plt.close(fig)


def cost_tradeoff_figure(answer_rows: list[dict]) -> None:
    """The other side of the ledger: how many retrievals each system spends."""
    plt = _lazy_plt()
    df = pd.DataFrame(answer_rows)
    means = df.groupby("system")["n_retrievals"].mean().reindex(["naive", "agent"])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(means.index, means.values, color=[_NAIVE_COLOR, _AGENT_COLOR])
    ax.set_ylabel("avg. retrieval calls per question")
    ax.set_title("Cost of the agent's iterative retrieval")
    for i, v in enumerate(means.values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "03_cost_tradeoff.png")
    plt.close(fig)


def all_figures(retrieval_rows: list[dict], answer_rows: list[dict]) -> None:
    """Regenerate every README figure. Called by run_eval.py."""
    config.ensure_dirs()
    retrieval_metrics_figure(retrieval_rows)
    if answer_rows:
        answer_quality_figure(answer_rows)
        cost_tradeoff_figure(answer_rows)


# --------------------------------------------------------------------------- #
# The retrieval benchmark's figures (run_retrieval_bench.py)
# --------------------------------------------------------------------------- #
# One colour per arm, used consistently across every benchmark figure so a reader
# can track an arm from one chart to the next without re-reading the legend.
_ARM_COLORS = {
    "dense":  "#4c78a8",   # the shipped system
    "bm25":   "#9aa5b1",   # the baseline it has to beat
    "hybrid": "#e0a458",
    "rerank": "#2f6f4f",   # the most expensive arm
}
_ARM_ORDER = ["dense", "bm25", "hybrid", "rerank"]


def _ordered_arms(present: set) -> list[str]:
    """Arms in cost order, restricted to those actually measured."""
    return [a for a in _ARM_ORDER if a in present]


def arm_comparison_figure(summary: list[dict]) -> None:
    """nDCG@10 per arm with bootstrap confidence intervals.

    Error bars are the whole point: without them this is four bars and a story,
    and with them a reader can see for themselves whether the gaps survive the
    noise of a 120-question sample.
    """
    plt = _lazy_plt()
    df = pd.DataFrame(summary)
    df = df[df["slice"] == "all"].set_index("arm")
    arms = _ordered_arms(set(df.index))

    means = [df.loc[a, "ndcg@k"] for a in arms]
    # Matplotlib wants distances from the mean, not absolute bounds.
    lower = [df.loc[a, "ndcg@k"] - df.loc[a, "ndcg@k_lo"] for a in arms]
    upper = [df.loc[a, "ndcg@k_hi"] - df.loc[a, "ndcg@k"] for a in arms]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(arms, means, color=[_ARM_COLORS[a] for a in arms],
           yerr=[lower, upper], capsize=5, error_kw={"ecolor": "#333", "lw": 1.2})
    for i, m in enumerate(means):
        ax.text(i, m + upper[i] + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"nDCG@{df['k'].iloc[0]}")
    ax.set_title("Retrieval quality by arm (95% bootstrap interval)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "04_arm_comparison.png")
    plt.close(fig)


def quality_vs_latency_figure(summary: list[dict]) -> None:
    """The headline: what each arm buys in nDCG and what it costs in time.

    Latency is on a log axis because the arms span orders of magnitude — on a
    linear axis the three cheap arms collapse into one another and the chart
    says only "the reranker is slow", which is not the finding.
    """
    plt = _lazy_plt()
    df = pd.DataFrame(summary)
    df = df[df["slice"] == "all"].set_index("arm")
    arms = _ordered_arms(set(df.index))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for a in arms:
        ax.scatter(df.loc[a, "latency_ms_median"], df.loc[a, "ndcg@k"],
                   s=170, color=_ARM_COLORS[a], zorder=3, label=a,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(
            f"  {a}\n  {df.loc[a, 'ndcg@k']:.3f} @ {df.loc[a, 'latency_ms_median']:.0f} ms",
            (df.loc[a, "latency_ms_median"], df.loc[a, "ndcg@k"]),
            fontsize=8.5, va="center", color="#333",
        )
    ax.set_xscale("log")
    ax.set_xlabel("median latency per query (ms, log scale)")
    ax.set_ylabel(f"nDCG@{df['k'].iloc[0]}")
    ax.set_title("Quality per millisecond: what each retrieval arm costs")
    ax.margins(x=0.28, y=0.18)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "05_quality_vs_latency.png")
    plt.close(fig)


# Query styles, ordered from most natural-language to least. This ordering is the
# x-axis of the robustness figure, so it is defined once here.
_STYLE_ORDER = ["abstract", "title", "keyword"]
_STYLE_LABELS = {
    "abstract": "abstract sentence\n(fluent prose)",
    "title": "title\n(short phrase)",
    "keyword": "keyword bag\n(no grammar)",
}


def query_style_figure(summary: list[dict]) -> None:
    """Each arm's nDCG across the three query styles.

    The three styles run from fluent prose to a shuffled bag of five words. An
    arm that only works on well-formed sentences will slope downwards; BM25,
    which never reads word order at all, should stay flat. The *slope* is the
    measurement — it says how much a retriever depends on the user phrasing their
    query nicely, which is not something a single averaged score can show.
    """
    plt = _lazy_plt()
    df = pd.DataFrame(summary)
    styles = [s for s in _STYLE_ORDER if s in set(df["slice"])]
    if len(styles) < 2:
        return  # nothing to contrast — e.g. a curated-only summary

    pivot = df[df["slice"].isin(styles)].pivot(index="arm", columns="slice",
                                               values="ndcg@k")
    arms = _ordered_arms(set(pivot.index))
    pivot = pivot.loc[arms, styles]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = range(len(styles))
    for arm in arms:
        ax.plot(x, pivot.loc[arm], marker="o", markersize=8, lw=2.2,
                color=_ARM_COLORS[arm], label=arm, zorder=3)
        drop = pivot.loc[arm, styles[0]] - pivot.loc[arm, styles[-1]]
        ax.annotate(f" {arm}  ({drop:+.2f})", (len(styles) - 1, pivot.loc[arm, styles[-1]]),
                    fontsize=8.5, va="center", color=_ARM_COLORS[arm])

    ax.set_xticks(list(x))
    ax.set_xticklabels([_STYLE_LABELS.get(s, s) for s in styles], fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("nDCG@10")
    ax.set_title("Robustness to query style\n"
                 "(labels show each arm's drop from prose to keywords)", fontsize=11)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.margins(x=0.30)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "06_query_style_robustness.png")
    plt.close(fig)


def drift_figure(drift_record: dict) -> None:
    """Index coverage of indexed vs newer papers, against the day-zero floor.

    The figure has to carry the correction, not just the number. An earlier
    version drew the raw drop against a flat 15% line and read as a verdict on
    the index; the honest version draws the *floor* — what this probe reads on an
    index built an hour ago — because the distance from the floor is the only
    part of the drop that means anything. See `src/drift.py` and
    `outputs/results/index_2026-07-27/` for the reading that exposed it.
    """
    plt = _lazy_plt()

    baseline = drift_record["baseline_top1_mean"]
    probe = drift_record["probe_top1_mean"]
    floor = drift_record.get("floor_relative_drop", 0.0)
    # Where the probe bar would sit if the index were brand new: the floor is a
    # relative drop, so it maps onto this run's own baseline.
    floor_value = baseline * (1.0 - floor)
    trigger_value = baseline * (1.0 - (floor + drift_record["trigger_excess"]))

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars = ax.bar(
        ["papers in the index", "newer, unseen papers"],
        [baseline, probe],
        color=["#4c78a8", "#e0a458"], width=0.55,
    )
    for bar, v in zip(bars, [baseline, probe]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=10)

    ax.axhline(floor_value, ls="-", lw=1.3, color="#2f6f5e")
    ax.text(1.42, floor_value, f" day-zero floor {floor_value:.3f}\n"
                               f" ({floor:.1%} drop on a fresh index)",
            va="bottom", fontsize=8.5, color="#2f6f5e")

    ax.axhline(trigger_value, ls="--", lw=1.3, color="#a6462f")
    ax.text(1.42, trigger_value, f" re-index below {trigger_value:.3f}\n"
                                 f" (floor + {drift_record['trigger_excess']:.0%})",
            va="top", fontsize=8.5, color="#a6462f")

    verdict = ("TRIGGERED — re-index" if drift_record["reindex_now"]
               else "at the floor — nothing to re-index for")
    ax.set_ylabel("mean top-1 cosine similarity")
    ax.set_ylim(0, max(baseline, probe) * 1.42)
    ax.set_title(f"{drift_record['relative_drop']:.1%} drop, "
                 f"{drift_record['excess_over_floor']:+.1%} vs the floor — {verdict}",
                 fontsize=11)
    ax.margins(x=0.30)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "07_drift.png")
    plt.close(fig)


def corpus_figure(papers: list) -> None:
    """What is actually in the corpus, by arXiv primary category.

    Written for the deck's data slide, and generated from `papers.jsonl` like
    everything else — the deck's rule is that no figure is drawn by hand, and for
    a while this one was: the PNG existed in `outputs/figures/deck/` with no code
    behind it. A figure nobody can regenerate is a number nobody can check.
    """
    plt = _lazy_plt()
    from collections import Counter

    # arXiv lists every category a paper is cross-listed under; the first is the
    # primary one, and counting all of them would make the bars sum to more than
    # the corpus and mean something other than "what this corpus is".
    counts = Counter(p.categories[0] for p in papers if p.categories)
    top = counts.most_common(10)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labels = [c for c, _ in top][::-1]
    values = [n for _, n in top][::-1]
    ax.barh(labels, values, color="#4c78a8")
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, str(v), va="center", fontsize=8.5)

    ax.set_xlabel("papers")
    ax.set_title(f"{len(papers):,} papers by primary arXiv category", fontsize=11)
    ax.set_xlim(0, max(values) * 1.12)
    fig.tight_layout()
    (config.FIGURES_DIR / "deck").mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "deck" / "deck_corpus_categories.png")
    plt.close(fig)


def significance_figure(records: list[dict]) -> None:
    """Paired nDCG differences between arms, with intervals, ordered by size.

    A forest plot rather than a bar chart, because the quantity being shown is a
    *difference with an interval* and the only question a reader has is whether
    each interval clears zero. The zero line is therefore the loudest thing on
    the chart, and the pairs Holm rejects are drawn hollow — significance is a
    property of the comparison, not a footnote under it.
    """
    plt = _lazy_plt()
    if not records:
        return

    records = sorted(records, key=lambda r: r["mean_diff"])
    labels = [f"{r['arm_a']} − {r['arm_b']}" for r in records]
    means = [r["mean_diff"] for r in records]
    lower = [r["mean_diff"] - r["mean_diff_lo"] for r in records]
    upper = [r["mean_diff_hi"] - r["mean_diff"] for r in records]
    y = list(range(len(records)))

    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(records) + 2.1))
    for i, rec in enumerate(records):
        significant = rec["significant"]
        ax.errorbar(
            means[i], i, xerr=[[lower[i]], [upper[i]]],
            fmt="o", markersize=7, capsize=4, lw=1.4,
            color="#2f6f4f" if significant else "#9aa5b1",
            markerfacecolor="#2f6f4f" if significant else "white",
        )
        ax.text(rec["mean_diff_hi"] + 0.012, i,
                f"p={rec['p_value_holm']:.3f}" if significant
                else f"p={rec['p_value_holm']:.2f} (n.s.)",
                va="center", fontsize=8,
                color="#2f6f4f" if significant else "#7a8590")

    ax.axvline(0, color="#a6462f", lw=1.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("difference in nDCG@10 (paired, per question)")
    ax.set_title("Which gaps survive a paired test?\n"
                 "sign-flip randomization, Holm-corrected over all pairs",
                 fontsize=11)
    ax.margins(x=0.22)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "08_arm_significance.png")
    plt.close(fig)


def k_sweep_figure(rows: list[dict]) -> None:
    """nDCG@k and recall@k against the cutoff, per arm.

    Two panels because the two curves answer different questions and share no
    useful axis: nDCG@k asks "how good is the ordering down to here" and can fall
    as k grows, while recall@k only ever climbs and shows where an arm has found
    everything it is going to find. The served cutoff (TOP_K) and the reported
    one (EVAL_K) are marked, since the point of the sweep is to check them.
    """
    plt = _lazy_plt()
    df = pd.DataFrame(rows)
    arms = _ordered_arms(set(df["arm"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    for metric, ax, title in (
        ("ndcg@k", axes[0], "Ordering quality: nDCG@k"),
        ("recall@k", axes[1], "Coverage: recall@k"),
    ):
        for arm in arms:
            sub = df[df.arm == arm].sort_values("k")
            ax.plot(sub["k"], sub[metric], marker="o", markersize=4.5,
                    color=_ARM_COLORS[arm], label=arm, lw=1.6)
            ax.fill_between(sub["k"], sub[f"{metric}_lo"], sub[f"{metric}_hi"],
                            color=_ARM_COLORS[arm], alpha=0.12, lw=0)
        ax.axvline(config.TOP_K, ls=":", lw=1.2, color="#7a8590")
        ax.axvline(config.EVAL_K, ls="--", lw=1.2, color="#7a8590")
        ax.set_xscale("log")
        ax.set_xticks(sorted(df["k"].unique()))
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("k (log scale)")
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.6)

    axes[0].set_ylabel("score")
    axes[0].annotate(f"served k={config.TOP_K}", xy=(config.TOP_K, 0.06),
                     fontsize=8, color="#7a8590", rotation=90, ha="right")
    axes[0].annotate(f"reported k={config.EVAL_K}", xy=(config.EVAL_K, 0.06),
                     fontsize=8, color="#7a8590", rotation=90, ha="right")
    axes[1].legend(fontsize=9, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "09_k_sweep.png")
    plt.close(fig)


def chunk_ablation_figure(rows: list[dict]) -> None:
    """nDCG@10 against chunk size, per arm — does the constant matter, and to whom?

    The interesting shape is divergence: if the two arms peak at different chunk
    sizes then `CHUNK_SIZE_CHARS` is a retriever-specific decision the project
    made once for both, which is a different (and more useful) finding than "1000
    was fine".
    """
    plt = _lazy_plt()
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for arm in _ordered_arms(set(df["arm"])):
        sub = df[df.arm == arm].sort_values("chunk_size")
        ax.errorbar(
            sub["chunk_size"], sub["ndcg@k"],
            yerr=[sub["ndcg@k"] - sub["ndcg@k_lo"], sub["ndcg@k_hi"] - sub["ndcg@k"]],
            marker="o", markersize=5, capsize=4, lw=1.6,
            color=_ARM_COLORS[arm], label=arm,
        )

    ax.axvline(config.CHUNK_SIZE_CHARS, ls="--", lw=1.2, color="#a6462f")
    ax.text(config.CHUNK_SIZE_CHARS, ax.get_ylim()[0],
            f" shipped: {config.CHUNK_SIZE_CHARS} chars", fontsize=8.5,
            color="#a6462f", va="bottom")
    ax.set_xlabel("chunk size (characters)")
    ax.set_ylabel(f"nDCG@{config.EVAL_K}")
    ax.set_title("Does the chunk size the project argued for actually matter?",
                 fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "10_chunk_ablation.png")
    plt.close(fig)


def benchmark_figures(summary: list[dict], rows: list[dict],
                      drift_record: dict | None,
                      significance_records: list[dict] | None = None,
                      k_sweep_rows: list[dict] | None = None) -> None:
    """Regenerate every retrieval-benchmark figure. Called by run_retrieval_bench.py.

    Each figure is written from the saved results and nowhere else, so a figure
    can never disagree with the JSON a reader checks it against. Figures whose
    inputs are missing are skipped rather than faked — the drift probe needs the
    network, and an older saved run may predate the significance and sweep
    results entirely.
    """
    config.ensure_dirs()
    arm_comparison_figure(summary)
    quality_vs_latency_figure(summary)
    query_style_figure(summary)
    if drift_record:
        drift_figure(drift_record)
    if significance_records:
        significance_figure(significance_records)
    if k_sweep_rows:
        k_sweep_figure(k_sweep_rows)
