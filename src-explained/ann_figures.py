from render_explained import build

build(
    "../src/figures.py",
    "figures.py_explained.pdf",
    "Every README figure, regenerated from the saved results",
    "No figure in this project is drawn by hand or lives only in a notebook cell. This module "
    "rebuilds each one from the evaluation results, so a figure can never drift from the numbers it "
    "claims to show. run_eval.py calls these after computing results.",
    [
        (7, "Lines 1 to 7 are the module docstring, stating the principle: every figure is produced "
            "here from saved results, so it is always reproducible and never drifts, and run_eval.py "
            "is the caller."),
        (6, "Lines 8 to 13 are imports: postponed annotations (line 9), pandas for shaping the "
            "results into frames (line 11), and the project config for output paths (line 13). "
            "Matplotlib is deliberately not imported here — see the lazy loader below."),
        (4, "Lines 14 to 17 define the palette: _NAIVE_COLOR (a muted grey) and _AGENT_COLOR (the "
            "project green), so naive and agent bars are colored consistently across every figure "
            "and the set reads as one visual family."),
        (12, "Lines 18 to 29 define _lazy_plt, which imports matplotlib only when a figure is "
             "actually drawn. Line 24 sets the Agg backend (headless: write files, never open a "
             "window), important for running under a script or notebook execution. Lines 27 to 28 "
             "apply a clean default style (DPI, font size, and hiding the top and right spines). "
             "Line 29 returns the pyplot module for the caller to use."),
        (24, "Lines 30 to 53 define retrieval_metrics_figure — a grouped bar chart of hit@k, "
             "recall@k, and MRR. It loads pyplot lazily (line 34), frames the rows (line 35), and "
             "names the three metrics (line 36). Line 38 splits the data into all, single-hop, and "
             "multi-hop groups. Lines 39 to 44 draw one bar cluster per group, guarding empty "
             "groups with a 0.0 fallback (line 43). Lines 45 to 50 set the tick labels, the 0-to-1 "
             "y-limit, axis label, title, and legend. Lines 51 to 53 tighten the layout, save the "
             "PNG to the figures directory, and close the figure to free memory."),
        (21, "Lines 54 to 74 define answer_quality_figure — naive versus agent, for groundedness and "
             "correctness, split by hop type. It frames the answer rows (line 59) and makes two side "
             "by side subplots sharing a y-axis (line 61). For each metric (line 62) it pivots the "
             "mean judge score by hop and system (line 63), orders the columns naive then agent "
             "(line 64), and draws a grouped bar with the palette, showing the legend only once "
             "(lines 65 to 66). Lines 67 to 70 set titles, x-labels, the 0-to-5 range, and the "
             "shared y-label. Lines 71 to 74 add the overall title, tighten, save, and close."),
        (17, "Lines 75 to 91 define cost_tradeoff_figure — the other side of the ledger: how many "
             "retrievals each system spends per question. It frames the rows (line 80) and computes "
             "the mean n_retrievals per system, ordered naive then agent (line 81). Lines 83 to 84 "
             "draw the two bars in the palette. Lines 85 to 86 set the y-label and title. Lines 87 "
             "to 88 annotate each bar with its value to one decimal. Lines 89 to 91 tighten, save "
             "the PNG, and close."),
        (None, "Lines 92 to 100 define all_figures, the single entry point run_eval.py calls. It "
                "ensures the output directories exist (line 96), always draws the retrieval-metrics "
                "figure (line 97, which needs no API key), and — only if answer rows are present "
                "(line 98) — draws the answer-quality and cost-tradeoff figures (lines 99 to 100). "
                "So the local retrieval figure always regenerates, while the key-gated answer "
                "figures render only once the answer evaluation has produced data."),
    ],
)
