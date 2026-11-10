# 09 · Evaluating RAG

A RAG system that you can't measure is a demo, not an engineering artifact. This
lesson is how to tell whether retrieval is finding the right things and whether the
answers are any good — and how to compare naive vs agentic RAG *fairly*. This is
the methodology that carries the project.

## Split the evaluation in two

The single most important idea: **evaluate retrieval and generation separately.**
They fail for different reasons and you fix them in different places.

- If **retrieval** is bad, the model never had a chance — it was handed the wrong
  passages. No prompt fixes that.
- If **retrieval** is good but the **answer** is bad, the problem is in generation
  or grounding — the evidence was there and the model fumbled it.

Lump them into one score and you can't tell which half to fix. So we measure each.

## Evaluating retrieval — with ground truth

For each evaluation question we know, in advance, which paper(s) genuinely answer
it (the `relevant_ids` in [`eval/qa_set.json`](../eval/qa_set.json)). Given that
ground truth, three standard metrics (all in
[`src/evaluation.py`](../src/evaluation.py)):

- **hit@k** — did *any* relevant paper appear in the top-k results? A blunt
  yes/no: did retrieval succeed at all?
- **recall@k** — what *fraction* of the relevant papers were retrieved? For a
  multi-hop question needing two papers, retrieving one is recall 0.5. This is the
  metric that exposes multi-hop weakness.
- **MRR** (Mean Reciprocal Rank) — 1 / (rank of the first relevant paper). It
  rewards putting the right paper *high*: rank 1 scores 1.0, rank 3 scores 0.33.
  Ranking matters because the LLM pays most attention to the top passages.

These use only the vector store, so they run offline with no API key and reproduce
exactly. Retrieval quality is a solved, quantitative thing.

## Evaluating answers — LLM-as-judge

Answer quality is fuzzier: there's no single correct string. The standard modern
approach is **LLM-as-judge** — use a separate LLM call to grade the answer against
a rubric. Here the judge scores two axes (1–5), in [`src/evaluation.py`](../src/evaluation.py):

- **Groundedness** — is every claim supported by the retrieved passages? (This is
  the anti-hallucination metric: a fluent answer that invents facts scores low.)
- **Correctness** — does it actually answer the question, accurately?

The judge is given the question, the sources, and the answer, and returns JSON.
It's not perfect — an LLM judge has its own biases — but it's consistent, cheap,
and it correlates well with human judgement, which makes it the practical tool for
comparing two systems at scale.

> **Why groundedness and correctness are separate.** An answer can be grounded but
> wrong (faithfully summarizes a passage that doesn't answer the question) or
> correct but ungrounded (right answer, pulled from the model's memory, not the
> sources). You want both, so you measure both.

## The fair comparison — controlling variables

The project's finding is a head-to-head: **naive RAG vs agentic RAG.** For that
comparison to mean anything, everything *except the strategy* must be held
constant — same corpus, same embedding model, same retriever, same judge, same
questions. The **only** difference is single-shot vs iterative retrieval. That's
why the repo has one `VectorStore` both systems share: it removes "maybe they just
searched differently" as an explanation. Any gap is attributable to the loop.

And we slice by **hop type**. The hypothesis (lesson 07) is specific: the agent
should win on **multi-hop** questions and roughly tie on **single-hop** ones. A
single overall average would blur that — the interesting result is *where* the
agent helps, not just whether it helps on average.

## Counting the cost, not just the quality

A quality win that costs 3× the tokens is a real trade, not a free lunch. So the
evaluation also records `n_retrievals` per question — the agent's extra searches
made visible. The `03_cost_tradeoff.png` figure puts quality and cost side by side,
so the conclusion is honest: *the agent answers multi-hop questions better, at N×
the retrieval cost.* That framing — benefit **and** price — is the defensible one.

## "Arm A scored higher" is not a finding yet

Suppose you measure two retrieval strategies and one scores 0.87 while the other
scores 0.75. Is the first one better?

You cannot tell yet. Both numbers are averages over a particular set of questions,
and a different set would have given different averages. The honest question is:
**if I had asked different questions, would the ordering have survived?**

The usual first answer is a confidence interval per arm — and this project reports
them (lesson 11). But there is a trap here that catches almost everyone, so it is
worth walking into deliberately.

### The overlap trap

Two arms, each with a 95% interval:

```
arm A   0.87   [0.82, 0.92]
arm B   0.75   [0.69, 0.81]      <- intervals do not overlap
arm C   0.83   [0.78, 0.88]      <- overlaps A heavily
```

It is tempting to read this as "A beats B, and A vs C is a coin flip". The first
half is fine. **The second half does not follow.** Overlapping intervals do not
mean there is no difference — the implication only runs one way.

Here is why, with two questions:

| | arm A | arm C | A − C |
|---|--:|--:|--:|
| q17 | 0.63 | 0.50 | +0.13 |
| q18 | 0.99 | 0.86 | +0.13 |

Each arm's own scores swing from 0.63 to 0.99, because **some questions are just
harder than others**. That swing is what makes each interval wide. But it is the
*same* swing for both arms — they answered the same questions — so it cancels
completely in the last column. A beat C by the same amount both times. There is
nothing ambiguous about it, and the marginal intervals could not see that.

### The fix: compare the differences, not the averages

When every arm answers the same question set, you have paired data, and the right
unit of analysis is the **per-question difference**. Two things fall out of it:

- **An effect size in the metric's own units** — "A beats C by 0.09 nDCG
  [0.05, 0.13]" — which is the number worth quoting, because it says *how much*
  and not just *whether*.
- **A significance test that assumes almost nothing.** Under the null hypothesis
  "these two arms are interchangeable", the sign of each question's difference is
  arbitrary. So flip the signs at random a few thousand times and see how often
  you get a mean as extreme as the real one. That is the **randomization test**,
  and it is the standard one in information retrieval. No bell curve, no
  assumption about the shape of the data — just the data, rearranged.

### And then pay for asking many questions

Four arms make six pairwise comparisons; five make ten. If each test has a 5%
false-positive rate, ten of them give you roughly a **40% chance** that at least
one pair looks different purely by luck. So the p-values get a **multiple-comparison
correction** (Holm, in this project) before anyone is allowed to say "significant".

This matters more than it sounds. A benchmark table with ten uncorrected
comparisons is a table where you should expect one wrong row — and it will be
whichever row is most surprising, because that is the one you will want to write
about.

> **The lesson in one line:** an average is a summary, an interval is an
> uncertainty, and only a paired test is a comparison. In this project the
> distinction changed a conclusion — see lesson 11.

The code is [`src/significance.py`](../src/significance.py), and it is deliberately
separate from `evaluation.py`: describing one arm and comparing two are different
jobs, and merging them is how the overlap trap gets built into a codebase.

## Reproducibility: cached, committed results

Every LLM call (answers and judgements) is cached to disk and committed. So the
retrieval metrics reproduce offline always, and the answer metrics reproduce from
the committed cache without a key — you only need a key to *regenerate* them from
scratch. The README says plainly which numbers came from live calls. Same
"reference-only, load the saved result" honesty the portfolio applies to its
GPU-trained notebooks.

## Where this shows up in the repo

- [`src/evaluation.py`](../src/evaluation.py) — metrics, the judge, and the
  naive-vs-agent harness.
- [`eval/qa_set.json`](../eval/qa_set.json) — the curated questions with ground
  truth and hop labels.
- [`run_eval.py`](../run_eval.py) — runs it all and writes the results + figures.

Next: **[10 · Failure modes and limits](10_failure_modes_and_limits.md)** — the
honest catalogue of where RAG breaks.
