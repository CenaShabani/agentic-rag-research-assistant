# Adjudication rule — how a document is judged relevant

Every retrieval number this project publishes is a comparison against a list of
"correct" documents. That list is the most attackable part of any retrieval
evaluation, so this file states exactly how it was produced, what it is good for,
and where it lies to you.

There are **two evaluation sets**, built by different methods, reported
separately, and never averaged together.

---

## Set 1 — `qa_set.json`: 11 hand-written questions

**How the labels were made.** By hand. Each question was written first, then the
arXiv ids that genuinely answer it were chosen deliberately and pinned in
`config.SEED_PAPER_IDS` so they are guaranteed present in the corpus.

**What counts as relevant.** A paper is relevant if a competent reader, holding
only that paper, could answer the question. Not "the paper is on the same topic"
— topical adjacency is not relevance. For the multi-hop questions, *all* listed
papers are required: the question asks something that no single paper answers,
which is the entire point of the slice.

**What it is good for.** It is the only set here containing genuine multi-hop
questions, and multi-hop is where the project's hypothesis lives. It is the
harder, more realistic set.

**What is wrong with it.** Eleven questions. That is far too few to support a
confidence interval — one question changing its mind moves the mean by nine
percentage points. Treat any difference between arms on this set as an anecdote,
and read the bootstrap intervals as the wide, honest things they are. It was also
written by the same person who built the retriever, which is a conflict of
interest no amount of care fully removes.

---

## Set 2 — `known_item_set.json`: 180 generated known-item queries

**How the labels were made.** Programmatically, by
[`src/evalset.py`](../src/evalset.py), seeded with `config.SEED`. 60 papers are
sampled from the corpus, and each yields three queries — one per **query style**:

| Style | Query text | Relevant document |
|---|---|---|
| `abstract` | the longest non-boilerplate sentence in its abstract | that paper |
| `title` | the paper's title | that paper |
| `keyword` | 5 content words sampled from across the abstract, stop-words removed, order shuffled | that paper |

Every eligible paper contributes all three, so the slices are perfectly paired:
same papers, same targets, only the phrasing differs. That pairing is what makes
comparing one arm across styles a controlled measurement rather than three
unrelated samples.

**What counts as relevant.** Exactly one paper per query: the paper the query was
extracted from. This is a *known-item* task — the query was manufactured from the
document, so the judgement is a matter of record, not of opinion. There is no
adjudicator to disagree with, which is the design's whole advantage.

**Binary, not graded.** nDCG normally consumes graded relevance (0–3). Here every
gain is 0 or 1, because a known-item query has one right answer and no partial
credit to distribute. nDCG@10 over binary gains is still meaningful — it rewards
putting the one correct document high rather than merely inside the top 10 — but
it is not directly comparable to a graded-relevance nDCG from BEIR or TREC. Do not
put this project's nDCG next to a leaderboard's and draw a conclusion.

**Keyword selection.** Stop-words are removed with a small hand-written list that
includes academic filler ("propose", "novel", "results") appearing in nearly every
abstract in the corpus. That list is used *only* to build queries — never inside a
retriever, where BM25's IDF already discounts ubiquitous terms and a curated stop
list would be an untunable thumb on the scale.

**Sentence selection.** The longest sentence is chosen because length correlates
with specificity, and a query has to be specific to identify one paper out of
2,012. Sentences under 40 characters (`config.MIN_QUERY_CHARS`) and boilerplate
openers ("We release our code…", "In this paper…") are dropped: every paper
contains those, so they identify nothing and would make the task impossible for
every arm equally — noise, not difficulty.

---

## The bias this set has, stated plainly

**Every query here is built from its target document's own words.** BM25 scores
documents by term overlap, so on this set BM25 is being handed something close to
the answer key. Any absolute claim of the form "BM25 achieves 0.99 nDCG" is a
statement about this construction as much as about BM25.

This is not a flaw discovered late and reluctantly disclosed — it is why the set
has three styles instead of one. Two things make it survivable:

**1. The three styles are a gradient, and the slope is the measurement.** The
first version of this set had only the `abstract` style, and it was useless:
every arm scored ~1.00, because a sentence lifted verbatim from an abstract is a
literal substring of the very passage it targets. That is a degenerate retrieval
task. The `keyword` style breaks it — five shuffled content words drawn from
across the whole abstract are not a substring of any single passage, and have no
grammar for a sentence-embedding model to lean on. Comparing one arm from
`abstract` to `keyword` measures **how much that arm depends on the user phrasing
their query as well-formed prose**, which is a real property of a real system and
not an artifact of how the labels were made.

**2. The hand-written set is the control.** Set 1's questions were written by a
person in their own words, so they do *not* inherit this construction bias. If
the two sets disagreed, the generated set would be the suspect. They agree — BM25
leads on both — which is the reason the headline finding is stated at all.

What this set still cannot do is prove BM25 would win on queries phrased in
vocabulary genuinely disjoint from the corpus. Nothing here samples that case,
and no offline construction can manufacture it without a human or a language
model writing the questions.

## What both sets still cannot tell you

- **One corpus, one domain.** 2,012 cs.CL / cs.LG arXiv papers. Nothing here
  generalises to legal, medical, or conversational retrieval.
- **Titles and abstracts only.** `provenance.json` records
  `redistributable_count: 0` — no paper in the corpus carried a licence permitting
  full-text storage, so every "document" is a title plus an abstract. Real
  retrieval over full papers is a harder and partly different problem.
- **Known-item is the easy end of retrieval.** One right answer, extracted from
  the target, no ambiguity, no competing near-duplicates that are *almost* right.
  Real queries are underspecified and have many partially-relevant answers.
  Absolute scores here will flatter every arm; the *ranking between arms* is what
  this set is for.
- **No human ever re-checked the generated labels.** The construction makes them
  correct by definition, but "correct by definition" and "a query a human would
  ever type" are different claims, and only the first one is being made.
