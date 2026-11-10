# 11 — Benchmarking retrieval: nDCG, MAP, and the interval that stops you fooling yourself

Lesson 09 said measure retrieval and answers separately. This lesson is the part
that turns "we measured retrieval" into "we can defend the number" — because the
metrics from lesson 09 (hit@k, recall@k, MRR) have a blind spot, and a single
number with no interval around it is not evidence.

This is also the lesson where the project's own conclusion got overturned. Worth
reading for that alone.

---

## 1. Why hit@k, recall@k and MRR are not enough

The three metrics we started with each answer a narrow question:

| Metric | Answers | Blind to |
|---|---|---|
| hit@k | did *anything* relevant show up? | how many, and where |
| recall@k | *what fraction* showed up? | order entirely |
| MRR | how early was the **first** hit? | everything after the first hit |

Consider two retrievers on a question with two relevant papers, `k = 10`:

```
A:  [relevant, relevant, junk, junk, junk, ...]
B:  [relevant, junk, junk, junk, junk, junk, junk, junk, relevant, junk]
```

**hit@10, recall@10 and MRR are identical for both** — 1.0, 1.0, 1.0. Yet A is
obviously the better retriever, and if you only feed the top 5 to your LLM, B
loses half its evidence. The metrics cannot see the difference.

## 2. nDCG — grading the whole ordering

**DCG** (discounted cumulative gain) sums the relevance of each result, divided
by a logarithm of its position:

```
DCG = Σ  gain(i) / log2(i + 1)
```

Position 1 is worth 1.00, position 2 about 0.63, position 10 about 0.29. The
logarithm encodes an assumption worth saying out loud: **attention falls off
sharply with rank, but never quite to zero.** Rank 10 is much worse than rank 1,
not worthless.

Raw DCG is not comparable across questions — a question with three relevant
papers can score higher than one with a single relevant paper purely because
there is more to find. So we normalise by the **ideal** DCG, the score the perfect
ranking would have achieved *for that question*:

```
nDCG = DCG / IDCG        → 0 to 1, where 1.0 means "nothing could be ordered better"
```

On the example above: A scores 1.000, B scores 0.755. The metric now sees it.

One implementation detail that matters: **the ideal is capped at k.** With five
relevant papers and `k = 2`, retrieving two of them must score 1.000 — otherwise
an arm could never score 1.0 on questions with more relevant papers than the
cutoff allows, and scores would stop being comparable across questions.

> In code: `evaluation.dcg` and `evaluation.ndcg_at_k`.

## 3. MAP — the same job, a different opinion

Average precision measures precision **at every hit**, then averages:

```
AP = (1/R) · Σ over hits of  (hits so far / rank of this hit)
```

For A: `(1/1 + 2/2)/2 = 1.000`. For B: `(1/1 + 2/9)/2 = 0.611`.

AP punishes a late second hit **harder** than nDCG's smooth logarithm does. Both
are reported, and the point is that **they disagree informatively**: a strong nDCG
with a weak MAP means the arm reliably finds *one* good document and then loses
interest. That is a diagnosis, not just a score.

The divisor is the number of relevant documents, not the number found — so a miss
costs, rather than being quietly excluded from the average.

> In code: `evaluation.average_precision`.

## 4. The bootstrap — the part most people skip

Suppose arm A scores 0.71 and arm B scores 0.68 over your questions. Is A better?

**You cannot tell from those numbers.** The honest question is: if I had drawn a
different sample of questions, how much would the mean have moved?

The bootstrap answers it without any distributional assumption:

1. Resample your questions **with replacement**, same sample size.
2. Compute the mean on that resample.
3. Do it 2,000 times.
4. Take the 2.5th and 97.5th percentiles of those means.

That is your 95% confidence interval.

**What you resample is a real decision.** We resample *questions*, because the
question set is the sample we are generalising over. The claim being defended is
"on queries like these", not "on this exact list of 180" — and resampling
questions is what makes the interval mean the former.

Why this project grew its eval set from 11 questions to 180:

| Set | BM25 nDCG@10 | 95% interval | Can it decide anything? |
|---|--:|:--:|---|
| 11 hand-written | 0.812 | [0.611, 0.962] | No — overlaps every other arm |
| 180 generated | 0.978 | [0.962, 0.992] | Yes |

Eleven questions cannot support a claim. The interval is what makes that visible
instead of arguable.

> In code: `evaluation.bootstrap_ci`, seeded so a published interval reproduces
> to the digit.

## 4b. One interval per arm is still not a comparison

Here is the trap lesson 09 set up, sprung on this project's real numbers.

The four intervals above describe **each arm on its own**. Reading two of them
side by side and checking whether they overlap is the natural next move, and it is
wrong — the implication only runs one way. Non-overlapping intervals do imply a
difference; overlapping ones imply nothing.

Every arm answered the **same 180 questions**, so the per-question difference
exists and is a much sharper instrument. Comparing arms by that difference, with a
sign-flip randomization test and a Holm correction across all six pairs:

| Comparison | Δ nDCG@10 | 95% CI | won / lost / tied | p (Holm) |
|---|--:|:--:|:--:|--:|
| bm25 vs dense | +0.227 | [0.171, 0.286] | 52 / 1 / 127 | 0.0006 |
| rerank vs dense | +0.213 | [0.160, 0.272] | 51 / 1 / 128 | 0.0006 |
| hybrid vs dense | +0.122 | [0.090, 0.156] | 46 / 0 / 134 | 0.0006 |
| bm25 vs hybrid | +0.105 | [0.071, 0.141] | 37 / 1 / 142 | 0.0006 |
| rerank vs hybrid | +0.090 | [0.057, 0.127] | 34 / 5 / 141 | 0.0006 |
| **bm25 vs rerank** | **+0.014** | **[−0.001, 0.033]** | **8 / 6 / 166** | **0.101** |

**The last row changed a conclusion.** On the marginal numbers BM25 (0.978) sits
above the cross-encoder (0.964), and it is tempting — the earlier version of this
project did exactly this — to say "the reranker scored *lower*". Paired, that gap
is 8 wins against 6 losses over 166 ties, and it does not survive the test. The
defensible claim is not "BM25 beats the reranker". It is:

> **The cross-encoder buys no measurable quality over BM25, and charges about
> 1,200× the latency for it.**

That is a *stronger* result, not a weaker one. "Slightly better" invites an
argument about whether the difference is worth paying for; "no measurable
difference at a thousand times the price" ends it.

Note also the wins/losses/ties columns. `hybrid vs dense` is 46–0: hybrid never
once lost. A mean difference alone cannot tell you whether an arm wins
consistently or wins twice enormously, and those are very different systems.

> In code: `significance.py`, figure `08_arm_significance.png`.

## 4c. Sweep the cutoffs you already committed to

Every number so far is "@10". Why 10? Because BEIR and TREC report at 10 — a fine
reason to *report* there, and no reason at all to believe it is where the
interesting behaviour is. Meanwhile the system serves the top **5** chunks to the
LLM, and the reranker rescores a shortlist of **50**. Three numbers, all argued in
a config file, none measured.

Sweeping k for every arm (one search at k = 50, then truncated — so the *cutoff* is
the only thing varying):

| k | bm25 | dense | hybrid | rerank |
|--:|--:|--:|--:|--:|
| 1 | 0.956 | 0.711 | 0.794 | 0.928 |
| 5 | **1.000** | 0.778 | 0.917 | 0.994 |
| 10 | 1.000 | 0.800 | 0.967 | 0.994 |
| 50 | 1.000 | **0.856** | 1.000 | 1.000 |

*(recall@k — what fraction of the target papers were found by depth k)*

Three things fall out that no fixed-k table could show:

1. **Serving 5 costs nothing under BM25.** Its recall is already 1.000 at k = 5,
   so the gap between the served cutoff and the reported one is not hiding losses.
   Under the *dense* arm it would: recall at 5 is 0.778 against 0.800 at 10.
2. **The dense arm never saturates.** Even at k = 50 it has found only 85.6% of
   the targets. That is not a ranking problem, it is a **recall ceiling** — and it
   is the mechanical reason the reranker cannot rescue it. A cross-encoder
   reorders what it is handed; it cannot retrieve what the first stage never
   returned.
3. **The reranker's shortlist is over-provisioned.** Hybrid recall hits 1.000 by
   k = 20, so a 50-candidate shortlist contains nothing after rank 20 that the
   reranker needs. `RERANK_CANDIDATES = 50` could be 20 for ~2.5× less
   cross-encoder work at no cost in recall — a real saving, invisible until swept.

> In code: `evaluation.sweep_k`, figure `09_k_sweep.png`.

## 4d. Ablate the constant you argued hardest for

Every project has a number that was chosen with a paragraph of reasoning and never
tested. Here it is `CHUNK_SIZE_CHARS = 1000`, justified entirely on **embedding
dilution** — too large and one vector must average several ideas.

Rebuilding the corpus at five sizes and re-scoring gives point estimates that look
like a clean story: dense peaks at 400, BM25 climbs all the way to 2000, so the two
arms want opposite things. Then apply §4b's discipline — paired against the shipped
setting, Holm-corrected — and half the story evaporates:

- **Dense: not one setting differs significantly from 1000.** The apparent
  preference for 400 chars is 12 questions better, 12 worse. Noise.
- **BM25: three of four do.** 400 is worse (−0.037, p = 0.0004), 1400 and 2000 are
  better (+0.018 and +0.020).

So the constant matters to the arm the argument was *not* about, and does not
measurably matter to the arm it *was* about. Two things to take from that:

**The mechanism is different for a lexical retriever.** BM25 scores *documents*,
and chunking splits a paper's term mass across several of them — each chunk then
matches fewer of the query's terms, while the ground truth stays per-paper.
Fragmentation is the cost, and it has nothing to do with vectors averaging ideas.

**Check whether the top of your sweep is degenerate.** At 2,000 characters this
corpus has 2,082 chunks for 2,012 papers — an abstract *is* about one chunk long,
so "BM25 prefers 2000" really means "BM25 would rather these were not chunked".
That is a statement about short documents, not a general result, and reporting it
as the latter would be the same error as reporting a benchmark whose queries were
built from the documents' own words without saying so.

> In code: `run_ablation.py`, figure `10_chunk_ablation.png`.

## 5. Latency is half the metric

A benchmark that reports only quality is answering half the question. Every arm
here was also timed, and the headline figure plots **nDCG against median
latency** — because the decision a team actually makes is not "which is best" but
"which is worth it".

Two measurement rules that are easy to get wrong:

- **Warm up first.** The first call to a dense retriever pays a one-off model load
  of tens of seconds. Including it does not make the arm look slow — it makes the
  *measurement wrong*.
- **Report a median, not a mean.** One scheduler hiccup can double a mean and tell
  you nothing about the arm.

## 6. What this project found — including where it was wrong

Four arms, 180 generated queries plus 11 hand-written ones:

| Arm | nDCG@10 | 95% CI | median latency |
|---|--:|:--:|--:|
| **BM25** (no model at all) | **0.978** | [0.962, 0.992] | **2.0 ms** |
| Cross-encoder rerank | 0.964 | [0.942, 0.982] | 2,465 ms |
| Hybrid (RRF) | 0.874 | [0.835, 0.911] | 18.3 ms |
| Dense (the shipped retriever) | 0.751 | [0.691, 0.811] | 17.4 ms |

Three lessons fall out, and none of them were the expected ones:

**The 1990s baseline beat the embedding model.** If you do not build a baseline,
you cannot know this — and almost no public RAG notebook builds one. "We used
embeddings and it worked" is not a finding; it is an untested default.

**The reranker was a bad buy.** An earlier version of this project's README
recommended a cross-encoder as "the obvious next iteration" that "would lift MRR".
It costs ~1,200× BM25's latency to buy **no measurable quality at all** (§4b).
Building it is what refuted the recommendation. A reranker earns its cost when
first-stage recall is high but the *ordering* is poor; here the ordering was
already near-optimal and there was nothing left to rescue.

**Fusion is not free insurance.** Hybrid RRF scored *below* BM25 alone. On one
curated question it inherited BM25's failure rather than dense's success —
fusing a confident-but-wrong ranking with a correct one pushed the right document
out of the top 10.

## 7. The aggregate always hides something — go look

The single most useful thing in this benchmark is not in the table above.

Question `s5` asks what **word2vec** learns. The correct paper is *"Efficient
Estimation of Word Representations in Vector Space"* — and the string `word2vec`
appears **zero times in all 4,298 chunks**. BM25 scores 0.00; it cannot match a
term that is not there. The dense arm scores 0.387 — it puts the right paper
around rank 5 — because *word2vec* and *word representations in vector space* sit
near each other in embedding space. A mediocre ranking beats an impossible one.

This is worth flagging as a **replication**, not just an anecdote. The corpus was
later rebuilt from scratch with zero paper overlap and the eval set regenerated;
`s5` behaved identically both times (BM25 0.00, dense non-zero, hybrid 0.00). A
finding that survives a complete change of corpus is a different class of evidence
from one observed once.

That is the canonical **vocabulary mismatch** case — the exact problem dense
retrieval was invented to solve — showing up live in an evaluation whose headline
says dense retrieval lost.

So the honest recommendation is **conditional**: BM25 for keyword-shaped queries
over a corpus whose vocabulary matches its users', with a dense fallback for the
terminology-mismatch case it alone survives. Not "BM25 wins".

**The general lesson: two arms with similar averages can fail on completely
different questions.** Always look at the per-question rows before you ship the
mean.

## 8. Your evaluation set will lie to you — say how

Every generated query in this benchmark was built from its target document's own
words, which structurally favours a lexical matcher. Pretending otherwise would
make the headline number dishonest.

Two things make it survivable, and both are design choices rather than excuses:

1. **Three query styles instead of one.** Fluent abstract sentence → title →
   shuffled keyword bag. Measuring an arm across the gradient measures how much it
   depends on well-formed prose. Dense fell 0.969 → 0.291; BM25 moved 0.065. That
   *slope* is a more robust finding than any single point.
2. **A control set with different provenance.** The 11 hand-written questions were
   written by a person in their own words, so they do not share the bias. They
   agree with the generated set — which is the only reason the headline claim gets
   stated at all.

The first version of the generated set used only abstract sentences and was
**useless**: every arm scored ~1.00, because a sentence lifted from an abstract is
a literal substring of the passage it targets. A benchmark where everything ties
has measured nothing. Noticing that and fixing it is most of the work.

> Written up in full in [`eval/ADJUDICATION.md`](../eval/ADJUDICATION.md).

## 9. Indexes go stale — and the obvious way to detect it does not work

Every number here describes an index built on one day, over a literature that does
not stop. The naive drift probe — "run queries about new papers and watch scores
fall" — measures nothing, because those papers are not in the index and *every*
arm scores zero.

The design that seems to work is comparing **top-1 similarity for queries about
indexed papers** against **queries about newer unseen ones**:

```
relative_drop = (indexed_similarity − newer_similarity) / indexed_similarity
```

This project measured 0.730 vs 0.532 — a **27.2% drop** — set a 15% trigger, and
recommended re-indexing. Then it re-indexed, and that is where the lesson is.

### What acting on the recommendation revealed

The rebuilt index shared **not one paper** with the old one. On that
one-hour-old index, the same probe read **26.7%**.

The largest freshness intervention that exists moved the metric by half a
percentage point. So the metric was not measuring staleness. It never had been.

**Why:** a baseline query is a sentence lifted verbatim out of a document *in* the
index, so its best match is a near-self-match by construction. A probe query comes
from a document that is absent by definition, so its best match is the closest
topical neighbour. The gap between "verbatim self-match" and "nearest neighbour"
is a property of **the measurement**, present in full on day one. A threshold on
the raw drop fires on every index, always, and says nothing about any of them.

### The fix, and the general lesson

Measure the **floor** — what the probe reads on a freshly built index — and
trigger on the *excess over it*:

```
excess = relative_drop − floor        # floor = 0.267, measured not assumed
re-index when excess > 0.10
```

The current index reads +0.05 percentage points over the floor. Nothing to do.

Three transferable points:

1. **A monitoring metric that cannot go down is not a monitoring metric.** Before
   trusting one, ask what it reads in the state you are trying to detect the
   absence of. If you cannot answer, you have not calibrated it — you have just
   picked a threshold.
2. **The floor has to be measured, and it expires.** Change the embedder, the
   chunk size or the corpus size and it moves, while the constant in your config
   does not. That makes it the most fragile number in the project, which is worth
   writing down next to it.
3. **Two readings are not a trend.** The excess-over-floor rule is the right
   *shape*; it has been evaluated at two points in time. What would justify the
   tolerance is a series of probes over months, which this project does not have
   and does not claim.

And the threshold is still a **stated decision, not a property of the data**:
re-embedding this corpus takes a few minutes, so the tolerance can be tight. Where
re-indexing is an overnight GPU job, the same measurement justifies a looser one.

> The mistake was findable from the start: the module's own docstring always said
> "the gap *widening* is the signal", and the code thresholded the level anyway.
> Prose and code disagreed, and the code was the one being run. Worth a look at
> your own repo for the same shape of bug.

## 10. Applying this to benchmarks you did not run

The same checklist works on someone else's numbers, and it is worth using,
because popularity is not evidence.

The most-voted RAG notebook on Kaggle has **36,657 votes** — roughly fifteen times
the next one. It is also a **Kaggle Learn course exercise**, which means that
number is largely *course completions*, not readers judging quality. Treating it
as a leaderboard is a category error. What it actually does is the commodity
pattern: wire an embedding model to a vector store, retrieve top-k, hand it to an
LLM, show one impressive answer. No baseline, no interval, no latency, no
per-question inspection.

That is not a criticism of a teaching notebook doing its job. It is a warning
about what a vote count measures. Four questions to ask of any reported retrieval
result:

1. **What is it compared against?** If there is no BM25 baseline, "our embeddings
   work" is untested — this project's own numbers show how that can go.
2. **What is the interval?** A mean over a handful of questions decides nothing.
3. **What did it cost?** Quality with no latency is half a result.
4. **Whose labels, made how?** If the evaluation set's construction is not
   written down, its biases are not absent — just undisclosed.

## The one idea

**A retrieval number without a baseline, an interval, and a latency is a claim,
not a measurement — and the per-question rows will usually tell you something the
average hid.**
