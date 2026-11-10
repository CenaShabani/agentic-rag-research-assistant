"""The deck's content: one function per slide, in presentation order.

Split out from `build_deck.py` so that file stays what it is — the layout engine —
and this one stays what it is: the argument. When a number changes, it changes
here, and every image referenced is a filename resolved under `outputs/figures/`
by `build_deck.figure`, which raises if it is missing.

Standard: ../../../05-communication/context/project-deck.md
"""
from __future__ import annotations

TITLE_CLAIM = (
    "A 1990s keyword baseline beat the embedding\n"
    "model this system ships - at 1/8 the latency"
)

TITLE_SUB = (
    "Agentic RAG Research Assistant  ·  4 retrieval arms, 191 questions, "
    "paired tests  ·  Cena Shabani"
)

TITLE_NOTES = (
    "Open on the finding, not the topic. Almost every public RAG write-up wires an embedding "
    "model to a vector store and shows one impressive answer. None of them build a baseline. "
    "I built one, and it won - nDCG 0.978 against 0.751, at two milliseconds against seventeen. "
    "And because every arm answered the same questions, I could test whether that gap is real "
    "rather than asking you to eyeball two error bars. That is the talk."
)

#: (title, body, notes, figure or None), in order.
SLIDES: list[tuple[str, str, str, str | None]] = [
    (
        "The decision: which retrieval strategy is worth its latency?",
        "A RAG demo is a commodity. A defensible evaluation of\none is not.\n\n"
        "Retrieval is the half nobody measures - no baseline to\nbeat, no confidence interval, "
        "no significance test, no\nlatency, no per-question look.\n\n"
        "So: four retrieval strategies, each one idea apart from\nthe last, holding the corpus, "
        "questions and metrics\nconstant - and price each one.\n\n"
        "dense  ·  bm25  ·  hybrid RRF  ·  cross-encoder rerank",
        "Frame it as a spending decision, because that is what it is. Every millisecond of "
        "retrieval is latency on every query in production, and every extra component is "
        "something to operate. I wanted the answer to be a number rather than a preference - "
        "and I was expecting the expensive options to win. Each arm adds exactly one idea to the "
        "previous one, so a difference has a cause; an arm that changed two things at once would "
        "be unattributable. The dense arm delegates to the shipped retriever rather than "
        "reimplementing it, and a test asserts they return identical hits - otherwise every "
        "number would describe a retriever the system does not actually use.",
        None,
    ),
    (
        "2,012 arXiv papers, with the ground truth deliberately seeded",
        "· cs.CL and cs.LG from the arXiv API, plus 12 seminal\n  papers fetched by id\n\n"
        "· Title + abstract only - licence-safe for every paper,\n  and each paper's licence "
        "is recorded anyway\n\n"
        "· 4,298 chunks, 384-dim local embeddings, exact FAISS\n\n"
        "· Provenance: query, date, count and SHA-256 of the file\n\n"
        "· 191 questions: 180 generated known-item + 11 written\n  by hand, reported separately",
        "The seeding is the part worth explaining. The evaluation questions have known-correct "
        "answers, and those answers have to be in the corpus or the retrieval metrics measure "
        "nothing. A recent-papers-only crawl might not contain the Transformer paper. "
        "It turned out to matter more than I expected: when I rebuilt the index sixteen days "
        "later, not one of sixty sampled papers had survived - the recent window rotates "
        "completely. Only the twelve pinned papers persisted, which is what let the hand-written "
        "questions keep meaning the same thing across the rebuild. Licences are recorded per "
        "paper so that extending to full text later stays defensible.",
        "deck_corpus_categories.png",
    ),
    (
        "BM25 wins on quality and costs 1,200x less than the reranker",
        "BM25      nDCG@10 0.978  [0.962, 0.992]        2.0 ms\n\n"
        "rerank    nDCG@10 0.964  [0.942, 0.982]    2,465 ms\n\n"
        "hybrid    nDCG@10 0.874  [0.835, 0.911]      18.3 ms\n\n"
        "dense     nDCG@10 0.751  [0.691, 0.811]      17.4 ms\n\n"
        "BM25 is the top-left point. Nothing else buys quality it\ndoes not already have, and "
        "everything else costs more.",
        "Reporting the intervals is the point - a benchmark without them is four bars and a "
        "story. But do not let me get away with reading them against each other, and that is "
        "exactly the next slide. The reranker line is the one I would dwell on: an earlier "
        "version of this project's README recommended a cross-encoder as the obvious next "
        "iteration that would lift MRR. I built it, and building the thing is what refuted my own "
        "recommendation - which is the argument for benchmarking rather than reasoning about it.",
        "05_quality_vs_latency.png",
    ),
    (
        "Two overlapping intervals are not a comparison - so I paired them",
        "Every arm answered the same 180 questions, so the\nper-question difference exists. "
        "Question difficulty is\nshared between arms, so it cancels.\n\n"
        "Sign-flip randomization test, Holm-corrected over all\nsix pairs:\n\n"
        "bm25 - dense       +0.227    52W / 1L    p = 0.0006\n"
        "hybrid - dense     +0.122    46W / 0L    p = 0.0006\n"
        "bm25 - rerank      +0.014     8W / 6L    p = 0.101\n\n"
        "The cross-encoder buys nothing measurable, at 1,200x\nthe latency.",
        "This is the slide I would want a retrieval team to ask about. Comparing two marginal "
        "confidence intervals and checking whether they overlap is the standard mistake, and I "
        "nearly published it: overlapping intervals do not imply no difference, the implication "
        "only runs the other way. The intervals are wide because questions differ in difficulty, "
        "and that variance is shared by both arms, so it cancels in the per-question difference. "
        "Paired, the dense gap is fifty-two wins to one loss. The reranker gap is eight wins to "
        "six losses over a hundred and sixty-six ties, and it does not survive correction. So my "
        "claim is a null, correctly stated - and it is stronger than what I had before. Slightly "
        "better invites an argument about whether it is worth paying for. No measurable "
        "difference at a thousand times the price ends the argument.",
        "08_arm_significance.png",
    ),
    (
        "Why: the dense arm needs the query to be well-formed prose",
        "Every paper produced a query in three styles - a fluent\nabstract sentence, a title, "
        "and a shuffled bag of five\nkeywords.\n\n"
        "BM25 never reads word order, so it barely moves: -0.065\n\n"
        "The dense arm loses two-thirds of its quality:\n"
        "0.969 → 0.291\n\n"
        "Real users type keywords, not sentences.",
        "This is the slide that shows I understood the result rather than just measured it. "
        "A bi-encoder is trained on natural sentences, so it has no particular reason to handle "
        "a keyword bag well - and it doesn't. The slope is the measurement here, not any single "
        "point, which is why the eval set was generated three ways instead of the one that would "
        "have looked tidiest. My first version used only abstract sentences and every arm scored "
        "1.00 - a saturated benchmark measures nothing, and noticing that was most of the work. "
        "The cutoff sweep adds the mechanism: the dense arm's recall is still only 0.856 at k of "
        "fifty, so it is not mis-ranking documents, it is never retrieving them - which is also "
        "why no reranker can rescue it.",
        "06_query_style_robustness.png",
    ),
    (
        "The average hides the row that decides the architecture",
        "s5 asks what word2vec learns.\n\n"
        "The right paper is ‘Efficient Estimation of Word\nRepresentations in Vector Space’ - "
        "and the string\n‘word2vec’ appears zero times in all 4,298 chunks.\n\n"
        "BM25 scores 0.00. The dense arm finds it, around rank 6.\n\n"
        "Hybrid RRF fails it too - fusion is not free insurance.\n\n"
        "So: BM25 as the default, dense as a fallback.\nNot ‘BM25 wins’.",
        "This is the most useful thing in the whole benchmark and it is invisible in the average. "
        "It is the canonical vocabulary-mismatch case - the exact problem dense retrieval was "
        "invented to solve - turning up live in an evaluation whose headline says dense retrieval "
        "lost. Rare here, catastrophic when it happens, which argues for a fallback rather than a "
        "replacement. Note hybrid loses it as well: fusing a confident-but-wrong ranking with a "
        "correct one pushed the right document out of the top ten. And this one replicated "
        "exactly when I rebuilt the corpus from scratch with zero paper overlap. If someone asks "
        "me one question about this project, I want it to be this slide.",
        None,
    ),
    (
        "My drift monitor was broken, and acting on it is what proved it",
        "The probe read a 27.2% similarity drop against a 15%\nre-index trigger. So I "
        "re-indexed.\n\n"
        "On an index one hour old, it read 26.7%.\n\n"
        "The drop has a structural floor: a baseline query is a\nsentence lifted out of an "
        "indexed document, so it\nnear-self-matches. That gap is there on day one.\n\n"
        "Now: trigger on excess over a measured floor.\nCurrent reading: +0.05pp. Nothing to do.",
        "I would rather present this than hide it. I had a monitoring metric with a threshold, it "
        "fired, I acted on it, and the largest possible intervention - replacing every paper in "
        "the corpus - moved the number by half a percentage point. That is the metric failing, "
        "not the index. The general lesson is the transferable part: a monitoring metric you have "
        "never evaluated in the healthy state is not calibrated, it just has a threshold on it. "
        "And the bug was findable from the start - the module's own docstring always said the gap "
        "widening was the signal, and the code thresholded the level anyway. The prose was right "
        "and unread, the code was wrong and trusted. The fix costs something too: the floor is a "
        "measured number that expires whenever the embedder or the corpus size changes, which is "
        "written down next to it in config.",
        "07_drift.png",
    ),
    (
        "What I'd do about it",
        "1. Switch the first stage to BM25. 8x the latency for\n"
        "   0.227 less nDCG, 52 wins to 1 - a default nobody\n   tested.\n\n"
        "2. Do not buy the cross-encoder. If you keep it anyway,\n"
        "   cut its shortlist from 50 to 20 - recall saturates\n   there.\n\n"
        "3. Keep dense as a fallback for vocabulary mismatch -\n"
        "   and not via naive RRF, which loses that case too.\n\n"
        "4. Do not re-index. The probe is at its floor.\n\n"
        "5. The residual multi-hop gap is still the agent's case.",
        "The point of measuring is that the next step is not a guess - including when the number "
        "contradicts what the project was built to believe, and including when the number itself "
        "turns out to be the thing that was wrong. Item two comes straight from the cutoff sweep: "
        "hybrid recall hits one point zero by rank twenty, so candidates twenty-one to fifty are "
        "pure cost. Item four is the reversal of what this slide said a fortnight ago, and I "
        "would rather show you a recommendation I withdrew on evidence than a list that never "
        "changed.",
        None,
    ),
    (
        "Limitations I would raise before you do",
        "· Generated queries are built from their target\n  document's own words - that "
        "favours BM25. Mitigated by\n  the slope across three query styles, by the\n  "
        "hand-written set agreeing, and by replicating on a\n  corpus with zero paper overlap.\n\n"
        "· Known-item retrieval is the easy end of the problem.\n\n"
        "· The curated set is 11 questions - intervals overlap\n  completely, and it is too "
        "small for a paired test.\n\n"
        "· The drift trigger has two readings. Two is not a trend.\n\n"
        "· One corpus, one embedder, one CPU.\n\n"
        "pytest → 98 passed, 1 skipped  ·  runs with no API key",
        "Say these before being asked. The first one is the real attack surface and I would raise "
        "it myself: if you manufacture queries out of a document's own words, you have built "
        "something that flatters keyword matching. Three things make it survivable - the styles "
        "measure a slope rather than a point, the hand-written questions do not share the bias "
        "and agree, and the whole result replicated on a completely different corpus. If those "
        "had disagreed I would be presenting a different conclusion. Everything here reproduces "
        "offline with no key and no account, which is what makes the headline checkable rather "
        "than claimed.",
        None,
    ),
]
