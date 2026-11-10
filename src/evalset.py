"""Generate the known-item evaluation set — 120 graded queries, no API key.

The original evaluation set is 11 hand-written questions (`eval/qa_set.json`).
They are good questions, and they stay. But 11 is too few to put a confidence
interval on: a single question flipping moves the mean by nine points, so any
"arm A beats arm B" claim over 11 rows is noise wearing a decimal point.

This module builds a larger set the only way it honestly can without a human
adjudicator or an API key: **known-item retrieval**. The construction is:

    take a paper -> derive a query from it -> the right answer is that paper

This is a standard information-retrieval evaluation design (it is how much of
the TREC known-item and BEIR-style corpora are built): the relevance judgement
is not a matter of opinion, because the query was *manufactured from* the
document it points to. That gives 120 unambiguous labels for free.

A known-item set built one way would be a single, arguable data point. So every
sampled paper yields **three queries, in three styles**, reported separately:

  * **abstract** — the longest sentence from the abstract. Fluent natural prose,
    and a verbatim substring of the passage it targets. The *easiest* possible
    query, and it turns out every arm solves it.
  * **title** — the paper's title. Still natural language, still literal overlap,
    but much shorter and denser in content words.
  * **keyword** — five content words sampled from across the abstract, stop-words
    removed and word order shuffled. No grammar, not a substring of any single
    passage. This is what people actually type into a search box, and it is the
    furthest thing here from the fluent sentences a sentence-embedding model was
    trained on.

The three styles form a gradient, and comparing one arm across them measures
something no single number can: **how much a retriever's quality depends on the
query being well-formed prose.** BM25 does not read word order at all, so it
should be nearly indifferent. A bi-encoder trained on natural sentences has every
reason not to be. That contrast is the finding, not a caveat — and it is why the
set is generated three ways instead of the one that would have looked tidiest.

`eval/ADJUDICATION.md` writes up what these labels are worth and where they
mislead.

Everything here is seeded with `config.SEED`, so the generated set is byte-identical
on every machine and every run.
"""

from __future__ import annotations

import json
import random
import re

from . import config
from .acquisition import Paper, load_papers

# A query is one sentence. Splitting on sentence-ending punctuation followed by
# whitespace and a capital letter is crude, but abstracts are well-punctuated
# prose and this avoids a heavyweight NLP dependency for a job that does not need
# one. Over-splitting is self-correcting: fragments below MIN_QUERY_CHARS are
# dropped anyway.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Sentences that begin like these identify the paper's *housekeeping*, not its
# content — every paper has them, so they point at no particular document and
# would be unanswerable by any retriever. Dropping them keeps the ground truth
# meaningful rather than making the benchmark artificially hard for everyone.
_BOILERPLATE_PREFIXES = (
    "we release", "our code", "code is", "code and", "the code",
    "data and", "we make", "all code", "we publicly", "we open",
    "in this paper", "in this work",
)

# Words too common to identify a paper. Used only to build the *keyword* query
# style — never by the retrievers themselves, where BM25's IDF already handles
# this and a hand-curated stop list would be an untunable thumb on the scale.
# Includes the academic-writing filler ("propose", "novel", "results") that
# appears in nearly every abstract in the corpus.
_STOPWORDS = frozenset("""
a an and are as at be been between both by can each for from had has have if in
into is it its may more most not of on or other our over paper propose proposed
result results show shows study such than that the their then these this those
three to two under use used using via was we were when where which while with
approach method methods new novel work
""".split())

# A content word: letters and hyphens, at least three characters. Numbers and
# single letters are dropped — "2" and "x" identify nothing.
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def _sentences(abstract: str) -> list[str]:
    """Split an abstract into candidate query sentences, longest-first.

    Longest-first because a longer sentence carries more of the paper's specific
    content, and specificity is exactly what makes a known-item query solvable.
    """
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(abstract) if s.strip()]
    keep = [
        s for s in parts
        if len(s) >= config.MIN_QUERY_CHARS
        and not s.lower().startswith(_BOILERPLATE_PREFIXES)
    ]
    return sorted(keep, key=len, reverse=True)


def _keyword_query(abstract: str, seed: int) -> str:
    """A short, unordered bag of content words drawn from the abstract.

    Deliberately *not* a sentence. Word order is shuffled and grammar is gone, so
    this is the query style furthest from the natural prose a sentence-embedding
    model was trained on — and closest to what people actually type into a search
    box. Seeded per paper, so the same paper always yields the same query.

    Sampling from the whole abstract rather than one sentence also means the
    query is not a substring of any single passage, which is what stops it from
    being a verbatim-match exercise.
    """
    rng = random.Random(seed)
    words: list[str] = []
    for w in _WORD.findall(abstract.lower()):
        if w not in _STOPWORDS and w not in words:
            words.append(w)
    if len(words) > config.KEYWORD_QUERY_WORDS:
        words = rng.sample(words, config.KEYWORD_QUERY_WORDS)
    rng.shuffle(words)
    return " ".join(words)


def _eligible(paper: Paper) -> bool:
    """True when a paper can yield both query flavours.

    A paper needs a usable title and at least one substantial, non-boilerplate
    abstract sentence. Requiring *both* flavours from every sampled paper keeps
    the two slices perfectly paired — same papers, same difficulty, only the
    query style differs — which is what makes the title-vs-abstract comparison a
    controlled one rather than two unrelated samples.
    """
    return len(paper.title) >= config.MIN_QUERY_CHARS and bool(_sentences(paper.abstract))


def build_known_item_set(papers: list[Paper] | None = None) -> list[dict]:
    """Build the known-item query set. Deterministic under `config.SEED`.

    Returns one dict per query, in the same shape the hand-written
    `eval/qa_set.json` uses (`qid`, `question`, `relevant_ids`, `hop`) so the
    existing metric code consumes both sets without a special case — plus a
    `flavour` field the benchmark slices on.

    Every eligible paper contributes one query in each of the three styles, so
    the slices are perfectly paired: same papers, same targets, only the query
    style differs. That pairing is what makes comparing an arm across styles a
    controlled measurement rather than three unrelated samples.

    `hop` is "single" for every generated query: a known-item query is answered
    by exactly one paper, by construction. Multi-hop questions cannot be
    manufactured this way, which is precisely why the 11 hand-written ones
    remain the harder slice and are reported separately.
    """
    if papers is None:
        papers = load_papers()

    eligible = sorted((p for p in papers if _eligible(p)), key=lambda p: p.arxiv_id)
    # Sort before sampling: `load_papers` returns whatever order the JSONL was
    # written in, and a seeded sample over an unsorted list is only reproducible
    # if the input order is too. Sorting by id pins it.

    rng = random.Random(config.SEED)
    n = min(config.KNOWN_ITEM_PAPERS, len(eligible))
    sampled = rng.sample(eligible, n)

    rows: list[dict] = []
    for i, paper in enumerate(sampled):
        rows.append(
            {
                "qid": f"kt{i:03d}",
                "question": paper.title,
                "relevant_ids": [paper.arxiv_id],
                "hop": "single",
                "flavour": "title",
            }
        )
        rows.append(
            {
                "qid": f"ka{i:03d}",
                "question": _sentences(paper.abstract)[0],
                "relevant_ids": [paper.arxiv_id],
                "hop": "single",
                "flavour": "abstract",
            }
        )
        rows.append(
            {
                "qid": f"kk{i:03d}",
                "question": _keyword_query(paper.abstract, config.SEED + i),
                "relevant_ids": [paper.arxiv_id],
                "hop": "single",
                "flavour": "keyword",
            }
        )
    return rows


def save_known_item_set(rows: list[dict]) -> None:
    """Write the generated set to `eval/known_item_set.json`.

    This file *is* committed, unlike the corpus. It is small, and every published
    number depends on it — a reviewer who wants to check a claim needs to see the
    exact queries it was measured on, not a promise that they were regenerable.
    """
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.KNOWN_ITEM_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def load_known_item_set() -> list[dict]:
    """Load the committed known-item set, generating it if absent."""
    if not config.KNOWN_ITEM_FILE.exists():
        rows = build_known_item_set()
        save_known_item_set(rows)
        return rows
    with open(config.KNOWN_ITEM_FILE, encoding="utf-8") as f:
        return json.load(f)
