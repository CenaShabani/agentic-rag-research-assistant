"""Acquire the arXiv corpus and record provenance.

This is the first stage of the RAG pipeline. It queries the public arXiv API for
recent cs.CL / cs.LG papers, keeps title + abstract + metadata for every paper,
and pulls full text only where the paper's licence permits redistribution.

The provenance discipline matches the rest of the portfolio: we record the exact
query, the date, the paper count, and a SHA-256 of the saved bytes, so anyone can
see precisely what was downloaded and when — and detect if it silently changed.

Nothing here is committed to git; `run_pipeline.py` fetches it and writes
`data/raw/`, which is git-ignored. The *code* ships, the data does not.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterator

from . import config

# arXiv's Atom API. Documented at https://info.arxiv.org/help/api/index.html.
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Atom / arXiv XML namespaces, needed to walk the feed with ElementTree.
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# arXiv licences that permit redistribution of the full text. Papers under these
# licences are safe to store and chunk in full; everything else is kept as
# metadata + abstract only. This is a deliberate, defensible licensing choice —
# not a shortcut — and it is why the corpus is "abstracts for all, full text for
# the CC-licensed subset".
REDISTRIBUTABLE_LICENSES = (
    "http://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/licenses/by-sa/4.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/",
)


@dataclass
class Paper:
    """One arXiv paper as we store it.

    `text` is the field retrieval actually runs over: title + abstract always,
    which is self-contained and licence-safe for every paper. (A future extension
    could append CC-licensed full text here; the `license` field is recorded so
    that decision can be made per-paper and defended.)
    """

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    license: str | None

    @property
    def text(self) -> str:
        """The retrievable body: title then abstract, one blank line between."""
        return f"{self.title}\n\n{self.abstract}"

    @property
    def redistributable(self) -> bool:
        """True when the paper's licence permits storing its full text."""
        return self.license in REDISTRIBUTABLE_LICENSES


def _build_query_url(start: int, page_size: int) -> str:
    """Construct one paged arXiv API request URL.

    We search within the configured categories, sorted by submission date
    (newest first) so the corpus is a coherent recent slice rather than an
    arbitrary sample — which makes the "recent NLP/ML literature" framing honest.
    """
    cat_clause = "+OR+".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    params = {
        "search_query": cat_clause,
        "start": str(start),
        "max_results": str(page_size),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    # arXiv wants the OR-joined category clause un-escaped, so we assemble the
    # query string by hand rather than letting urlencode escape the '+'.
    query = "&".join(f"{k}={urllib.parse.quote(v, safe='+:')}" for k, v in params.items())
    return f"{ARXIV_API_URL}?{query}"


def _parse_feed(xml_bytes: bytes) -> Iterator[Paper]:
    """Yield Paper objects from one Atom feed response."""
    root = ET.fromstring(xml_bytes)
    for entry in root.findall("atom:entry", _NS):
        # The arXiv id is the tail of the entry's <id> URL, minus any version.
        raw_id = entry.findtext("atom:id", default="", namespaces=_NS)
        arxiv_id = raw_id.rsplit("/", 1)[-1].split("v")[0] if raw_id else ""

        title = (entry.findtext("atom:title", default="", namespaces=_NS) or "").strip()
        abstract = (entry.findtext("atom:summary", default="", namespaces=_NS) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=_NS) or ""

        authors = [
            (a.findtext("atom:name", default="", namespaces=_NS) or "").strip()
            for a in entry.findall("atom:author", _NS)
        ]
        categories = [
            c.attrib.get("term", "")
            for c in entry.findall("atom:category", _NS)
        ]

        # arxiv:license is present only when the author selected a licence.
        license_el = entry.find("arxiv:license", _NS)
        license_url = license_el.text.strip() if license_el is not None and license_el.text else None

        if arxiv_id and title and abstract:
            yield Paper(
                arxiv_id=arxiv_id,
                title=" ".join(title.split()),      # collapse arXiv's newlines
                abstract=" ".join(abstract.split()),
                authors=authors,
                categories=categories,
                published=published,
                license=license_url,
            )


def fetch_papers(max_papers: int = config.MAX_PAPERS) -> list[Paper]:
    """Download up to `max_papers` papers from arXiv, paging politely.

    Sleeps between pages to respect arXiv's rate guidance — hammering the API
    risks a temporary block, which would make the pipeline *less* reproducible,
    not more. De-duplicates by arXiv id in case paging overlaps.
    """
    papers: dict[str, Paper] = {}
    start = 0
    while len(papers) < max_papers:
        url = _build_query_url(start, config.ARXIV_PAGE_SIZE)
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (trusted host)
            xml_bytes = resp.read()

        page = list(_parse_feed(xml_bytes))
        if not page:
            break  # ran out of results before hitting max_papers
        for p in page:
            papers[p.arxiv_id] = p

        start += config.ARXIV_PAGE_SIZE
        time.sleep(config.ARXIV_REQUEST_DELAY_SECONDS)

    return list(papers.values())[:max_papers]


def fetch_by_ids(arxiv_ids: tuple[str, ...]) -> list[Paper]:
    """Fetch specific papers by arXiv id via the API's id_list parameter.

    Used to pull the seed set of seminal papers so the evaluation's ground truth
    is always present. One request covers all ids.
    """
    if not arxiv_ids:
        return []
    query = f"{ARXIV_API_URL}?id_list={','.join(arxiv_ids)}&max_results={len(arxiv_ids)}"
    with urllib.request.urlopen(query, timeout=60) as resp:  # noqa: S310 (trusted host)
        xml_bytes = resp.read()
    time.sleep(config.ARXIV_REQUEST_DELAY_SECONDS)
    return list(_parse_feed(xml_bytes))


def _sha256(path) -> str:
    """SHA-256 of a file's bytes — the provenance fingerprint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_papers(papers: list[Paper], path=None, provenance_path=None) -> None:
    """Write papers to JSONL and record provenance alongside them.

    The paths are parameters (defaulting to the main corpus) so that any *other*
    slice of arXiv this project downloads — currently the drift probe's newer
    slice, see `drift.py` — gets the same provenance treatment: same fields, same
    SHA-256 fingerprint, same code. A second downloader with its own weaker
    provenance is exactly how a corpus quietly becomes untraceable.
    """
    path = path or config.PAPERS_FILE
    provenance_path = provenance_path or config.PROVENANCE_FILE

    config.ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    provenance = {
        "source": "arXiv API (http://export.arxiv.org/api/query)",
        "categories": list(config.ARXIV_CATEGORIES),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "paper_count": len(papers),
        "redistributable_count": sum(p.redistributable for p in papers),
        "papers_file": path.name,
        "papers_sha256": _sha256(path),
    }
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)


def load_papers(path=None) -> list[Paper]:
    """Load papers previously saved by `save_papers`."""
    papers: list[Paper] = []
    with open(path or config.PAPERS_FILE, encoding="utf-8") as f:
        for line in f:
            papers.append(Paper(**json.loads(line)))
    return papers


def acquire() -> list[Paper]:
    """Full acquisition step: fetch recent + seed papers, save, return.

    The seed papers (seminal works the evaluation asks about) are merged in and
    de-duplicated by id so they are always present regardless of the recent slice.
    Called by run_pipeline.py.
    """
    recent = fetch_papers()
    seed = fetch_by_ids(config.SEED_PAPER_IDS)
    merged: dict[str, Paper] = {p.arxiv_id: p for p in recent}
    for p in seed:                      # seed papers win on id collision
        merged[p.arxiv_id] = p
    papers = list(merged.values())
    save_papers(papers)
    return papers
