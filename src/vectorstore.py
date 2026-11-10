"""The FAISS vector store — build, save, load, and search the index.

This is the single definition of "search the corpus" in the whole project. Both
the plain RAG pipeline (`rag.py`) and the agentic system (`agent.py`) call
`VectorStore.search`; neither re-implements retrieval. That is deliberate — if
there were two search paths they could disagree, and no published number could be
trusted. (Same "single source of truth" rule the rest of the portfolio applies to
its core logic.)

Why FAISS? It is an in-process, file-backed nearest-neighbour index — no server,
no Docker, no network. The whole thing runs on a laptop, which is what keeps the
retrieval path reproducible top-to-bottom in the capstone notebook.

Why an inner-product index? Our embeddings are L2-normalized (see embeddings.py),
so inner product == cosine similarity. `IndexFlatIP` does an *exact* search — it
compares the query against every chunk. For a few thousand chunks that is
instant, and it removes approximate-search recall as a variable, so retrieval
quality reflects the embeddings and chunking, not an ANN approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .embeddings import embed_query, embed_texts


@dataclass
class Retrieved:
    """One search hit: a chunk, its citation metadata, and its similarity score."""

    chunk_id: str
    arxiv_id: str
    title: str
    text: str
    score: float        # cosine similarity in [-1, 1]; higher is more similar


class VectorStore:
    """An in-memory FAISS index plus the row-aligned chunk metadata.

    The index stores only vectors; the metadata frame stores the text and
    citation fields. They are kept in the same row order, so FAISS returning row
    `i` means "chunk at metadata row `i`". Keeping them aligned is the one
    invariant this class protects.
    """

    def __init__(self, index, meta: pd.DataFrame):
        self._index = index
        self._meta = meta.reset_index(drop=True)

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def build(cls, chunks: pd.DataFrame) -> "VectorStore":
        """Embed every chunk and build the index. This is the expensive step."""
        import faiss

        vectors = embed_texts(chunks["text"].tolist())
        index = faiss.IndexFlatIP(config.EMBED_DIM)
        index.add(vectors)
        # Store exactly the columns needed to return and cite a hit.
        meta = chunks[["chunk_id", "arxiv_id", "title", "text"]].copy()
        return cls(index, meta)

    # ---- persistence ----------------------------------------------------- #
    def save(self) -> None:
        """Persist the index and metadata to `data/index/`."""
        import faiss

        config.ensure_dirs()
        faiss.write_index(self._index, str(config.FAISS_INDEX_FILE))
        self._meta.to_parquet(config.CHUNK_META_FILE, index=False)

    @classmethod
    def load(cls) -> "VectorStore":
        """Load a previously-saved index. Fast — no embedding required."""
        import faiss

        index = faiss.read_index(str(config.FAISS_INDEX_FILE))
        meta = pd.read_parquet(config.CHUNK_META_FILE)
        return cls(index, meta)

    # ---- search ---------------------------------------------------------- #
    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return int(self._index.ntotal)

    @property
    def meta(self) -> pd.DataFrame:
        """The row-aligned chunk metadata.

        Exposed so the benchmark's other retrieval arms (`retrievers.py`) can
        index the *same* chunks in the *same* row order as this index. Sharing
        the frame is what lets a lexical ranking and a dense ranking be fused by
        row id without a join — and what guarantees the arms are compared on an
        identical collection rather than two slightly different ones.
        """
        return self._meta

    def search_rows(self, query: str, k: int = config.TOP_K) -> list[tuple[int, float]]:
        """Return `(row, score)` pairs for the `k` nearest chunks, best first.

        The raw ranking, before metadata is attached. `search` builds on this,
        and so does reciprocal-rank fusion — which needs positions and row ids,
        not citation fields. Keeping both on top of one method means there is
        still exactly one place where this project asks FAISS a question.

        FAISS returns -1 for empty slots when `k` exceeds the index size; those
        are skipped rather than returned as rows.
        """
        q = embed_query(query)
        scores, idxs = self._index.search(q, k)
        return [
            (int(row), float(score))
            for score, row in zip(scores[0], idxs[0])
            if row >= 0
        ]

    def search(self, query: str, k: int = config.TOP_K) -> list[Retrieved]:
        """Return the `k` chunks most similar to `query`, best first.

        Embed the query, ask FAISS for the top-k by inner product, then map each
        returned row back to its chunk metadata.
        """
        hits: list[Retrieved] = []
        for row, score in self.search_rows(query, k=k):
            m = self._meta.iloc[row]
            hits.append(
                Retrieved(
                    chunk_id=m["chunk_id"],
                    arxiv_id=m["arxiv_id"],
                    title=m["title"],
                    text=m["text"],
                    score=float(score),
                )
            )
        return hits
