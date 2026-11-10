"""Split documents into overlapping chunks for retrieval.

Why chunk at all? An embedding model turns a piece of text into ONE vector. Feed
it a whole 4,000-word paper and that single vector has to average away every
distinct idea in the paper — retrieval on it is mush. Feed it a few sentences and
the vector actually means something specific, so a query about that specific idea
lands on it. Chunking is what makes semantic retrieval precise.

The two knobs — chunk size and overlap — are named constants in `config.py` with
their rationale. This module is deliberately simple and character-based (no
tokenizer dependency) so the whole retrieval path stays reproducible and easy to
reason about; every chunk carries its parent paper's metadata so an answer can
cite the paper it came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config
from .acquisition import Paper


@dataclass
class Chunk:
    """One retrievable unit of text plus the metadata needed to cite it."""

    chunk_id: str        # e.g. "2401.01234:2" — paper id + position
    arxiv_id: str
    title: str
    text: str
    position: int        # 0-based index of this chunk within its paper


def split_text(text: str,
               size: int = config.CHUNK_SIZE_CHARS,
               overlap: int = config.CHUNK_OVERLAP_CHARS) -> list[str]:
    """Slice `text` into overlapping windows of `size` characters.

    Each window starts `size - overlap` characters after the previous one, so
    consecutive chunks share `overlap` characters. That shared margin is what
    stops a fact that straddles a boundary from being lost — it survives in at
    least one chunk. We try to end each window at a whitespace boundary so chunks
    don't split mid-word, which would corrupt the embedding.

    Guarding the two edge cases explicitly: overlap must be smaller than size (or
    the window never advances and we loop forever), and text shorter than one
    window is returned as a single chunk.
    """
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Prefer to break at the last space in the window so we don't cut a word
        # in half — but only if that space isn't so early it shrinks the chunk to
        # nothing (which would happen on long unbroken strings like URLs).
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start + size // 2:
                end = last_space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        # Advance so the next window overlaps the current one by `overlap` chars.
        start = end - overlap
    return [c for c in chunks if c]


def chunk_paper(paper: Paper,
                size: int = config.CHUNK_SIZE_CHARS,
                overlap: int = config.CHUNK_OVERLAP_CHARS) -> list[Chunk]:
    """Turn one paper into its list of Chunks, carrying metadata through."""
    return [
        Chunk(
            chunk_id=f"{paper.arxiv_id}:{i}",
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            text=piece,
            position=i,
        )
        for i, piece in enumerate(split_text(paper.text, size=size, overlap=overlap))
    ]


def chunk_papers(papers: list[Paper],
                 size: int = config.CHUNK_SIZE_CHARS,
                 overlap: int = config.CHUNK_OVERLAP_CHARS) -> list[Chunk]:
    """Chunk every paper in the corpus.

    `size` and `overlap` default to the shipped constants and are parameters only
    so `run_ablation.py` can rebuild the corpus at other settings and measure what
    the constants are worth. The pipeline never passes them — there is still one
    chunking configuration in the shipped system, defined in `config.py`.
    """
    chunks: list[Chunk] = []
    for p in papers:
        chunks.extend(chunk_paper(p, size=size, overlap=overlap))
    return chunks


def chunks_to_frame(chunks: list[Chunk]) -> pd.DataFrame:
    """Chunks → DataFrame, the on-disk form and the shape the vector store wants."""
    return pd.DataFrame(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "arxiv_id": [c.arxiv_id for c in chunks],
            "title": [c.title for c in chunks],
            "text": [c.text for c in chunks],
            "position": [c.position for c in chunks],
        }
    )


def save_chunks(chunks: list[Chunk]) -> pd.DataFrame:
    """Chunk the corpus to parquet and return the frame. Called by the pipeline."""
    config.ensure_dirs()
    frame = chunks_to_frame(chunks)
    frame.to_parquet(config.CHUNKS_FILE, index=False)
    return frame


def load_chunks() -> pd.DataFrame:
    """Load the chunk frame written by `save_chunks`."""
    return pd.read_parquet(config.CHUNKS_FILE)
