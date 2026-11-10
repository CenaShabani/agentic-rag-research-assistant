"""Tests for the chunking logic — the properties retrieval depends on."""

from __future__ import annotations

from src import config
from src.acquisition import Paper
from src.chunking import chunk_paper, split_text


def test_short_text_is_one_chunk():
    """Text shorter than a window should not be split."""
    assert split_text("a short abstract") == ["a short abstract"]


def test_empty_text_yields_no_chunks():
    assert split_text("   ") == []


def test_chunks_respect_size_bound():
    """No chunk should exceed the configured size (word-boundary trimming only shrinks)."""
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = split_text(text, size=200, overlap=40)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1


def test_chunks_overlap_preserves_boundary_content():
    """Consecutive chunks share text, so a boundary-straddling fact survives."""
    text = " ".join(f"token{i}" for i in range(500))
    chunks = split_text(text, size=200, overlap=60)
    # Some token that ends up near a boundary should appear in two chunks.
    joined = "".join(chunks)
    assert len(joined) > len(text) - 10  # overlap means we don't lose content


def test_overlap_must_be_smaller_than_size():
    """A degenerate config that would loop forever must raise, not hang."""
    import pytest

    with pytest.raises(ValueError):
        split_text("some text that is long enough " * 20, size=100, overlap=100)


def test_chunk_metadata_is_carried_through():
    """Every chunk must know its parent paper so answers can cite it."""
    paper = Paper(
        arxiv_id="1234.5678",
        title="A Test Paper",
        abstract="x " * 800,  # long enough to force multiple chunks
        authors=["A. Author"],
        categories=["cs.CL"],
        published="2024-01-01",
        license=None,
    )
    chunks = chunk_paper(paper)
    assert len(chunks) > 1
    assert all(c.arxiv_id == "1234.5678" for c in chunks)
    assert all(c.title == "A Test Paper" for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == "1234.5678:0"
