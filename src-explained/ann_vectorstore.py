from render_explained import build

build(
    "../src/vectorstore.py",
    "vectorstore.py_explained.pdf",
    "The FAISS vector store — the one place the corpus is searched",
    "This module builds, saves, loads, and searches the <b>FAISS</b> index. It is the single "
    "definition of retrieval in the project: both the naive and agentic paths call "
    "<b>VectorStore.search</b>, so a measured difference between them is always about strategy, "
    "never an accidental difference in search.",
    [
        (19, "Lines 1 to 19 are the module docstring. They make three arguments: this is the single "
             "source of truth for search (both pipelines call it, so they cannot disagree); FAISS is "
             "chosen because it is in-process and file-backed with no server or Docker; and an "
             "inner-product index is used because the embeddings are normalized, so inner product "
             "equals cosine similarity, and the exact IndexFlatIP removes approximate-search recall "
             "as a confounding variable."),
        (10, "Lines 20 to 29 are imports. Line 21 enables postponed annotations. Line 23 imports "
             "dataclass for the Retrieved type. Lines 25 to 26 import numpy and pandas (arrays and "
             "the metadata frame). Line 28 imports config; line 29 imports the two embedding "
             "helpers, embed_query and embed_texts, that turn text into the vectors FAISS stores "
             "and searches."),
        (11, "Lines 30 to 40 define the Retrieved dataclass — one search hit. Its fields are the "
             "chunk_id (line 36), the arxiv_id and title used for citation (lines 37 to 38), the "
             "chunk text itself (line 39), and the similarity score (line 40, cosine in the range "
             "minus one to one, higher meaning more similar). The comment on line 40 documents that "
             "range."),
        (14, "Lines 41 to 54 open the VectorStore class and its constructor. The docstring (lines 44 "
             "to 50) states the one invariant the class protects: the FAISS index and the metadata "
             "frame are kept in the same row order, so FAISS returning row i means the chunk at "
             "metadata row i. The __init__ (lines 52 to 54) stores the index and calls "
             "reset_index(drop=True) on the metadata so its row labels are a clean 0..n range that "
             "lines up with FAISS row numbers."),
        (13, "Lines 55 to 67 define build(), a classmethod that constructs a store from a chunk "
             "frame — the expensive step. faiss is imported inside the method (line 60) so importing "
             "the module stays light. Line 62 embeds every chunk's text. Line 63 creates a flat "
             "inner-product index of the right dimension. Line 64 adds the vectors. Line 66 keeps "
             "only the four columns needed to return and cite a hit. Line 67 returns a new instance "
             "via cls(index, meta)."),
        (9, "Lines 68 to 76 define save(), which persists the store to data/index/. It imports faiss "
            "(line 72), ensures the output directories exist (line 74), writes the FAISS index to "
            "the configured path (line 75), and writes the metadata frame to parquet without the "
            "index column (line 76)."),
        (9, "Lines 77 to 85 define load(), the classmethod inverse of save — and fast, because no "
            "embedding is needed. It reads the FAISS index from disk (line 83), reads the metadata "
            "parquet (line 84), and returns a reconstructed VectorStore (line 85). This is what the "
            "notebooks and ask.py call to reuse an already-built index."),
        (6, "Lines 86 to 91 define the size property. Marked with @property so it is accessed like "
            "an attribute (store.size). It returns index.ntotal — the number of vectors FAISS holds "
            "— cast to a plain int (line 91)."),
        (None, "Lines 92 to 116 define search(), the heart of the class. It embeds the query (line "
                "100), asks FAISS for the top-k by inner product, receiving parallel arrays of "
                "scores and row indices (line 101), and prepares an empty hits list (line 102). The "
                "loop (lines 103 to 115) pairs each score with its row; it skips rows equal to minus "
                "one (line 104 to 105), which FAISS returns as empty slots when k exceeds the index "
                "size; otherwise it looks up that row in the metadata (line 106) and appends a "
                "Retrieved built from the chunk's fields and the float score (lines 107 to 115). "
                "Line 116 returns the hits, already ordered best-first by FAISS."),
    ],
)
