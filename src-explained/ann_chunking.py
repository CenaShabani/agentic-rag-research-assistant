from render_explained import build

build(
    "../src/chunking.py",
    "chunking.py_explained.pdf",
    "Splitting documents into overlapping, citable chunks",
    "Retrieval runs on <b>chunks</b>, not whole documents, because one embedding vector can only "
    "mean one thing. This module slices text into overlapping character windows and attaches each "
    "chunk's source metadata so answers can cite it.",
    [
        (14, "Lines 1 to 14 are the module docstring. They justify chunking (a whole-paper embedding "
             "averages away every idea and retrieval on it is mush; a few sentences embed to "
             "something specific), note the two knobs live in config.py, and explain the deliberate "
             "choice of simple character-based splitting with no tokenizer dependency, plus the rule "
             "that every chunk carries its parent paper's metadata."),
        (9, "Lines 15 to 23 are imports: postponed annotations (line 16), dataclass for the Chunk "
            "type (line 18), pandas for the output frame (line 20), the project config (line 22), "
            "and the Paper type from acquisition (line 23) that this module consumes."),
        (11, "Lines 24 to 34 define the Chunk dataclass — one retrievable unit. Fields: chunk_id "
             "(line 30, formatted as paper-id colon position, e.g. 2401.01234:2), the arxiv_id and "
             "title for citation (lines 31 to 32), the chunk text (line 33), and position (line 34, "
             "the 0-based index of the chunk within its paper). The inline comments document the "
             "id format and the position meaning."),
        (19, "Lines 35 to 53 begin split_text. The signature (lines 37 to 39) defaults size and "
             "overlap to the config constants. The docstring (lines 40 to 51) explains the "
             "overlapping-window scheme and the two guarded edge cases. Lines 52 to 53 enforce the "
             "first guard: if overlap is greater than or equal to size the window would never "
             "advance, so it raises ValueError rather than looping forever."),
        (5, "Lines 54 to 58 handle the short-text case. Line 55 strips surrounding whitespace. Lines "
            "56 to 57 return the whole text as a single-element list when it fits in one window (or "
            "an empty list if the text was blank). Line 58 is a blank separator before the main "
            "loop."),
        (17, "Lines 59 to 75 are the windowing loop. Line 59 initialises the chunks list; line 60 "
             "sets start to 0; line 61 loops while there is text left. Line 62 sets the window end. "
             "Lines 63 to 69 try to move that end back to the last space in the window (so words are "
             "not cut in half), but only if the space is past the window's midpoint, guarding "
             "against long unbroken strings like URLs. Line 70 appends the stripped window. Lines 71 "
             "to 72 break when the end reaches the text end. Line 74 advances start to end minus "
             "overlap, creating the intended overlap. Line 75 returns all non-empty chunks."),
        (14, "Lines 76 to 89 define chunk_paper, which turns one Paper into a list of Chunks. It is "
             "a list comprehension (lines 80 to 88) that calls split_text on the paper's combined "
             "text and, for each resulting piece with its index i, builds a Chunk stamped with the "
             "paper's id (as id colon i), arxiv_id, title, the text piece, and position i — carrying "
             "the metadata through so citations survive."),
        (8, "Lines 90 to 97 define chunk_papers, which chunks the whole corpus. It initialises an "
            "empty list (line 94), loops over every paper (line 95), extends the list with that "
            "paper's chunks (line 96), and returns the flattened result (line 97)."),
        (13, "Lines 98 to 110 define chunks_to_frame, converting the list of Chunk objects into a "
             "pandas DataFrame — the on-disk form and the shape the vector store expects. It builds "
             "the frame column by column (lines 104 to 108), each column a list comprehension "
             "pulling one field from every chunk."),
        (8, "Lines 111 to 118 define save_chunks, called by the pipeline. It ensures the output "
            "directories exist (line 115), builds the frame (line 116), writes it to parquet without "
            "the row index (line 117), and returns the frame (line 118) so the caller can use it "
            "immediately without re-reading disk."),
        (None, "Lines 119 to 123 define load_chunks, the inverse: it reads and returns the chunk "
                "frame previously written by save_chunks from the configured parquet path."),
    ],
)
