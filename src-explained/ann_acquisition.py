from render_explained import build

build(
    "../src/acquisition.py",
    "acquisition.py_explained.pdf",
    "Fetching the arXiv corpus and recording provenance",
    "This is the first stage of the pipeline. It queries the public arXiv API for recent cs.CL and "
    "cs.LG papers, keeps title, abstract, and metadata for each, and records provenance (query, "
    "date, count, SHA-256) so anyone can see exactly what was downloaded. The code ships; the data "
    "does not.",
    [
        (13, "Lines 1 to 13 are the module docstring. They describe the stage (query arXiv, keep "
             "title/abstract/metadata, pull full text only where the licence permits), the "
             "provenance discipline (exact query, date, count, and a checksum so silent changes are "
             "detectable), and the rule that nothing here is committed to git — the code ships, the "
             "raw data stays git-ignored."),
        (14, "Lines 14 to 27 are the imports: hashlib for the checksum (line 17), json for JSONL "
             "output (line 18), time for polite sleeps (line 19), urllib.parse and urllib.request to "
             "build and send API requests (lines 20 to 21), ElementTree aliased ET to parse the XML "
             "feed (line 22), asdict and dataclass (line 23), datetime and timezone for the "
             "provenance timestamp (line 24), Iterator for a generator return type (line 25), and "
             "config (line 27)."),
        (9, "Lines 28 to 36 define two module constants. ARXIV_API_URL (line 30) is the Atom API "
            "endpoint. _NS (lines 33 to 36) is the namespace map — the Atom and arXiv XML "
            "namespaces — needed so ElementTree can find the right tags when walking the feed."),
        (11, "Lines 37 to 47 define REDISTRIBUTABLE_LICENSES, a tuple of the Creative Commons and "
             "public-domain licence URLs under which a paper's full text may be stored and chunked. "
             "The comment (lines 38 to 42) frames this as a deliberate licensing choice: full text "
             "for the CC-licensed subset, abstracts for everyone else."),
        (11, "Lines 48 to 58 open the Paper dataclass — one arXiv paper as stored. The docstring "
             "(lines 52 to 58) explains that the text used for retrieval is always title plus "
             "abstract (self-contained and licence-safe for every paper), while the recorded "
             "license field leaves the door open to appending CC-licensed full text later, decided "
             "per paper."),
        (8, "Lines 59 to 66 are the Paper fields: arxiv_id, title, abstract, the authors and "
            "categories lists, the published date, and license (a string or None, since arXiv only "
            "includes a licence when the author selected one)."),
        (10, "Lines 67 to 76 define two computed properties. text (lines 68 to 71) returns the "
             "retrievable body — the title, a blank line, then the abstract. redistributable (lines "
             "73 to 76) returns True when the paper's licence is in the redistributable set, giving "
             "a clean per-paper check used when recording provenance."),
        (21, "Lines 77 to 97 define _build_query_url, which constructs one paged API request. The "
             "docstring (lines 80 to 85) notes results are sorted newest-first so the corpus is a "
             "coherent recent slice. Line 86 builds the category clause (cs.CL OR cs.LG). Lines 87 "
             "to 93 assemble the query parameters (search string, paging offset, page size, sort "
             "order). Lines 94 to 96 hand-encode the query string so the OR clause's plus signs are "
             "not escaped, and line 97 returns the full URL."),
        (37, "Lines 98 to 134 define _parse_feed, a generator that yields Paper objects from one "
             "Atom response. Line 102 parses the XML; line 103 loops over each entry. Lines 105 to "
             "106 extract the arXiv id from the tail of the entry's id URL, stripping any version "
             "suffix. Lines 108 to 110 pull the title, abstract, and published date, defaulting to "
             "empty strings. Lines 112 to 119 collect the author names and category terms into "
             "lists. Lines 121 to 123 read the optional licence element. Line 125 keeps only entries "
             "that have an id, title, and abstract, and lines 126 to 134 yield a Paper, collapsing "
             "arXiv's internal newlines in the title and abstract with a split-and-join."),
        (26, "Lines 135 to 160 define fetch_papers, which downloads up to max_papers, paging "
             "politely. The docstring (lines 138 to 143) explains the sleeps respect arXiv's rate "
             "guidance and that results are de-duplicated by id. Lines 144 to 145 initialise a "
             "dict keyed by id and a paging offset. The loop (lines 146 to 158) builds each page "
             "URL, fetches and parses it, stops early if a page is empty (lines 152 to 153), stores "
             "each paper by id (lines 154 to 155), advances the offset, and sleeps between requests. "
             "Line 160 returns the papers, capped at max_papers."),
        (15, "Lines 161 to 175 define fetch_by_ids, which fetches specific papers by id using the "
             "API's id_list parameter — used to pull the seed set of seminal papers so the "
             "evaluation's ground truth is always present. Lines 169 to 170 short-circuit on an "
             "empty input. Line 171 builds the id_list query; lines 172 to 173 fetch it; line 174 "
             "sleeps; line 175 returns the parsed papers. One request covers all ids."),
        (10, "Lines 176 to 185 define _sha256, the provenance fingerprint. Line 180 creates a "
             "SHA-256 hasher; lines 181 to 183 read the file in one-megabyte blocks (the "
             "iter-with-sentinel idiom) so a large file is not loaded fully into memory; line 184 "
             "returns the hex digest."),
        (19, "Lines 186 to 204 define save_papers, which writes the corpus and its provenance. Lines "
             "190 to 192 write one JSON object per paper to the JSONL file (asdict turns each Paper "
             "into a dict). Lines 194 to 202 build the provenance record — source, categories, a "
             "UTC timestamp, the paper count, how many are redistributable, the filename, and the "
             "file's SHA-256 — and lines 203 to 204 write it as indented JSON alongside the data."),
        (9, "Lines 205 to 213 define load_papers, the inverse of save_papers: it reads the JSONL "
            "file line by line (lines 210 to 212), reconstructing a Paper from each JSON object, and "
            "returns the list."),
        (None, "Lines 214 to 230 define acquire, the full first stage called by run_pipeline.py. The "
                "docstring (lines 217 to 222) explains the merge. Line 223 fetches the recent slice; "
                "line 224 fetches the seed papers. Line 225 builds a dict of the recent papers keyed "
                "by id, and lines 226 to 227 overlay the seed papers so they win on any id "
                "collision and are always present. Line 228 flattens back to a list, line 229 saves "
                "it with provenance, and line 230 returns it."),
    ],
)
