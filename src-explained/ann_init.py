from render_explained import build

build(
    "../src/__init__.py",
    "__init__.py_explained.pdf",
    "The package marker that makes `src` importable",
    "This file is what turns the <b>src</b> folder into an importable Python package. It "
    "contains only a docstring, but its mere presence lets every other part of the project write "
    "<b>from src import ...</b>. There is no executable code here on purpose.",
    [
        (6, "Lines 1 to 6 are the package docstring. A string literal placed as the very first "
            "statement in a package's __init__.py becomes the package's documentation, readable via "
            "help(src). The text states the project's guiding rule: this package is the single "
            "source of truth, and the entry-point scripts, notebooks, and tests all import from it "
            "rather than re-implementing retrieval, chunking, or generation. Because the file has no "
            "imports or statements, importing src is instant and free of side effects."),
    ],
)
