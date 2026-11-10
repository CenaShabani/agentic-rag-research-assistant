# `src/__init__.py` — the package marker, and the rule it encodes

## What it's for

Makes `src/` an importable package, so `from src import config` works identically from a
notebook, a test, `ask.py`, or the command line. It contains a docstring and nothing else — no
re-exports, no `__all__`, no eager imports.

That emptiness is the design. This file is where a project quietly acquires a second personality,
and this one doesn't.

## The public surface

None. Nothing imports a name *from* `src` itself; every consumer imports the module it needs:

```python
from src import config, evaluation, retrievers     # yes
from src.vectorstore import VectorStore            # yes
from src import search                             # there is no such convenience alias
```

## The walkthrough

There is no logic to walk through, so the interesting content is the two things deliberately
**not** done.

**No re-exports.** A common convenience is to hoist the good bits into the package root —
`from .vectorstore import VectorStore` in `__init__.py`, so callers can write
`from src import VectorStore`. It is rejected here because it creates a second name for the same
thing, and the project's central invariant is that there is exactly *one* definition of
retrieval. Two import paths to `VectorStore` is how a codebase starts drifting toward two
retrievers, and the benchmark would then be measuring a lookalike of the shipped system rather
than the system (see `retrievers.md` and the test that pins the dense arm to `VectorStore.search`).

**No imports at all.** An `__init__.py` that imports its submodules makes `import src` load
everything — including `llm.py` and the Anthropic client, and `embeddings.py` with its 90 MB
sentence-transformer. `tests/test_metrics.py` needs none of that and would pay for all of it,
and `significance.py` is pure arithmetic that must stay importable in a process with no model
and no key. Keeping this file inert is what makes the keyless, offline half of the project
genuinely keyless and offline.

## The non-obvious bits

- **The docstring is the module's actual job.** It states the one-source-of-truth rule that
  every other module depends on, at the first place a reader lands when they open `src/`.
- **There is no `__version__`.** The project is a repo, not a distributed package; a version
  string here would be a number nothing increments and nobody checks.
- **`src/` is not `pip install`-able and is not meant to be.** Imports work because the project
  root is the working directory. The notebooks add it to `sys.path` explicitly in their setup
  cell rather than relying on where Jupyter happened to be launched from.

## How it can break

- **Adding a convenience re-export here breaks nothing immediately**, which is exactly the
  problem: it would pass every test and quietly permit the duplicate-definition failure the rest
  of the project is arranged to prevent.
- **Adding an import here has a measurable cost**, not a stylistic one: it would pull the
  embedding model and the API client into every test process. If a fast keyless suite ever
  becomes slow, this file is the first place to look.
- **Renaming the package** breaks every `from src import ...` in the notebooks, tests and
  entry-point scripts at once. There is no indirection layer here to soften that, on purpose.

## Where it's used

Implicitly, by every import in the project. Nothing imports it directly.
