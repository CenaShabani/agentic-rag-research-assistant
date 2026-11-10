# `src/llm.py` — the API boundary, and the reason the project reproduces without a key

## What it's for

Every call to the Anthropic API in this project goes through this one module. It exists to do two
things that matter more than wrapping an SDK:

1. **Cache every response**, keyed by a hash of its inputs, into a committed file.
2. **Fail clearly and catchably with no key**, so callers can degrade instead of crashing.

Together those are what let the README claim that the published numbers reproduce from a clean
clone with no API key. That claim is a property of this file.

## The public surface

| Member | In | Out |
|---|---|---|
| `call(messages, system, tools=None)` | message list, system prompt, optional tools | `dict` — the response, from cache or live |
| `first_text(result)` | a response dict | `str` — the first text block |
| `LLMKeyMissing` | — | exception type callers catch to fall back to cache-only |

`LLMKeyMissing` being its own class is the point of it: `rag.py`, `agent.py`, the eval harness and
`ask.py` can all distinguish "no key, skip this step gracefully" from a genuine SDK failure. A
generic `RuntimeError` would force callers to string-match an error message.

## The walkthrough

### The cache is a published artifact, not an optimisation

`outputs/results/llm_cache.json` is **committed**. That is unusual and deliberate: it is the same
convention as the portfolio's Kaggle-GPU notebooks, where expensive results are saved and loaded
rather than re-run. Here it means the naive-vs-agent comparison — the half that costs money —
reproduces exactly for a reader with no key.

`_load_cache` is `lru_cache(maxsize=1)`, so the file is read once per process rather than on every
call. `_save_cache` writes the whole dict back.

### The cache key is a hash of the full payload

`_cache_key` hashes the entire request — messages, system prompt, tools, model, token cap. That
matters more than it looks:

- Hash **too little** (say, just the question) and a changed system prompt silently returns the
  old answer. Every published number would then describe a prompt that no longer exists.
- Hash **too much** (including a timestamp or a request id) and nothing ever hits the cache.

Hashing exactly the inputs that determine the output is what makes the cache safe to commit. A
consequence worth knowing: **editing a system prompt invalidates every cached response that used
it**, so the affected numbers need regenerating with a key. That is correct behaviour, not a bug.

### The no-key path

```python
raise LLMKeyMissing(...)
```

Only reached when the call is **not** already cached. Cached calls never touch the key or the
network — which is why the tests pass with no key and why 24 of 26 tests run in CI.

### `_blocks_to_dicts`

Converts SDK response objects into plain dicts before they are cached or returned. Two reasons:
SDK objects are not JSON-serialisable, so the cache needs plain data; and it keeps the rest of the
codebase free of SDK types, so `agent.py` can inspect `result["content"]` blocks with ordinary
dict access. It is also what lets the tests mock `llm.call` with plain dictionaries.

`first_text` then pulls the first text block out — necessary because a tool-use response contains
a mix of `text` and `tool_use` blocks and the text is not always first.

## The non-obvious bits

- **Model id, token cap and cache path are all in `config.py`.** Nothing about which model is used
  is hard-coded here.
- **Uses the official `anthropic` SDK with adaptive thinking**, not raw HTTP, per current Claude
  API guidance — the SDK handles retries, streaming semantics and message-format changes.
- **`_get_client` is lazily constructed** so importing the module doesn't require the SDK or a key.
  Same pattern as `embeddings.get_model`.
- **The cache is keyed, not append-only.** Re-running the pipeline never pays twice for an
  identical call.

## How it can break

- **A cache file committed from a different model version** would reproduce numbers the current
  model would not produce. The model id is inside the hashed payload, so a model change misses the
  cache rather than silently reusing it — the failure mode is "needs a key", not "wrong answer".
- **Concurrent writes** would race: `_save_cache` rewrites the whole file. The pipeline is
  single-process, so this doesn't bite, but parallelising the eval harness would need a lock.
- **A corrupt or truncated cache JSON** raises on load and takes everything down with it. Blunt,
  but preferable to silently starting from an empty cache and billing for a full re-run.
- **Cache growth.** Every distinct prompt is stored forever; the file only shrinks if pruned by
  hand.
