"""Claude wrapper with an on-disk response cache — the project's LLM boundary.

Everything that calls the Anthropic API goes through here. Two jobs:

1. **Cache every response by a hash of its inputs.** The cache is committed to
   `outputs/results/llm_cache.json`, so the notebook, the tests, and
   `test_pipeline_integrity.py` reproduce every published number *without an API
   key* — exactly the "reference-only, load the saved result" convention the
   portfolio's Kaggle-GPU notebooks use. Re-running the pipeline also never pays
   for an identical call twice.

2. **Degrade gracefully with no key.** If `ANTHROPIC_API_KEY` is unset and a call
   isn't already cached, we raise a clear, catchable error rather than a confusing
   SDK exception — the caller can then decide to skip that step.

The model id, token cap, and cache path are all named constants in config.py.
Uses the official `anthropic` SDK (not raw HTTP) with adaptive thinking, per the
current Claude API guidance.
"""

from __future__ import annotations

import functools
import hashlib
import json
from typing import Any

from . import config


class LLMKeyMissing(RuntimeError):
    """Raised when a live call is needed but no API key is available.

    Distinct exception type so callers (the eval harness, ask.py) can catch it
    and fall back to "reproduce from cache only" instead of crashing.
    """


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _load_cache() -> dict[str, Any]:
    """Load the on-disk response cache (once per process)."""
    if config.LLM_CACHE_FILE.exists():
        with open(config.LLM_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    """Persist the response cache."""
    config.ensure_dirs()
    with open(config.LLM_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _cache_key(payload: dict[str, Any]) -> str:
    """Deterministic hash of the request payload.

    Sorting keys makes the hash independent of dict ordering, so the same logical
    request always maps to the same cache entry.
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _get_client():
    """Construct the Anthropic client (lazy — only when a live call happens)."""
    import anthropic

    return anthropic.Anthropic()


def _blocks_to_dicts(content) -> list[dict[str, Any]]:
    """Serialize Claude response content blocks to plain dicts for caching."""
    out: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            out.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            out.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
            )
        # thinking blocks carry no content under the default display; skip them.
    return out


def call(
    messages: list[dict[str, Any]],
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    model: str = config.GEN_MODEL,
    max_tokens: int = config.MAX_ANSWER_TOKENS,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Call Claude once and return a plain-dict response.

    The return shape is `{"stop_reason": str, "content": [blocks...]}` where each
    block is a serialized dict — everything the agent loop and the answer parser
    need, and everything that round-trips cleanly through the JSON cache.

    On a cache hit we never touch the network (so no key is needed). On a miss
    with no key we raise `LLMKeyMissing` so the caller can fall back.
    """
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "tools": tools,
    }
    key = _cache_key(payload)

    if use_cache:
        cache = _load_cache()
        if key in cache:
            return cache[key]

    if not config.api_key_available():
        raise LLMKeyMissing(
            "No ANTHROPIC_API_KEY set and this request is not cached. "
            "Set the key to generate it, or run only the cached/local steps."
        )

    client = _get_client()
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        # Adaptive thinking: Claude decides how much to reason per request.
        "thinking": {"type": "adaptive"},
    }
    if system is not None:
        request["system"] = system
    if tools is not None:
        request["tools"] = tools

    response = client.messages.create(**request)
    result = {
        "stop_reason": response.stop_reason,
        "content": _blocks_to_dicts(response.content),
    }

    if use_cache:
        cache = _load_cache()
        cache[key] = result
        _save_cache(cache)
        _load_cache.cache_clear()  # so the next read sees the write

    return result


def first_text(result: dict[str, Any]) -> str:
    """Extract the concatenated text of a response (ignoring tool_use blocks)."""
    return "".join(b["text"] for b in result["content"] if b["type"] == "text").strip()
