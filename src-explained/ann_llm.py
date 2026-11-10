from render_explained import build

build(
    "../src/llm.py",
    "llm.py_explained.pdf",
    "The Claude boundary — one wrapper, with an on-disk response cache",
    "Every call to the Anthropic API goes through this module. It caches each response by a hash of "
    "its inputs (so notebooks and tests reproduce numbers without a key) and degrades gracefully "
    "when no key is present, raising a clear, catchable error instead of a confusing SDK crash.",
    [
        (19, "Lines 1 to 19 are the module docstring. They state the two jobs: cache every response "
             "by an input hash (committed so results reproduce without a key, and never paying twice "
             "for an identical call), and degrade gracefully with no key by raising a clear error. "
             "They also note it uses the official anthropic SDK with adaptive thinking."),
        (9, "Lines 20 to 28 are imports: postponed annotations (line 21), functools for the caching "
            "decorators (line 23), hashlib for the input hash (line 24), json for cache "
            "serialization (line 25), Any for type hints (line 26), and the project config (line "
            "28)."),
        (9, "Lines 29 to 37 define LLMKeyMissing, a custom exception subclassing RuntimeError. The "
            "docstring explains the point of a distinct type: callers such as the eval harness and "
            "ask.py can catch exactly this and fall back to reproduce-from-cache instead of "
            "crashing."),
        (11, "Lines 38 to 48 define _load_cache. The @functools.lru_cache(maxsize=1) decorator (line "
             "42) means the cache file is read from disk only once per process. Lines 45 to 47 open "
             "and JSON-load the cache file if it exists; line 48 returns an empty dict otherwise, so "
             "a first run with no cache still works."),
        (9, "Lines 49 to 57 define _save_cache, which persists the cache dict back to disk. Line 53 "
            "ensures the output directory exists; lines 54 to 55 write the cache as indented JSON "
            "with ensure_ascii=False so non-ASCII text is stored readably."),
        (8, "Lines 58 to 65 define _cache_key, the deterministic hash of a request. Line 64 dumps "
            "the payload to JSON with sort_keys=True so dictionary ordering never affects the hash; "
            "line 65 returns the SHA-256 hex digest of those bytes. Identical logical requests "
            "therefore map to the same cache entry."),
        (11, "Lines 66 to 76 define _get_client, which constructs the Anthropic SDK client. It is "
             "also lru_cache'd (line 71) so the client is built once, and the import of anthropic is "
             "inside the function (line 74) so it only happens when a live call is actually needed — "
             "keeping key-free runs import-light."),
        (14, "Lines 77 to 90 define _blocks_to_dicts, which converts Claude's response content "
             "blocks into plain dictionaries that survive JSON caching. It loops over the blocks "
             "(line 82); text blocks become a type/text dict (lines 83 to 84); tool_use blocks keep "
             "their id, name, and input (lines 85 to 88); thinking blocks carry no displayable "
             "content and are skipped (line 89). Line 90 returns the list."),
        (26, "Lines 91 to 116 begin call(), the one function everything uses. Its parameters (lines "
             "93 to 100) are the messages, an optional system prompt, optional tools, and the model, "
             "max_tokens, and use_cache flag defaulting from config. The docstring (lines 101 to "
             "109) documents the return shape (a stop_reason plus serialized content blocks) and the "
             "cache-hit and key-missing behaviours. Lines 110 to 116 assemble the payload dict "
             "(model, tokens, system, messages, tools) and — with line 117 — hash it into a cache "
             "key."),
        (7, "Lines 117 to 123 handle the cache lookup. Line 117 computes the key. When caching is on "
            "(line 119) it loads the cache (line 120) and, on a hit (line 121), returns the stored "
            "response immediately (line 122) — no network, no key required."),
        (5, "Lines 124 to 128 are the graceful-degradation path: if no API key is available "
            "(line 124), and we reached here, the request was not cached, so it raises LLMKeyMissing "
            "with a message telling the user to set the key or run only cached/local steps."),
        (13, "Lines 129 to 141 build the live request. Line 130 gets the cached client. Lines 131 to "
             "137 assemble the request with model, tokens, messages, and adaptive thinking (letting "
             "Claude decide how much to reason). Lines 138 to 141 add the system prompt and tools "
             "only if they were provided, keeping the request minimal otherwise."),
        (7, "Lines 142 to 148 execute the call. Line 143 sends the request via "
            "client.messages.create. Lines 144 to 147 build the plain-dict result: the response's "
            "stop_reason and its content run through _blocks_to_dicts so it is JSON-serializable."),
        (7, "Lines 149 to 155 write the result to the cache when caching is on: reload the cache "
            "(line 150), store the result under the key (line 151), save it (line 152), and clear "
            "the lru_cache (line 153) so the next _load_cache sees the new entry. Line 155 returns "
            "the result."),
        (None, "Lines 156 to 160 define first_text, a small helper that concatenates the text of "
                "every text block in a response (ignoring tool_use blocks) and strips whitespace — "
                "the convenient way for rag.py and agent.py to get the plain answer string."),
    ],
)
