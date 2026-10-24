"""Token accounting.

The bot never tokenizes history at prompt-assembly time: counts are computed
once when a turn is persisted and stored on the row. This module is the single
place that knows how counting works.

``cl100k_base`` is used as a stand-in for every provider's tokenizer. It is an
approximation (DeepSeek and most OpenRouter models use different vocabularies),
which is fine because budgets carry generous headroom.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache

import tiktoken

ENCODING_NAME = "cl100k_base"
ELLIPSIS = "…"
#: Chat formats wrap each message in role delimiters; ~4 tokens is the
#: conventional OpenAI estimate and keeps us on the safe side.
MESSAGE_OVERHEAD_TOKENS = 4


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text, disallowed_special=()))


def tokens_for_messages(messages: Iterable[Mapping[str, object]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        total += MESSAGE_OVERHEAD_TOKENS
        if isinstance(content, str):
            total += count_tokens(content)
    return total


def truncate_tokens(text: str, max_tokens: int, ellipsis: str = ELLIPSIS) -> str:
    """Shrink ``text`` until it fits, marking the cut with an ellipsis.

    Multi-byte characters can round-trip to more tokens after decoding, so the
    result is re-counted and trimmed further until the budget genuinely holds.
    """

    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    if max_tokens == 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    encoding = _encoding()
    budget = max_tokens - count_tokens(ellipsis)
    if budget <= 0:
        return ""

    token_ids = encoding.encode(text, disallowed_special=())[:budget]
    while token_ids:
        candidate = encoding.decode(token_ids).rstrip() + ellipsis
        if count_tokens(candidate) <= max_tokens:
            return candidate
        token_ids.pop()
    return ""
