"""Snippet recall from other threads (spec §5.3).

The current message is turned into a handful of quoted terms, and the store
returns the best BM25 matches from other live threads in the same channel. The
current thread is excluded because it is already in the short-term window.

This module owns the query shape; the store owns the ranking. That keeps FTS
syntax out of ``memory/`` and lets an embedding retriever replace the store
method later without changing the prompt side.
"""

from __future__ import annotations

import re

from slackbot_vaultnel.history.store import MessageMatch, Store
from slackbot_vaultnel.memory.context import Snippet

#: FTS operators and function words match half the corpus and add nothing.
_STOPWORD_TEXT = """
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours yourself
yourselves
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
MAX_QUERY_TERMS = 12
TERM_RE = re.compile(r"[A-Za-z0-9_]+")


def query_terms(text: str) -> list[str]:
    """Lowercase, de-duplicated, stopword-free terms, capped for stability."""

    terms: list[str] = []
    seen: set[str] = set()
    for raw in TERM_RE.findall(text or ""):
        term = raw.lower()
        if len(term) < 2 or term in _STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= MAX_QUERY_TERMS:
            break
    return terms


async def recall_for_prompt(
    store: Store,
    *,
    channel_id: str,
    thread_ts: str,
    text: str,
    limit: int,
) -> list[Snippet]:
    """Best matches from elsewhere in the channel, ready for prompt assembly."""

    terms = query_terms(text)
    if not terms or limit <= 0:
        return []
    matches = await store.search_messages(
        channel_id=channel_id,
        terms=terms,
        exclude_thread_ts=thread_ts,
        limit=limit,
    )
    return [_snippet(match) for match in matches]


def _snippet(match: MessageMatch) -> Snippet:
    author = match.display_name
    if author is None and match.role == "assistant":
        author = "Vaultnel"
    return Snippet(
        content=match.content,
        display_name=author,
        slack_user=match.slack_user,
        created_at=match.created_at,
        channel_id=match.channel_id,
        thread_ts=match.thread_ts,
    )
