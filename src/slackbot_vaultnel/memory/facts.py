"""Long-term facts: parsing, scoping and retrieval (spec §5.3).

The extraction model returns a JSON array of ``{"content", "scope"}`` objects.
This module is the trust boundary: unknown scopes, empty or absurdly long
contents are dropped without failing the rest of the batch. Scope keys are
assigned here, because the model never sees them:

* ``user``      → the Slack user who spoke in the window
* ``channel``   → the channel the thread lives in
* ``workspace`` → one shared bucket (the bot is single-workspace in v1)

Facts are stored with a normalized hash, so repeated extraction over the same
turns is idempotent. The oldest entries are evicted once a scope exceeds its cap.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from slackbot_vaultnel.history.store import FactScope, Store, StoredFact
from slackbot_vaultnel.memory.context import Fact

#: Facts should be one-liners. Long ones are clipped; absurd ones are rejected.
FACT_MAX_CHARS = 240
FACT_MAX_CHARS_HARD = 3 * FACT_MAX_CHARS
#: Single-workspace v1 has no team id in the event payload.
WORKSPACE_KEY = "*"

FACT_SCOPES: tuple[FactScope, ...] = ("user", "channel", "workspace")
_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*|\s*```$")


@dataclass(frozen=True)
class FactCandidate:
    content: str
    scope: FactScope


def parse_fact_entries(raw: str) -> list[FactCandidate] | None:
    """Parse a model response into usable candidates.

    ``None`` means the response was not a JSON array at all — the caller should
    keep the extraction watermark where it is and retry later. A list (possibly
    empty) means the response was understood, even if every entry was dropped.
    """

    payload = _load_array(raw)
    if payload is None:
        return None

    candidates: list[FactCandidate] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        scope = entry.get("scope")
        if not isinstance(content, str) or not isinstance(scope, str):
            continue
        candidate = _candidate(content, scope)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


async def store_candidates(
    store: Store,
    candidates: Iterable[FactCandidate],
    *,
    channel_id: str,
    user_id: str | None,
    source_msg_id: int | None,
    max_per_scope: int,
) -> int:
    """Persist candidates that belong to a real scope key and enforce the cap."""

    touched: set[tuple[FactScope, str]] = set()
    inserted = 0
    for candidate in candidates:
        scope_key = _scope_key(candidate.scope, channel_id, user_id)
        if scope_key is None:
            continue
        added = await store.add_fact(
            StoredFact(
                scope=candidate.scope,
                scope_key=scope_key,
                content=candidate.content,
                source_msg_id=source_msg_id,
            )
        )
        inserted += int(added)
        touched.add((candidate.scope, scope_key))

    for scope, scope_key in touched:
        await store.trim_facts(scope, scope_key, max_per_scope)
    return inserted


async def facts_for_prompt(
    store: Store,
    *,
    channel_id: str,
    user_id: str | None,
) -> list[Fact]:
    """Load every fact in scope, newest first within each scope."""

    facts: list[Fact] = []
    for scope, scope_key in _scopes(channel_id, user_id):
        rows = await store.facts(scope, scope_key)
        facts.extend(
            Fact(
                content=row.content,
                scope=row.scope,
                scope_key=row.scope_key,
                created_at=row.created_at,
            )
            for row in reversed(rows)
        )
    return facts


# -- internals --------------------------------------------------------------


def _load_array(raw: str) -> list[object] | None:
    text = _FENCE_RE.sub("", raw.strip()).strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, list) else None


def _candidate(content: str, scope: str) -> FactCandidate | None:
    normalized_scope = scope.strip().lower()
    if normalized_scope not in FACT_SCOPES:
        return None
    cleaned = " ".join(content.split())
    if not cleaned or len(cleaned) > FACT_MAX_CHARS_HARD:
        return None
    if len(cleaned) > FACT_MAX_CHARS:
        cleaned = cleaned[: FACT_MAX_CHARS - 1].rstrip() + "…"
    return FactCandidate(cleaned, normalized_scope)  # type: ignore[arg-type]


def _scope_key(scope: FactScope, channel_id: str, user_id: str | None) -> str | None:
    if scope == "user":
        return user_id
    if scope == "channel":
        return channel_id
    return WORKSPACE_KEY


def _scopes(channel_id: str, user_id: str | None) -> Sequence[tuple[FactScope, str]]:
    scopes: list[tuple[FactScope, str]] = []
    if user_id:
        scopes.append(("user", user_id))
    scopes.append(("channel", channel_id))
    scopes.append(("workspace", WORKSPACE_KEY))
    return scopes
