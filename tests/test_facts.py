"""Fact parsing, scope keys, dedupe and caps (spec §5.3).

The model only supplies `content` and `scope`; this layer decides which entries
are usable, which scope key they belong to, and when the oldest fact has to go.
Malformed output is a failed extraction, not a crash.
"""

from __future__ import annotations

import json

from fakes import InMemoryStore

from slackbot_vaultnel.history.store import StoredFact
from slackbot_vaultnel.memory.facts import (
    FACT_MAX_CHARS,
    FACT_MAX_CHARS_HARD,
    WORKSPACE_KEY,
    FactCandidate,
    facts_for_prompt,
    parse_fact_entries,
    store_candidates,
)

CHANNEL = "C0123456"
USER = "U0123456"


def payload(*entries: object) -> str:
    return json.dumps(list(entries))


# -- parsing ----------------------------------------------------------------


def test_valid_entries_are_kept_as_candidates():
    raw = payload(
        {"content": "Ships on Fly.io", "scope": "channel"},
        {"content": "Prefers TypeScript", "scope": "user"},
        {"content": "Four-day week", "scope": "workspace"},
    )

    assert parse_fact_entries(raw) == [
        FactCandidate("Ships on Fly.io", "channel"),
        FactCandidate("Prefers TypeScript", "user"),
        FactCandidate("Four-day week", "workspace"),
    ]


def test_unknown_scopes_and_malformed_entries_are_dropped():
    raw = payload(
        {"content": "keep me", "scope": "channel"},
        {"content": "bad scope", "scope": "planet"},
        {"content": "bad type", "scope": 7},
        {"scope": "user"},
        "not an object",
        {"content": "   ", "scope": "user"},
    )

    assert parse_fact_entries(raw) == [FactCandidate("keep me", "channel")]


def test_invalid_json_is_reported_as_a_failure():
    assert parse_fact_entries("{not json") is None
    assert parse_fact_entries('{"content": "x", "scope": "user"}') is None
    assert parse_fact_entries("") is None


def test_a_fenced_json_array_is_accepted():
    raw = "```json\n" + payload({"content": "one", "scope": "channel"}) + "\n```"

    assert parse_fact_entries(raw) == [FactCandidate("one", "channel")]


def test_long_facts_are_clipped_and_absurd_ones_are_dropped():
    clipped = parse_fact_entries(
        payload({"content": "x" * (FACT_MAX_CHARS + 20), "scope": "channel"})
    )
    dropped = parse_fact_entries(
        payload({"content": "x" * (FACT_MAX_CHARS_HARD + 1), "scope": "channel"})
    )

    assert clipped is not None and len(clipped[0].content) <= FACT_MAX_CHARS
    assert clipped[0].content.endswith("…")
    assert dropped == []


# -- storage ----------------------------------------------------------------


async def test_candidates_are_scoped_and_deduped():
    store = InMemoryStore()
    candidates = [
        FactCandidate("Prefers TypeScript", "user"),
        FactCandidate("prefers   typescript", "user"),
        FactCandidate("Ships on Fly.io", "channel"),
        FactCandidate("Four-day week", "workspace"),
    ]

    stored = await store_candidates(
        store,
        candidates,
        channel_id=CHANNEL,
        user_id=USER,
        source_msg_id=42,
        max_per_scope=10,
    )

    assert stored == 3
    user_facts = await store.facts("user", USER)
    assert [f.content for f in user_facts] == ["Prefers TypeScript"]
    assert user_facts[0].source_msg_id == 42
    assert [f.content for f in await store.facts("channel", CHANNEL)] == ["Ships on Fly.io"]
    assert [f.content for f in await store.facts("workspace", WORKSPACE_KEY)] == ["Four-day week"]


async def test_user_facts_are_dropped_without_a_speaker():
    store = InMemoryStore()

    stored = await store_candidates(
        store,
        [FactCandidate("Likes tea", "user")],
        channel_id=CHANNEL,
        user_id=None,
        source_msg_id=1,
        max_per_scope=10,
    )

    assert stored == 0
    assert await store.facts("user", USER) == []


async def test_the_cap_evicts_the_oldest_fact():
    store = InMemoryStore()
    candidates = [FactCandidate(f"fact {i}", "channel") for i in range(4)]

    await store_candidates(
        store,
        candidates,
        channel_id=CHANNEL,
        user_id=USER,
        source_msg_id=1,
        max_per_scope=2,
    )

    assert [f.content for f in await store.facts("channel", CHANNEL)] == ["fact 2", "fact 3"]


# -- prompt retrieval -------------------------------------------------------


async def test_facts_for_prompt_returns_newest_first_per_scope():
    store = InMemoryStore()
    for content in ("old channel", "new channel"):
        await store.add_fact(StoredFact("channel", CHANNEL, content))
    await store.add_fact(StoredFact("user", USER, "about me"))
    await store.add_fact(StoredFact("workspace", WORKSPACE_KEY, "org rule"))

    facts = await facts_for_prompt(store, channel_id=CHANNEL, user_id=USER)

    assert [f.content for f in facts] == [
        "about me",
        "new channel",
        "old channel",
        "org rule",
    ]
    assert [f.scope for f in facts] == ["user", "channel", "channel", "workspace"]
