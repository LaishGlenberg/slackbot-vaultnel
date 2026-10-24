"""Snippet recall: query shaping and prompt mapping (spec §5.3).

The store owns ranking; this layer decides which words are worth searching and
turns hits into labelled snippets. Message text must never reach FTS as syntax.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fakes import InMemoryStore, ManualClock

from slackbot_vaultnel.history.store import Turn
from slackbot_vaultnel.memory.recall import (
    MAX_QUERY_TERMS,
    query_terms,
    recall_for_prompt,
)

CHANNEL = "C0123456"
OTHER_CHANNEL = "C9999999"
THREAD_TS = "1700000000.000100"
OTHER_TS = "1700000000.000200"
OTHER_THREAD = "1700000000.000300"


def test_query_terms_lowercase_and_drop_stopwords():
    assert query_terms("How do we DEPLOY the service?") == ["deploy", "service"]


def test_query_terms_deduplicate_and_cap():
    terms = query_terms(" ".join(f"term{i}" for i in range(MAX_QUERY_TERMS + 5)))
    assert terms == [f"term{i}" for i in range(MAX_QUERY_TERMS)]

    assert query_terms("sqlite sqlite SQLite wal") == ["sqlite", "wal"]


def test_query_terms_ignore_punctuation_operators_and_noise():
    assert query_terms('"wal" AND NOT sqlite* -mode') == ["wal", "sqlite", "mode"]
    assert query_terms("what is the?") == []
    assert query_terms("") == []


async def seed(store, thread_ts=OTHER_TS, content="we settled on wal mode", **kwargs):
    thread = await store.get_or_create_thread(CHANNEL, thread_ts)
    return await store.append_message(thread.id, Turn("user", content, 6, **kwargs))


async def test_hits_become_snippets_with_author_date_and_thread():
    clock = ManualClock()
    store = InMemoryStore(clock=clock.now)
    await seed(store, display_name="Bob", slack_user="U1")

    snippets = await recall_for_prompt(
        store,
        channel_id=CHANNEL,
        thread_ts=THREAD_TS,
        text="what did we decide about wal mode?",
        limit=3,
    )

    assert len(snippets) == 1
    snippet = snippets[0]
    assert snippet.content == "we settled on wal mode"
    assert snippet.display_name == "Bob"
    assert snippet.slack_user == "U1"
    assert snippet.channel_id == CHANNEL
    assert snippet.thread_ts == OTHER_TS
    assert snippet.created_at == datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


async def test_assistant_hits_are_labelled_as_the_bot():
    store = InMemoryStore(clock=ManualClock().now)
    thread = await store.get_or_create_thread(CHANNEL, OTHER_TS)
    await store.append_message(thread.id, Turn("assistant", "use wal mode for sqlite", 6))

    snippets = await recall_for_prompt(
        store,
        channel_id=CHANNEL,
        thread_ts=THREAD_TS,
        text="wal mode sqlite",
        limit=3,
    )

    assert snippets[0].display_name == "Vaultnel"


async def test_current_thread_is_not_recalled():
    store = InMemoryStore(clock=ManualClock().now)
    await seed(store, thread_ts=THREAD_TS, content="wal mode inside this thread")
    await seed(store, thread_ts=OTHER_TS, content="wal mode from elsewhere")

    snippets = await recall_for_prompt(
        store,
        channel_id=CHANNEL,
        thread_ts=THREAD_TS,
        text="wal mode",
        limit=3,
    )

    assert [snippet.content for snippet in snippets] == ["wal mode from elsewhere"]


async def test_other_channels_are_not_recalled():
    store = InMemoryStore(clock=ManualClock().now)
    thread = await store.get_or_create_thread(OTHER_CHANNEL, OTHER_TS)
    await store.append_message(thread.id, Turn("user", "wal mode in another channel", 6))

    snippets = await recall_for_prompt(
        store,
        channel_id=CHANNEL,
        thread_ts=THREAD_TS,
        text="wal mode",
        limit=3,
    )

    assert snippets == []


async def test_limit_is_passed_through_and_zero_means_nothing():
    store = InMemoryStore(clock=ManualClock().now)
    for index in range(4):
        await seed(store, thread_ts=f"1700000000.00040{index}", content=f"wal mode note {index}")

    snippets = await recall_for_prompt(
        store, channel_id=CHANNEL, thread_ts=THREAD_TS, text="wal mode", limit=2
    )
    assert len(snippets) == 2

    assert (
        await recall_for_prompt(
            store, channel_id=CHANNEL, thread_ts=THREAD_TS, text="wal mode", limit=0
        )
        == []
    )


async def test_meaningless_questions_skip_the_store_entirely():
    class ExplodingStore(InMemoryStore):
        async def search_messages(self, **kwargs):
            raise AssertionError("should not search for stopwords only")

    store = ExplodingStore(clock=ManualClock().now)
    assert (
        await recall_for_prompt(
            store, channel_id=CHANNEL, thread_ts=THREAD_TS, text="what is it?", limit=3
        )
        == []
    )
