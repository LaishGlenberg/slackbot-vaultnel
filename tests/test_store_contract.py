import pytest
from fakes import InMemoryStore

from slackbot_vaultnel.history.store import SqliteStore, Turn


@pytest.fixture(params=["memory", "sqlite"])
async def store(request, tmp_path):
    match request.param:
        case "memory":
            instance = InMemoryStore()
        case "sqlite":
            instance = SqliteStore(tmp_path / "contract.db")
        case _:
            raise AssertionError(f"unknown store {request.param}")
    await instance.open()
    try:
        yield instance
    finally:
        await instance.close()


CHANNEL = "C0123456"
THREAD_TS = "1700000000.000100"


def turn(content: str = "hello", role: str = "user", tokens: int = 3):
    return Turn(role=role, content=content, tokens=tokens)


async def test_created_thread_has_sane_defaults(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)

    assert thread.id > 0
    assert thread.channel_id == CHANNEL
    assert thread.thread_ts == THREAD_TS
    assert thread.summary == ""
    assert thread.summarized_through_msg_id == 0
    assert thread.last_extracted_msg_id == 0
    assert thread.archived is False
    assert thread.created_at <= thread.updated_at


async def test_same_key_returns_the_same_thread(store):
    first = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    second = await store.get_or_create_thread(CHANNEL, THREAD_TS)

    assert first.id == second.id


async def test_different_keys_are_different_threads(store):
    a = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    b = await store.get_or_create_thread(CHANNEL, "1700000000.000200")
    c = await store.get_or_create_thread("C9999999", THREAD_TS)

    assert len({a.id, b.id, c.id}) == 3


async def test_appending_returns_a_persisted_turn(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)

    saved = await store.append_message(thread.id, turn("what is sqlite?"))

    assert saved.id > 0
    assert saved.content == "what is sqlite?"
    assert saved.role == "user"
    assert saved.tokens == 3
    assert saved.created_at is not None


async def test_appended_turns_get_increasing_ids(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)

    ids = [(await store.append_message(thread.id, turn(f"m{i}"))).id for i in range(4)]

    assert ids == sorted(ids)
    assert len(set(ids)) == 4


async def test_appending_to_an_unknown_thread_fails(store):
    with pytest.raises(ValueError, match="thread"):
        await store.append_message(999_999, turn())


async def test_appending_bumps_updated_at(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)

    await store.append_message(thread.id, turn())

    refreshed = await store.get_thread(CHANNEL, THREAD_TS)
    assert refreshed is not None
    assert refreshed.updated_at >= thread.updated_at


async def test_messages_are_chronological(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    for i in range(5):
        await store.append_message(thread.id, turn(f"m{i}"))

    messages = await store.messages(thread.id)

    assert [m.content for m in messages] == [f"m{i}" for i in range(5)]


async def test_after_id_returns_only_newer_turns(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    saved = [await store.append_message(thread.id, turn(f"m{i}")) for i in range(5)]

    messages = await store.messages(thread.id, after_id=saved[2].id)

    assert [m.content for m in messages] == ["m3", "m4"]


async def test_limit_returns_the_newest_turns_in_order(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    for i in range(5):
        await store.append_message(thread.id, turn(f"m{i}"))

    messages = await store.messages(thread.id, limit=2)

    assert [m.content for m in messages] == ["m3", "m4"]


async def test_message_count(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)

    assert await store.message_count(thread.id) == 0
    await store.append_message(thread.id, turn())
    assert await store.message_count(thread.id) == 1


async def test_set_summary_stores_text_and_high_water_mark(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    saved = [await store.append_message(thread.id, turn(f"m{i}")) for i in range(3)]

    updated = await store.set_summary(thread.id, "they discussed tokens", saved[1].id)

    assert updated.summary == "they discussed tokens"
    assert updated.summarized_through_msg_id == saved[1].id
    assert (await store.get_thread(CHANNEL, THREAD_TS)).summary == "they discussed tokens"


async def test_high_water_mark_cannot_move_backwards(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    saved = [await store.append_message(thread.id, turn(f"m{i}")) for i in range(3)]
    await store.set_summary(thread.id, "two folded", saved[1].id)

    with pytest.raises(ValueError, match="summarized_through_msg_id"):
        await store.set_summary(thread.id, "one folded", saved[0].id)


async def test_summary_must_reference_a_message_in_this_thread(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    other = await store.get_or_create_thread(CHANNEL, "1700000000.999999")
    foreign = await store.append_message(other.id, turn("not yours"))
    await store.append_message(thread.id, turn("mine"))

    with pytest.raises(ValueError, match="summarized_through_msg_id"):
        await store.set_summary(thread.id, "nope", foreign.id)

    with pytest.raises(ValueError, match="summarized_through_msg_id"):
        await store.set_summary(thread.id, "nope", 424242)


async def test_mark_extracted_is_monotonic(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    saved = [await store.append_message(thread.id, turn(f"m{i}")) for i in range(3)]

    await store.mark_extracted(thread.id, saved[1].id)

    with pytest.raises(ValueError, match="last_extracted_msg_id"):
        await store.mark_extracted(thread.id, saved[0].id)


async def test_archive_flags_thread_and_messages(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    await store.append_message(thread.id, turn())

    archived = await store.archive_thread(CHANNEL, THREAD_TS)

    assert archived.archived is True
    assert await store.messages(thread.id, include_archived=False) == []
    assert len(await store.messages(thread.id)) == 1


async def test_archiving_is_idempotent(store):
    await store.get_or_create_thread(CHANNEL, THREAD_TS)

    first = await store.archive_thread(CHANNEL, THREAD_TS)
    second = await store.archive_thread(CHANNEL, THREAD_TS)

    assert first.archived is second.archived is True


async def test_reopening_an_archived_thread_starts_a_fresh_context(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    saved = [await store.append_message(thread.id, turn(f"m{i}")) for i in range(3)]
    await store.set_summary(thread.id, "old summary", saved[2].id)
    await store.archive_thread(CHANNEL, THREAD_TS)

    reopened = await store.get_or_create_thread(CHANNEL, THREAD_TS)

    assert reopened.archived is False
    assert reopened.summary == ""
    # everything before the reset is outside the live window and never
    # re-summarized, while the rows themselves survive for audit
    assert reopened.summarized_through_msg_id == saved[2].id
    assert await store.messages(thread.id, after_id=reopened.summarized_through_msg_id) == []


async def test_thread_lookup_of_unknown_key_returns_none(store):
    assert await store.get_thread(CHANNEL, "does-not-exist") is None


async def test_archiving_an_unknown_thread_fails(store):
    with pytest.raises(ValueError, match="thread"):
        await store.archive_thread(CHANNEL, "does-not-exist")
