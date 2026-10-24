import pytest
from fakes import InMemoryStore

from slackbot_vaultnel.history.store import SqliteStore, StoredFact, Turn


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


# -- facts ------------------------------------------------------------------


def fact(content="ships on Fly.io", *, scope="channel", scope_key=CHANNEL):
    return StoredFact(scope=scope, scope_key=scope_key, content=content)


async def test_facts_are_stored_with_a_hash_and_timestamp(store):
    inserted = await store.add_fact(fact("Ships on Fly.io with sqlite"))

    assert inserted is True
    [saved] = await store.facts("channel", CHANNEL)
    assert saved.id > 0
    assert saved.content == "Ships on Fly.io with sqlite"
    assert saved.content_hash
    assert saved.created_at is not None


async def test_duplicate_facts_are_ignored_after_normalizing_case_and_space(store):
    assert await store.add_fact(fact("Ships on   Fly.io")) is True
    assert await store.add_fact(fact("ships on fly.io")) is False

    assert len(await store.facts("channel", CHANNEL)) == 1


async def test_the_same_fact_can_exist_in_different_scopes_or_keys(store):
    assert await store.add_fact(fact("TypeScript", scope="user", scope_key="U1")) is True
    assert await store.add_fact(fact("TypeScript", scope="user", scope_key="U2")) is True
    assert await store.add_fact(fact("TypeScript", scope="channel", scope_key=CHANNEL)) is True

    assert len(await store.facts("user", "U1")) == 1
    assert len(await store.facts("user", "U2")) == 1
    assert len(await store.facts("channel", CHANNEL)) == 1


async def test_facts_are_returned_oldest_first(store):
    for content in ("one", "two", "three"):
        await store.add_fact(fact(content))

    assert [f.content for f in await store.facts("channel", CHANNEL)] == ["one", "two", "three"]


async def test_trim_facts_evicts_the_oldest_entries(store):
    for content in ("one", "two", "three", "four"):
        await store.add_fact(fact(content))

    removed = await store.trim_facts("channel", CHANNEL, 2)

    assert removed == 2
    assert [f.content for f in await store.facts("channel", CHANNEL)] == ["three", "four"]


async def test_trim_facts_only_touches_one_scope_key(store):
    await store.add_fact(fact("one"))
    await store.add_fact(fact("two"))
    await store.add_fact(fact("keep", scope="user", scope_key="U1"))

    assert await store.trim_facts("channel", CHANNEL, 1) == 1
    assert [f.content for f in await store.facts("user", "U1")] == ["keep"]


async def test_delete_facts_removes_only_the_requested_scope_key(store):
    await store.add_fact(fact("mine", scope="user", scope_key="U1"))
    await store.add_fact(fact("theirs", scope="user", scope_key="U2"))
    await store.add_fact(fact("project", scope="channel", scope_key=CHANNEL))

    deleted = await store.delete_facts("user", "U1")

    assert deleted == 1
    assert await store.facts("user", "U1") == []
    assert len(await store.facts("user", "U2")) == 1
    assert len(await store.facts("channel", CHANNEL)) == 1


async def test_deleting_missing_facts_is_a_no_op(store):
    assert await store.delete_facts("user", "nobody") == 0


# -- keyword recall ---------------------------------------------------------


async def seed_thread(store, *contents, channel=CHANNEL, thread_ts=THREAD_TS, role="user"):
    thread = await store.get_or_create_thread(channel, thread_ts)
    return [await store.append_message(thread.id, turn(content, role=role)) for content in contents]


async def test_recall_finds_terms_in_another_thread_of_the_same_channel(store):
    await seed_thread(store, "we should use postgres")
    await seed_thread(store, "we settled on wal mode", thread_ts="1700000000.000200")

    [hit] = await store.search_messages(
        channel_id=CHANNEL, terms=["wal"], exclude_thread_ts=THREAD_TS
    )

    assert hit.content == "we settled on wal mode"
    assert hit.channel_id == CHANNEL
    assert hit.thread_ts == "1700000000.000200"
    assert hit.role == "user"
    assert hit.created_at is not None


async def test_recall_keeps_author_names(store):
    await seed_thread(store, "we settled on wal mode", thread_ts="1700000000.000200")
    thread = await store.get_thread(CHANNEL, "1700000000.000200")
    await store.append_message(
        thread.id, Turn("assistant", "wal mode it is", 4, display_name="Vaultnel")
    )

    hits = await store.search_messages(
        channel_id=CHANNEL, terms=["wal"], exclude_thread_ts=THREAD_TS
    )

    assert sorted(hit.display_name or "" for hit in hits) == ["", "Vaultnel"]


async def test_recall_ignores_other_channels(store):
    await seed_thread(store, "wal mode elsewhere", channel="C9999999")

    assert (
        await store.search_messages(channel_id=CHANNEL, terms=["wal"], exclude_thread_ts=THREAD_TS)
        == []
    )


async def test_recall_excludes_the_current_thread(store):
    await seed_thread(store, "wal mode here")
    await seed_thread(store, "wal mode there", thread_ts="1700000000.000200")

    hits = await store.search_messages(
        channel_id=CHANNEL, terms=["wal"], exclude_thread_ts=THREAD_TS
    )

    assert [hit.thread_ts for hit in hits] == ["1700000000.000200"]

    # Without the exclusion the caller's own thread is visible to the store;
    # prompt assembly is what hides it from the final prompt.
    all_hits = await store.search_messages(channel_id=CHANNEL, terms=["wal"])
    assert {hit.thread_ts for hit in all_hits} == {THREAD_TS, "1700000000.000200"}


async def test_recall_hides_archived_threads(store):
    await seed_thread(store, "wal mode never mind", thread_ts="1700000000.000200")
    await store.archive_thread(CHANNEL, "1700000000.000200")

    assert (
        await store.search_messages(channel_id=CHANNEL, terms=["wal"], exclude_thread_ts=THREAD_TS)
        == []
    )


async def test_recall_hides_archived_rows_after_a_thread_is_reopened(store):
    await seed_thread(store, "wal mode before reset", thread_ts="1700000000.000200")
    await store.archive_thread(CHANNEL, "1700000000.000200")
    reopened = await store.get_or_create_thread(CHANNEL, "1700000000.000200")
    await store.append_message(reopened.id, turn("wal mode after reset"))

    hits = await store.search_messages(
        channel_id=CHANNEL, terms=["wal", "reset"], exclude_thread_ts=THREAD_TS
    )

    assert [hit.content for hit in hits] == ["wal mode after reset"]


async def test_recall_ranks_more_relevant_messages_first(store):
    await seed_thread(store, "wal and bananas", thread_ts="1700000000.000200")
    await seed_thread(store, "wal mode for sqlite", thread_ts="1700000000.000300")

    hits = await store.search_messages(
        channel_id=CHANNEL,
        terms=["wal", "mode", "sqlite"],
        exclude_thread_ts=THREAD_TS,
        limit=1,
    )

    assert [hit.content for hit in hits] == ["wal mode for sqlite"]


async def test_recall_respects_the_limit_and_empty_terms(store):
    for index in range(3):
        await seed_thread(store, f"wal mode note {index}", thread_ts=f"1700000000.00040{index}")

    hits = await store.search_messages(
        channel_id=CHANNEL, terms=["wal"], exclude_thread_ts=THREAD_TS, limit=2
    )
    assert len(hits) == 2

    assert await store.search_messages(channel_id=CHANNEL, terms=[]) == []
    assert (
        await store.search_messages(
            channel_id=CHANNEL, terms=["wal"], exclude_thread_ts=THREAD_TS, limit=0
        )
        == []
    )


async def test_reopening_an_old_database_rebuilds_the_recall_index(tmp_path):
    import aiosqlite

    path = tmp_path / "old.db"
    store = SqliteStore(path)
    await store.open()
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    await store.append_message(thread.id, turn("wal mode survived the upgrade"))
    await store.close()

    # Simulate a database written before the FTS table existed.
    conn = await aiosqlite.connect(path)
    await conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('delete-all')")
    await conn.execute("PRAGMA user_version = 0")
    await conn.commit()
    await conn.close()

    reopened = SqliteStore(path)
    await reopened.open()
    try:
        hits = await reopened.search_messages(
            channel_id=CHANNEL, terms=["upgrade"], exclude_thread_ts="0000.0000"
        )
    finally:
        await reopened.close()

    assert [hit.content for hit in hits] == ["wal mode survived the upgrade"]


# -- reset plumbing ---------------------------------------------------------


async def test_archiving_clears_the_summary(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    saved = await store.append_message(thread.id, turn())
    await store.set_summary(thread.id, "old summary", saved.id)

    archived = await store.archive_thread(CHANNEL, THREAD_TS)

    assert archived.summary == ""
    stored = await store.get_thread(CHANNEL, THREAD_TS)
    assert stored is not None and stored.summary == ""


async def test_bot_messages_map_back_to_their_thread(store):
    thread = await store.get_or_create_thread(CHANNEL, THREAD_TS)

    await store.link_bot_messages(thread.id, ["111.111", "222.222", ""])

    found = await store.thread_for_bot_message("222.222")
    assert found is not None and found.id == thread.id
    assert await store.thread_for_bot_message("333.333") is None


async def test_relinking_a_message_points_at_the_newest_thread(store):
    first = await store.get_or_create_thread(CHANNEL, THREAD_TS)
    second = await store.get_or_create_thread(CHANNEL, "1700000000.000200")

    await store.link_bot_messages(first.id, ["111.111"])
    await store.link_bot_messages(second.id, ["111.111"])

    found = await store.thread_for_bot_message("111.111")
    assert found is not None and found.id == second.id


async def test_linking_a_message_to_an_unknown_thread_fails(store):
    with pytest.raises(ValueError, match="thread"):
        await store.link_bot_messages(999, ["111.111"])
