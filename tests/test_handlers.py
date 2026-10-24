"""Slack event handling: filtering, threading, memory, streaming (spec §4).

The Slack client, store and LLM are all injected, so these tests observe the
same things a user would: which thread got the reply, what the final message
says, and what ended up in thread memory.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fakes import FakeSlackClient, InMemoryStore, ManualClock, ScriptedLLM

from slackbot_vaultnel.config import Settings
from slackbot_vaultnel.history.store import StoredFact, Turn
from slackbot_vaultnel.llm.client import AuthError
from slackbot_vaultnel.memory.facts import WORKSPACE_KEY
from slackbot_vaultnel.memory.writer import MemoryWriter
from slackbot_vaultnel.slack.handlers import Handlers
from slackbot_vaultnel.slack.render import strip_status

BOT = "UBOT"
CHANNEL = "C0123456"
USER = "U0123456"
ROOT_TS = "1700000000.000100"


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "slack_bot_token": "xoxb-test-token",
        "slack_app_token": "xapp-test-token",
        "llm_api_key": "sk-test-key",
        "include_speaker_names": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make(
    *,
    reply: str = "the answer",
    clock: ManualClock | None = None,
    llm: ScriptedLLM | None = None,
    slack: FakeSlackClient | None = None,
    store=None,
    writer=None,
    **settings_overrides: object,
):
    clock = clock or ManualClock()
    slack = slack or FakeSlackClient(clock=clock)
    llm = llm or ScriptedLLM(reply)
    store = store or InMemoryStore(clock=clock.now)
    handlers = Handlers(
        client=slack,
        store=store,
        llm=llm,
        settings=settings(**settings_overrides),
        bot_user_id=BOT,
        clock=clock.now,
        writer=writer,
    )
    return handlers, slack, store, llm


class RecordingWriter:
    def __init__(self, store) -> None:
        self.store = store
        self.batches: list[tuple[int, list[tuple[str, str]]]] = []

    async def record(self, thread, turns) -> None:
        self.batches.append((thread.id, [(turn.role, turn.content) for turn in turns]))
        for turn in turns:
            await self.store.append_message(thread.id, turn)

    async def link(self, thread, slack_ts) -> None:
        await self.store.link_bot_messages(thread.id, slack_ts)


def mention(
    text: str,
    *,
    ts: str = ROOT_TS,
    thread_ts: str | None = None,
    user: str = USER,
    **extra: object,
) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "app_mention",
        "channel": CHANNEL,
        "user": user,
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    event.update(extra)
    return event


def dm(
    text: str,
    *,
    ts: str = ROOT_TS,
    thread_ts: str | None = None,
    **extra: object,
) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "message",
        "channel": CHANNEL,
        "channel_type": "im",
        "user": USER,
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    event.update(extra)
    return event


def visible(client: FakeSlackClient) -> str:
    """The first reply message as a reader sees it (marker stripped)."""

    return strip_status(client.texts[client.posts[0].ts])


async def stored_turns(store, thread_ts: str) -> list[tuple[str, str]]:
    thread = await store.get_thread(CHANNEL, thread_ts)
    assert thread is not None
    return [(turn.role, turn.content) for turn in await store.messages(thread.id)]


# -- mentions ---------------------------------------------------------------


async def test_mention_replies_in_the_message_thread_and_persists_both_turns():
    handlers, slack, store, _ = make(reply="Hi there")

    assert await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")

    assert len(slack.posts) == 1
    assert slack.posts[0].channel == CHANNEL
    assert slack.posts[0].thread_ts == ROOT_TS
    assert visible(slack) == "Hi there"
    assert ":white_check_mark:" in slack.texts[slack.posts[0].ts]
    assert await stored_turns(store, ROOT_TS) == [
        ("user", "hello"),
        ("assistant", "Hi there"),
    ]


async def test_mention_inside_a_thread_replies_in_that_thread():
    handlers, slack, store, _ = make()

    event = mention(f"<@{BOT}> follow up", ts="1700000000.000200", thread_ts=ROOT_TS)
    assert await handlers.handle_event(event, event_id="e1")

    assert slack.posts[0].thread_ts == ROOT_TS
    assert await stored_turns(store, ROOT_TS)


async def test_mention_stripping_accepts_the_labelled_form():
    handlers, slack, _, llm = make()

    await handlers.handle_event(mention(f"<@{BOT}|vaultnel>   what is up?"), event_id="e1")

    prompt = llm.calls[0]
    assert prompt[-1] == {"role": "user", "content": "what is up?"}
    assert strip_status(slack.texts[slack.posts[0].ts]) == "the answer"


async def test_a_bare_mention_gets_a_usage_hint_without_calling_the_model():
    handlers, slack, store, llm = make()

    assert await handlers.handle_event(mention(f"<@{BOT}>   "), event_id="e1")

    assert llm.calls == []
    assert len(slack.posts) == 1
    assert "ask me" in slack.posts[0].text.lower()
    assert await store.get_thread(CHANNEL, ROOT_TS) is None


async def test_second_turn_sees_the_first_turn_in_the_prompt():
    handlers, _, _, llm = make(reply="Your name is Ada.")

    await handlers.handle_event(mention(f"<@{BOT}> my name is Ada"), event_id="e1")
    await handlers.handle_event(
        mention(f"<@{BOT}> what is my name?", ts="1700000000.000200", thread_ts=ROOT_TS),
        event_id="e2",
    )

    second_prompt = llm.calls[1]
    contents = [message["content"] for message in second_prompt]
    assert "my name is Ada" in contents
    assert "Your name is Ada." in contents
    assert second_prompt[-1]["content"] == "what is my name?"


async def test_scoped_facts_are_injected_into_the_prompt():
    clock = ManualClock()
    store = InMemoryStore(clock=clock.now)
    await store.add_fact(StoredFact("user", USER, "Prefers TypeScript"))
    await store.add_fact(StoredFact("channel", CHANNEL, "Ships on Fly.io"))
    await store.add_fact(StoredFact("workspace", WORKSPACE_KEY, "Four-day week"))
    handlers, _, _, llm = make(clock=clock, store=store)

    await handlers.handle_event(mention(f"<@{BOT}> what do you know?"), event_id="e1")

    system = "\n".join(m["content"] for m in llm.calls[0] if m["role"] == "system")
    assert "Prefers TypeScript" in system
    assert "Ships on Fly.io" in system
    assert "Four-day week" in system


async def test_facts_extracted_after_one_turn_reach_the_next_prompt():
    clock = ManualClock()
    store = InMemoryStore(clock=clock.now)
    config = settings(fact_extraction_every=1)
    llm = ScriptedLLM(
        "noted",
        completions=[json.dumps([{"content": "Prefers TypeScript", "scope": "user"}])],
    )
    writer = MemoryWriter(store=store, llm=llm, settings=config, clock=clock.now)
    await writer.start()
    try:
        handlers = Handlers(
            client=FakeSlackClient(clock=clock),
            store=store,
            llm=llm,
            settings=config,
            bot_user_id=BOT,
            clock=clock.now,
            writer=writer,
        )
        await handlers.handle_event(mention(f"<@{BOT}> I like TypeScript"), event_id="e1")
        await writer.flush()
        await handlers.handle_event(
            mention(f"<@{BOT}> what do you know?", ts="1700000000.000200", thread_ts=ROOT_TS),
            event_id="e2",
        )
    finally:
        await writer.stop()

    system = "\n".join(m["content"] for m in llm.calls[1] if m["role"] == "system")
    assert "Prefers TypeScript" in system


async def test_recalled_snippets_from_other_threads_reach_the_prompt():
    handlers, _, store, llm = make()
    other = await store.get_or_create_thread(CHANNEL, "1700000000.000900")
    await store.append_message(
        other.id, Turn("user", "we settled on wal mode for sqlite", 8, display_name="Bob")
    )

    await handlers.handle_event(
        mention(f"<@{BOT}> what did we settle on about wal mode?"), event_id="e1"
    )

    system = "\n".join(m["content"] for m in llm.calls[0] if m["role"] == "system")
    assert "Relevant messages from other conversations" in system
    assert "we settled on wal mode for sqlite" in system


async def test_recall_failure_does_not_block_a_reply():
    class FailingRecallStore(InMemoryStore):
        async def search_messages(self, **kwargs):
            raise RuntimeError("fts is down")

    handlers, slack, _, _ = make(store=FailingRecallStore())

    assert await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")

    assert visible(slack) == "the answer"


async def test_speaker_names_are_resolved_when_enabled():
    clock = ManualClock()
    slack = FakeSlackClient(clock=clock, users={USER: "Ada"})
    handlers, _, _, llm = make(clock=clock, slack=slack, include_speaker_names=True)

    await handlers.handle_event(mention(f"<@{BOT}> hi"), event_id="e1")

    assert "The current speaker is Ada." in llm.calls[0][0]["content"]


# -- DMs and filtering ------------------------------------------------------


async def test_dm_is_answered_without_a_mention():
    handlers, slack, store, _ = make(reply="DM answer")

    assert await handlers.handle_event(dm("hello there"), event_id="e1")

    assert slack.posts[0].thread_ts == ROOT_TS
    assert await stored_turns(store, ROOT_TS) == [
        ("user", "hello there"),
        ("assistant", "DM answer"),
    ]


async def test_dm_thread_replies_stay_in_the_thread():
    handlers, slack, _, _ = make()

    event = dm("again", ts="1700000000.000200", thread_ts=ROOT_TS)
    assert await handlers.handle_event(event, event_id="e1")

    assert slack.posts[0].thread_ts == ROOT_TS


@pytest.mark.parametrize(
    "event",
    [
        mention(f"<@{BOT}> hi", user=BOT),
        mention(f"<@{BOT}> hi", bot_id="B999"),
        dm("hi", subtype="bot_message"),
        dm("edited", subtype="message_changed"),
        {**dm("hi"), "channel_type": "channel"},
        {**dm("hi"), "user": BOT},
    ],
)
async def test_ignored_events_never_reach_the_model(event):
    handlers, slack, store, llm = make()

    assert await handlers.handle_event(event, event_id="e1") is False

    assert slack.posts == []
    assert llm.calls == []
    assert await store.get_thread(CHANNEL, event.get("thread_ts", event["ts"])) is None


async def test_sharing_a_thread_serializes_answers():
    handlers, _, _, llm = make(reply="ok", llm=ScriptedLLM("ok", delay=0.01))

    first = mention(f"<@{BOT}> one", ts="1700000000.000200", thread_ts=ROOT_TS)
    second = mention(f"<@{BOT}> two", ts="1700000000.000300", thread_ts=ROOT_TS)

    await asyncio.gather(
        handlers.handle_event(first, event_id="e1"),
        handlers.handle_event(second, event_id="e2"),
    )

    assert len(llm.calls) == 2
    assert llm.max_active == 1


async def test_turns_are_handed_to_the_writer_as_one_failed_or_successful_batch():
    clock = ManualClock()
    store = InMemoryStore(clock=clock.now)
    writer = RecordingWriter(store)
    handlers, _, _, _ = make(clock=clock, store=store, writer=writer)

    await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")

    assert writer.batches == [(1, [("user", "hello"), ("assistant", "the answer")])]


async def test_a_duplicate_event_is_processed_once():
    handlers, slack, _, llm = make()
    event = mention(f"<@{BOT}> hello")

    assert await handlers.handle_event(event, event_id="same") is True
    assert await handlers.handle_event(event, event_id="same") is False

    assert len(llm.calls) == 1
    assert len(slack.posts) == 1


# -- reactions --------------------------------------------------------------


async def test_ack_reaction_becomes_a_check_mark_on_success():
    handlers, slack, _, _ = make()

    await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")

    assert slack.reactions == [
        ("add", "eyes", ROOT_TS),
        ("remove", "eyes", ROOT_TS),
        ("add", "white_check_mark", ROOT_TS),
    ]


async def test_reaction_failures_do_not_break_the_reply():
    clock = ManualClock()
    slack = FakeSlackClient(clock=clock, fail_reactions=True)
    handlers, _, _, _ = make(clock=clock, slack=slack)

    assert await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")

    assert visible(slack) == "the answer"


# -- reset via reaction -----------------------------------------------------


def reset_reaction(ts: str, *, name: str = "wastebasket", user: str = USER):
    return {
        "type": "reaction_added",
        "user": user,
        "reaction": name,
        "item": {"type": "message", "channel": CHANNEL, "ts": ts},
    }


async def test_wastebasket_reaction_archives_the_thread():
    handlers, slack, store, _ = make()
    await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")
    bot_ts = slack.posts[0].ts

    assert await handlers.handle_event(reset_reaction(bot_ts), event_id="r1")

    thread = await store.get_thread(CHANNEL, ROOT_TS)
    assert thread is not None and thread.archived is True
    assert thread.summary == ""


async def test_wastebasket_reaction_starts_a_fresh_context():
    handlers, slack, store, llm = make()
    await handlers.handle_event(mention(f"<@{BOT}> my codeword is pineapple"), event_id="e1")
    await handlers.handle_event(reset_reaction(slack.posts[0].ts), event_id="r1")

    await handlers.handle_event(
        mention(f"<@{BOT}> what is the codeword?", ts="1700000000.000200", thread_ts=ROOT_TS),
        event_id="e2",
    )

    prompt = "\n".join(m["content"] for m in llm.calls[1])
    assert "pineapple" not in prompt
    assert await stored_turns(store, ROOT_TS) == [
        ("user", "my codeword is pineapple"),
        ("assistant", "the answer"),
        ("user", "what is the codeword?"),
        ("assistant", "the answer"),
    ]


async def test_every_chunk_of_a_long_reply_can_be_reacted_to():
    handlers, slack, store, _ = make(reply="a paragraph of words\n\n" * 300)
    await handlers.handle_event(mention(f"<@{BOT}> tell me everything"), event_id="e1")
    assert len(slack.posts) > 1

    assert await handlers.handle_event(reset_reaction(slack.posts[-1].ts), event_id="r1")

    thread = await store.get_thread(CHANNEL, ROOT_TS)
    assert thread is not None and thread.archived is True


async def test_reaction_on_someone_elses_message_is_ignored():
    handlers, slack, store, _ = make()
    await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")

    ignored = await handlers.handle_event(reset_reaction("1700000000.999999"), event_id="r1")

    assert ignored is False
    thread = await store.get_thread(CHANNEL, ROOT_TS)
    assert thread is not None and thread.archived is False


async def test_other_reactions_do_not_reset_the_thread():
    handlers, slack, store, _ = make()
    await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")

    ignored = await handlers.handle_event(
        reset_reaction(slack.posts[0].ts, name="thumbsup"), event_id="r1"
    )

    assert ignored is False
    thread = await store.get_thread(CHANNEL, ROOT_TS)
    assert thread is not None and thread.archived is False


async def test_the_bot_cannot_reset_its_own_thread_with_a_reaction():
    handlers, slack, store, _ = make()
    await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")

    ignored = await handlers.handle_event(
        reset_reaction(slack.posts[0].ts, user=BOT), event_id="r1"
    )

    assert ignored is False
    thread = await store.get_thread(CHANNEL, ROOT_TS)
    assert thread is not None and thread.archived is False


# -- failures ---------------------------------------------------------------


async def test_model_failure_is_reported_and_keeps_the_user_turn():
    llm = ScriptedLLM(error=AuthError("bad key"))
    handlers, slack, store, _ = make(llm=llm)

    assert await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")

    assert ":warning:" in slack.texts[slack.posts[0].ts]
    assert "LLM_API_KEY" in slack.texts[slack.posts[0].ts]
    assert await stored_turns(store, ROOT_TS) == [("user", "hello")]
    assert ("add", "warning", ROOT_TS) in slack.reactions


async def test_an_empty_answer_is_reported_as_a_warning():
    handlers, slack, store, _ = make(reply="")

    await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")

    assert ":warning:" in slack.texts[slack.posts[0].ts]
    assert await stored_turns(store, ROOT_TS) == [("user", "hello")]


# -- /ask -------------------------------------------------------------------


async def test_ask_answers_in_the_channel_and_keeps_slash_memory():
    handlers, slack, store, _ = make(reply="slash answer")

    await handlers.handle_ask(
        {"command": "/ask", "text": "how are you", "channel_id": CHANNEL, "user_id": USER}
    )

    assert slack.posts[0].thread_ts is None
    assert visible(slack) == "slash answer"
    assert await store.get_thread(CHANNEL, f"slash-{CHANNEL}") is not None


async def test_ask_in_a_thread_uses_that_thread():
    handlers, slack, store, _ = make()

    await handlers.handle_ask(
        {
            "command": "/ask",
            "text": "thread question",
            "channel_id": CHANNEL,
            "user_id": USER,
            "thread_ts": ROOT_TS,
        }
    )

    assert slack.posts[0].thread_ts == ROOT_TS
    assert await stored_turns(store, ROOT_TS) == [
        ("user", "thread question"),
        ("assistant", "the answer"),
    ]


async def test_ask_without_text_gets_a_usage_hint_ephemerally():
    handlers, slack, _, llm = make()

    await handlers.handle_ask(
        {"command": "/ask", "text": "   ", "channel_id": CHANNEL, "user_id": USER}
    )

    assert llm.calls == []
    assert len(slack.ephemerals) == 1
    assert "usage" in slack.ephemerals[0][2].lower()


async def test_ask_reset_archives_the_thread_and_confirms_ephemerally():
    handlers, slack, store, _ = make()

    await handlers.handle_event(mention(f"<@{BOT}> hello"), event_id="e1")
    await handlers.handle_ask(
        {
            "command": "/ask",
            "text": "reset",
            "channel_id": CHANNEL,
            "user_id": USER,
            "thread_ts": ROOT_TS,
        }
    )

    thread = await store.get_thread(CHANNEL, ROOT_TS)
    assert thread is not None and thread.archived is True
    assert len(slack.ephemerals) == 1
    assert "cleared" in slack.ephemerals[0][2].lower()


async def test_ask_reset_without_a_thread_says_nothing_was_stored():
    handlers, slack, _, _ = make()

    await handlers.handle_ask(
        {"command": "/ask", "text": "reset", "channel_id": CHANNEL, "user_id": USER}
    )

    assert len(slack.ephemerals) == 1
    assert "no thread memory" in slack.ephemerals[0][2].lower()


async def test_ask_forget_me_deletes_only_that_users_facts():
    store = InMemoryStore()
    await store.add_fact(StoredFact("user", USER, "Prefers TypeScript"))
    await store.add_fact(StoredFact("user", "U999", "Prefers Go"))
    await store.add_fact(StoredFact("channel", CHANNEL, "Ships on Fly.io"))
    handlers, slack, _, _ = make(store=store)

    await handlers.handle_ask(
        {"command": "/ask", "text": "forget me", "channel_id": CHANNEL, "user_id": USER}
    )

    assert await store.facts("user", USER) == []
    assert len(await store.facts("user", "U999")) == 1
    assert len(await store.facts("channel", CHANNEL)) == 1
    assert "Deleted 1 fact" in slack.ephemerals[0][2]


async def test_ask_forget_me_with_nothing_stored_says_so():
    handlers, slack, _, _ = make()

    await handlers.handle_ask(
        {"command": "/ask", "text": "forget me", "channel_id": CHANNEL, "user_id": USER}
    )

    assert "No facts" in slack.ephemerals[0][2]


async def test_ask_reset_clears_the_channel_slash_memory():
    handlers, slack, store, _ = make()
    command = {"command": "/ask", "text": "hello", "channel_id": CHANNEL, "user_id": USER}

    await handlers.handle_ask(command)
    await handlers.handle_ask({**command, "text": "reset"})

    thread = await store.get_thread(CHANNEL, f"slash-{CHANNEL}")
    assert thread is not None and thread.archived is True
    assert "cleared" in slack.ephemerals[0][2].lower()
