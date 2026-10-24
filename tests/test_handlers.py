"""Slack event handling: filtering, threading, memory, streaming (spec §4).

The Slack client, store and LLM are all injected, so these tests observe the
same things a user would: which thread got the reply, what the final message
says, and what ended up in thread memory.
"""

from __future__ import annotations

import asyncio

import pytest
from fakes import FakeSlackClient, InMemoryStore, ManualClock, ScriptedLLM

from slackbot_vaultnel.config import Settings
from slackbot_vaultnel.llm.client import AuthError
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


async def test_ask_reset_clears_the_channel_slash_memory():
    handlers, slack, store, _ = make()
    command = {"command": "/ask", "text": "hello", "channel_id": CHANNEL, "user_id": USER}

    await handlers.handle_ask(command)
    await handlers.handle_ask({**command, "text": "reset"})

    thread = await store.get_thread(CHANNEL, f"slash-{CHANNEL}")
    assert thread is not None and thread.archived is True
    assert "cleared" in slack.ephemerals[0][2].lower()
