"""The fake Slack client used by the local terminal harness."""

from __future__ import annotations

from slackbot_vaultnel.dev.chat import DEV_BOT, DEV_THREAD, build_event
from slackbot_vaultnel.dev.slack import FakeSlack
from slackbot_vaultnel.slack.render import PLACEHOLDER, SUCCESS_MARKER


async def test_streamed_updates_collapse_into_one_rendered_message():
    slack = FakeSlack()
    ts = str((await slack.chat_postMessage(channel="C", text=PLACEHOLDER))["ts"])

    await slack.chat_update(channel="C", ts=ts, text=f"hello\n\n{SUCCESS_MARKER}")

    assert slack.take_messages() == ["hello"]
    assert slack.take_messages() == []


async def test_long_replies_keep_every_chunk_in_order():
    slack = FakeSlack()
    first = str((await slack.chat_postMessage(channel="C", text=PLACEHOLDER))["ts"])
    second = str((await slack.chat_postMessage(channel="C", text=PLACEHOLDER))["ts"])

    await slack.chat_update(channel="C", ts=first, text="part one")
    await slack.chat_update(channel="C", ts=second, text=f"part two\n\n{SUCCESS_MARKER}")

    assert slack.take_messages() == ["part one", "part two"]


async def test_ephemeral_messages_reactions_and_names_are_captured():
    slack = FakeSlack(user_name="Ada")

    info = await slack.users_info(user="U1")
    await slack.chat_postEphemeral(channel="C", user="U1", text="cleared")
    await slack.reactions_add(channel="C", timestamp="1", name="eyes")
    await slack.reactions_remove(channel="C", timestamp="1", name="eyes")

    assert info["user"]["profile"]["display_name"] == "Ada"  # type: ignore[index]
    assert slack.take_ephemeral() == ["cleared"]
    assert slack.reactions == [("add", "eyes", "1"), ("remove", "eyes", "1")]


def test_events_look_like_threaded_mentions():
    event = build_event("what changed?", ts="7.000")

    assert event["type"] == "app_mention"
    assert event["channel"] == "C_DEV"
    assert event["thread_ts"] == DEV_THREAD
    assert event["text"] == f"<@{DEV_BOT}> what changed?"
