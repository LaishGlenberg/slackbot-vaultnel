"""Streaming renderer: chunking, throttling and fence repair (spec §4.2–4.3).

The Slack client and the clock are injected, so every assertion here is about
observable calls: which message was posted, what text a user would see, and how
far apart updates landed. ``StreamerMachine`` hunts for orderings of deltas and
ticks that break the prefix or never-shrink guarantees.
"""

from __future__ import annotations

import asyncio

import pytest
from fakes import FakeSlackClient, ManualClock
from hypothesis import settings as hyp_settings
from hypothesis import stateful
from hypothesis import strategies as st

from slackbot_vaultnel.slack.render import (
    PLACEHOLDER,
    SLACK_TEXT_LIMIT,
    Renderer,
    split_message,
    strip_status,
)

CHANNEL = "C0123456"
THREAD_TS = "1700000000.000100"


def make(*, interval: float = 1.0, fail_updates: int = 0):
    clock = ManualClock()
    client = FakeSlackClient(clock=clock, fail_updates=fail_updates)
    renderer = Renderer(
        client,
        channel=CHANNEL,
        thread_ts=THREAD_TS,
        interval=interval,
        clock=clock.now,
    )
    return renderer, client, clock


# -- chunking ---------------------------------------------------------------


def test_short_text_is_a_single_chunk():
    assert split_message("hello") == ["hello"]


def test_chunks_respect_the_limit_and_rejoin_to_the_original():
    text = "\n\n".join(f"paragraph {i} " + "word " * 200 for i in range(6))

    chunks = split_message(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 3800 for chunk in chunks)
    assert "".join(chunks) == text


def test_paragraph_boundaries_are_preferred():
    text = "A" * 1900 + "\n\n" + "B" * 1900

    chunks = split_message(text)

    assert chunks[0].endswith("\n\n")
    assert chunks[1].startswith("B")


def test_sentence_boundaries_are_the_fallback():
    text = "x" * 2000 + ". " + "y" * 1900

    chunks = split_message(text)

    assert chunks[0].endswith(". ")
    assert "".join(chunks) == text


def test_hard_cut_when_there_is_no_boundary():
    text = "a" * 9000

    chunks = split_message(text)

    assert [len(chunk) for chunk in chunks] == [3800, 3800, 1400]
    assert "".join(chunks) == text


def test_code_fences_are_closed_and_reopened_across_chunks():
    body = "```python\n" + "\n".join(f"line {i} = {i}" for i in range(400)) + "\n```"
    assert len(body) > 3800

    chunks = split_message(body)

    assert len(chunks) > 1
    assert all(len(chunk) <= 3800 for chunk in chunks)
    for chunk in chunks:
        assert chunk.count("```") % 2 == 0
    assert chunks[1].startswith("```python\n")
    assert "line 399 = 399" in chunks[-1]


def test_fence_and_hard_cut_stay_inside_the_limit():
    text = "```\n" + "x" * 9000 + "\n```"

    chunks = split_message(text)

    assert all(len(chunk) <= 3800 for chunk in chunks)
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks)


def test_invalid_limit_is_rejected():
    with pytest.raises(ValueError):
        split_message("hello", limit=0)


# -- status decoration ------------------------------------------------------


def test_strip_status_removes_success_and_warning_markers():
    assert strip_status("answer\n\n:white_check_mark:") == "answer"
    assert strip_status("answer\n\n:warning: boom") == "answer"
    assert strip_status("plain answer") == "plain answer"
    assert strip_status(PLACEHOLDER) == ""


# -- renderer lifecycle -----------------------------------------------------


async def test_start_posts_the_placeholder_in_the_thread():
    renderer, client, _ = make()

    await renderer.start()

    assert len(client.posts) == 1
    assert client.posts[0].channel == CHANNEL
    assert client.posts[0].thread_ts == THREAD_TS
    assert "Thinking" in client.posts[0].text


async def test_small_delta_waits_for_the_interval():
    renderer, client, clock = make()
    await renderer.start()

    clock.advance(0.2)
    await renderer.push("hello")

    assert client.updates == []

    clock.advance(0.9)
    await renderer.tick()

    assert [u.text for u in client.updates] == ["hello"]


async def test_a_large_pending_buffer_flushes_without_waiting():
    renderer, client, _ = make()
    await renderer.start()

    await renderer.push("x" * 120)

    assert [u.text for u in client.updates] == ["x" * 120]


async def test_updates_are_prefixes_and_never_shrink():
    renderer, client, clock = make()
    await renderer.start()

    for delta in ("hello ", "there, ", "friend"):
        clock.advance(1.1)
        await renderer.push(delta)

    assert [u.text for u in client.updates] == ["hello ", "hello there, ", "hello there, friend"]


async def test_tick_without_new_text_does_not_update():
    renderer, client, clock = make()
    await renderer.start()
    clock.advance(2)
    await renderer.tick()

    assert client.updates == []


async def test_finish_posts_the_complete_answer_with_a_success_marker():
    renderer, client, clock = make()
    await renderer.start()
    clock.advance(1.1)
    await renderer.push("the answer is 4")

    result = await renderer.finish()

    visible = client.texts[result.ts]
    assert strip_status(visible) == "the answer is 4"
    assert visible.endswith(":white_check_mark:")
    assert result.text == "the answer is 4"


async def test_finish_without_any_text_marks_a_warning():
    renderer, client, _ = make()
    await renderer.start()

    result = await renderer.finish()

    assert ":warning:" in client.texts[result.ts]


async def test_long_answer_is_split_into_thread_messages():
    renderer, client, _ = make()
    await renderer.start()
    text = "\n\n".join(f"part {i} " + "word " * 300 for i in range(5))
    assert len(text) > SLACK_TEXT_LIMIT

    await renderer.push(text)
    result = await renderer.finish()

    assert len(result.chunk_ts) > 1
    assert all(len(client.texts[ts]) <= SLACK_TEXT_LIMIT for ts in result.chunk_ts)
    assert all(post.thread_ts == THREAD_TS for post in client.posts)
    assert strip_status("".join(client.texts[ts] for ts in result.chunk_ts)) == text


async def test_fail_keeps_partial_text_and_marks_the_error():
    renderer, client, _ = make()
    await renderer.start()
    await renderer.push("partial answer")

    result = await renderer.fail("the model gave up")

    visible = client.texts[result.ts]
    assert "partial answer" in visible
    assert ":warning: the model gave up" in visible


async def test_failed_updates_fall_back_to_a_fresh_message():
    renderer, client, clock = make(fail_updates=100)
    await renderer.start()
    clock.advance(1.1)
    await renderer.push("the full answer")

    result = await renderer.finish()

    assert any("the full answer" in post.text for post in client.posts[1:])
    assert strip_status(client.texts[result.chunk_ts[-1]]) == "the full answer"


async def test_push_after_finish_is_rejected():
    renderer, _, _ = make()
    await renderer.start()
    await renderer.finish()

    with pytest.raises(RuntimeError):
        await renderer.push("more")


# -- stateful ---------------------------------------------------------------

hyp_settings.register_profile("streamer", max_examples=40, deadline=None, stateful_step_count=30)

LOWERCASE = st.characters(min_codepoint=97, max_codepoint=122)
DELTAS = st.text(alphabet=st.one_of(LOWERCASE, st.just(" ")), min_size=1, max_size=20)


class StreamerMachine(stateful.RuleBasedStateMachine):
    """Arbitrary deltas and clock ticks, checked against one shadow string."""

    def __init__(self) -> None:
        super().__init__()
        self._loop = asyncio.new_event_loop()
        self.clock = ManualClock()
        self.client = FakeSlackClient(clock=self.clock)
        self.interval = 1.0
        self.renderer = Renderer(
            self.client,
            channel=CHANNEL,
            thread_ts=THREAD_TS,
            interval=self.interval,
            clock=self.clock.now,
        )
        self.full = ""
        self.finished = False
        self._previous_visible = ""
        self._run(self.renderer.start())

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def teardown(self) -> None:
        self._loop.close()
        super().teardown()

    @stateful.precondition(lambda self: not self.finished)
    @stateful.rule(text=DELTAS)
    def push(self, text: str) -> None:
        self.full += text
        self._run(self.renderer.push(text))

    @stateful.precondition(lambda self: not self.finished)
    @stateful.rule(seconds=st.floats(min_value=0, max_value=3))
    def tick(self, seconds: float) -> None:
        self.clock.advance(seconds)
        self._run(self.renderer.tick())

    @stateful.rule()
    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        self._run(self.renderer.finish())
        raw = self.client.texts[self.renderer.last_ts]
        if self.full:
            assert strip_status(raw) == self.full
        else:
            assert ":warning:" in raw

    @stateful.invariant()
    def streamed_text_is_a_prefix(self) -> None:
        if self.finished and not self.full:
            return
        visible = strip_status(self.client.texts[self.renderer.last_ts])
        assert self.full.startswith(visible)
        assert len(visible) >= len(self._previous_visible)
        self._previous_visible = visible

    @stateful.invariant()
    def updates_respect_the_interval_or_a_burst(self) -> None:
        if self.finished:
            return
        last_when = self.client.posts[0].when
        last_len = 0
        for update in self.client.updates:
            gap = (update.when - last_when).total_seconds()
            if gap < self.interval - 1e-9:
                assert len(update.text) - last_len >= 120
            last_when = update.when
            last_len = len(strip_status(update.text))


TestStreamer = StreamerMachine.TestCase
