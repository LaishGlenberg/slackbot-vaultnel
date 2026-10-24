"""Background writer + rolling compaction (spec §5.2, §5.5).

Compaction is policy, so these tests script the summarizer and assert the
observable results: which turns were folded, where the watermark stopped, and
whether a bad summary was allowed through. A summary that is empty, huge or
failed must leave the previous one in place and be retried on the next turn.
"""

from __future__ import annotations

import json

from fakes import InMemoryStore, ManualClock, ScriptedLLM

from slackbot_vaultnel.config import Settings
from slackbot_vaultnel.history.store import Thread, Turn
from slackbot_vaultnel.llm.client import LLMError
from slackbot_vaultnel.memory.facts import WORKSPACE_KEY
from slackbot_vaultnel.memory.writer import MemoryWriter

CHANNEL = "C0123456"
ROOT_TS = "1700000000.000100"


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "slack_bot_token": "xoxb-test-token",
        "slack_app_token": "xapp-test-token",
        "llm_api_key": "sk-test-key",
        "compact_trigger_tokens": 20,
        "keep_recent_turns": 2,
        "summary_max_tokens": 15,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def turn(
    content: str,
    *,
    role: str = "user",
    tokens: int = 6,
    slack_user: str | None = None,
) -> Turn:
    return Turn(role=role, content=content, tokens=tokens, slack_user=slack_user)


class Harness:
    def __init__(self, *, llm: ScriptedLLM, **overrides: object) -> None:
        self.clock = ManualClock()
        self.store = InMemoryStore(clock=self.clock.now)
        self.llm = llm
        self.writer = MemoryWriter(
            store=self.store,
            llm=self.llm,
            settings=settings(**overrides),
            clock=self.clock.now,
        )

    async def __aenter__(self) -> Harness:
        await self.writer.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.writer.stop()

    async def thread(self) -> Thread:
        return await self.store.get_or_create_thread(CHANNEL, ROOT_TS)

    async def record(self, thread: Thread, *turns: Turn) -> None:
        await self.writer.record(thread, turns)

    async def flush(self) -> None:
        await self.writer.flush()


async def test_record_persists_turns_in_order_without_compacting_below_the_trigger():
    async with Harness(llm=ScriptedLLM(completions=["unused"])) as h:
        thread = await h.thread()

        await h.record(thread, turn("one", tokens=2), turn("two", role="assistant", tokens=2))

        stored = await h.store.messages(thread.id)
        assert [(t.role, t.content) for t in stored] == [("user", "one"), ("assistant", "two")]
        assert h.llm.completion_calls == []
        assert (await h.store.get_thread(CHANNEL, ROOT_TS)).summary == ""


async def test_compaction_folds_old_turns_and_keeps_the_newest_verbatim():
    async with Harness(llm=ScriptedLLM(completions=["folded summary"])) as h:
        thread = await h.thread()
        await h.record(thread, *(turn(f"turn {i}") for i in range(1, 6)))

        fresh = await h.store.get_thread(CHANNEL, ROOT_TS)
        stored = await h.store.messages(thread.id)
        folded = stored[:3]

        assert fresh.summary == "folded summary"
        assert fresh.summarized_through_msg_id == folded[-1].id
        assert [t.content for t in stored[3:]] == ["turn 4", "turn 5"]
        prompt = "\n".join(message["content"] for message in h.llm.completion_calls[0])
        assert "turn 1" in prompt
        assert "turn 3" in prompt
        assert "turn 4" not in prompt
        assert h.llm.completion_kwargs[0]["model"] == "deepseek-flash"
        assert h.llm.completion_kwargs[0]["max_tokens"] == 15


async def test_a_second_compaction_rolls_the_previous_summary_forward():
    async with Harness(llm=ScriptedLLM(completions=["first summary", "second summary"])) as h:
        thread = await h.thread()
        await h.record(thread, *(turn(f"turn {i}") for i in range(1, 6)))
        thread = await h.store.get_thread(CHANNEL, ROOT_TS)
        await h.record(thread, turn("turn 6"), turn("turn 7"))

        fresh = await h.store.get_thread(CHANNEL, ROOT_TS)
        assert fresh.summary == "second summary"
        prompt = "\n".join(message["content"] for message in h.llm.completion_calls[1])
        assert "first summary" in prompt
        assert "turn 4" in prompt
        assert "turn 5" in prompt
        assert "turn 6" not in prompt
        assert "turn 7" not in prompt


async def test_an_empty_summary_is_rejected_and_retried_next_turn():
    async with Harness(llm=ScriptedLLM(completions=["   ", "good summary"])) as h:
        thread = await h.thread()
        await h.record(thread, *(turn(f"turn {i}") for i in range(1, 6)))

        fresh = await h.store.get_thread(CHANNEL, ROOT_TS)
        assert fresh.summary == ""
        assert fresh.summarized_through_msg_id == 0

        await h.record(thread, turn("turn 6"))
        fresh = await h.store.get_thread(CHANNEL, ROOT_TS)
        assert fresh.summary == "good summary"
        assert fresh.summarized_through_msg_id > 0


async def test_an_overlong_summary_is_rejected():
    long_text = "this summary is far too long to accept"
    async with Harness(
        llm=ScriptedLLM(completions=[long_text]),
        summary_max_tokens=2,
        compact_trigger_tokens=20,
    ) as h:
        thread = await h.thread()
        await h.record(thread, *(turn(f"turn {i}") for i in range(1, 6)))

        fresh = await h.store.get_thread(CHANNEL, ROOT_TS)
        assert fresh.summary == ""
        assert fresh.summarized_through_msg_id == 0


async def test_a_failed_summary_call_leaves_memory_untouched():
    async with Harness(llm=ScriptedLLM(completion_error=LLMError("nope"))) as h:
        thread = await h.thread()
        await h.record(thread, *(turn(f"turn {i}") for i in range(1, 6)))

        fresh = await h.store.get_thread(CHANNEL, ROOT_TS)
        assert fresh.summary == ""
        assert fresh.summarized_through_msg_id == 0
        assert (await h.store.message_count(thread.id)) == 5


async def test_nothing_is_folded_when_every_turn_is_inside_the_recent_window():
    async with Harness(llm=ScriptedLLM(completions=["unused"])) as h:
        thread = await h.thread()
        await h.record(thread, turn("one", tokens=12), turn("two", tokens=12))

        assert h.llm.completion_calls == []
        assert (await h.store.get_thread(CHANNEL, ROOT_TS)).summary == ""


async def test_extraction_waits_for_four_assistant_turns_and_scopes_the_facts():
    completion = json.dumps(
        [
            {"content": "Prefers TypeScript", "scope": "user"},
            {"content": "Ships on Fly.io", "scope": "channel"},
            {"content": "Four-day week", "scope": "workspace"},
        ]
    )
    async with Harness(llm=ScriptedLLM(completions=[completion])) as h:
        thread = await h.thread()
        await h.record(
            thread,
            turn("hello", tokens=1, slack_user="U1"),
            *(turn(f"answer {i}", role="assistant", tokens=1) for i in range(4)),
        )
        await h.flush()

        assert [f.content for f in await h.store.facts("user", "U1")] == ["Prefers TypeScript"]
        assert [f.content for f in await h.store.facts("channel", CHANNEL)] == ["Ships on Fly.io"]
        assert [f.content for f in await h.store.facts("workspace", WORKSPACE_KEY)] == [
            "Four-day week"
        ]
        fresh = await h.store.get_thread(CHANNEL, ROOT_TS)
        assert fresh.last_extracted_msg_id == max(m.id for m in await h.store.messages(thread.id))
        prompt = "\n".join(message["content"] for message in h.llm.completion_calls[0])
        assert "answer 3" in prompt


async def test_extraction_does_not_run_before_the_interval():
    async with Harness(llm=ScriptedLLM(completions=["[]"])) as h:
        thread = await h.thread()
        await h.record(
            thread,
            turn("hello", tokens=1),
            *(turn(f"answer {i}", role="assistant", tokens=1) for i in range(3)),
        )
        await h.flush()

        assert h.llm.completion_calls == []
        assert (await h.store.get_thread(CHANNEL, ROOT_TS)).last_extracted_msg_id == 0


async def test_invalid_json_leaves_the_watermark_and_retries_later():
    good = json.dumps([{"content": "Prefers TypeScript", "scope": "user"}])
    async with Harness(llm=ScriptedLLM(completions=["{not json", good])) as h:
        thread = await h.thread()
        await h.record(
            thread,
            turn("hello", tokens=1, slack_user="U1"),
            *(turn(f"answer {i}", role="assistant", tokens=1) for i in range(4)),
        )
        await h.flush()

        assert (await h.store.get_thread(CHANNEL, ROOT_TS)).last_extracted_msg_id == 0
        assert await h.store.facts("user", "U1") == []

        await h.record(thread, turn("answer 4", role="assistant", tokens=1))
        await h.flush()

        assert (await h.store.get_thread(CHANNEL, ROOT_TS)).last_extracted_msg_id > 0
        assert [f.content for f in await h.store.facts("user", "U1")] == ["Prefers TypeScript"]


async def test_the_next_extraction_only_sees_turns_since_the_watermark():
    async with Harness(llm=ScriptedLLM(completions=["[]", "[]"])) as h:
        thread = await h.thread()
        await h.record(
            thread,
            turn("hello", tokens=1),
            *(turn(f"old answer {i}", role="assistant", tokens=1) for i in range(4)),
        )
        await h.flush()
        await h.record(
            thread,
            *(turn(f"new answer {i}", role="assistant", tokens=1) for i in range(4)),
        )
        await h.flush()

        second_prompt = "\n".join(message["content"] for message in h.llm.completion_calls[1])
        assert "old answer 0" not in second_prompt
        assert "new answer 3" in second_prompt


async def test_extraction_failure_is_silent_and_retryable():
    async with Harness(llm=ScriptedLLM(completion_error=LLMError("down"))) as h:
        thread = await h.thread()
        await h.record(
            thread,
            turn("hello", tokens=1),
            *(turn(f"answer {i}", role="assistant", tokens=1) for i in range(4)),
        )
        await h.flush()

        assert (await h.store.get_thread(CHANNEL, ROOT_TS)).last_extracted_msg_id == 0
        assert await h.store.facts("channel", CHANNEL) == []


async def test_an_empty_array_still_advances_the_watermark():
    async with Harness(llm=ScriptedLLM(completions=["[]"])) as h:
        thread = await h.thread()
        await h.record(
            thread,
            turn("hello", tokens=1),
            *(turn(f"answer {i}", role="assistant", tokens=1) for i in range(4)),
        )
        await h.flush()

        assert (await h.store.get_thread(CHANNEL, ROOT_TS)).last_extracted_msg_id > 0
        assert await h.store.facts("channel", CHANNEL) == []


async def test_archived_threads_are_not_compacted_or_appended_to():
    async with Harness(llm=ScriptedLLM(completions=["unused"])) as h:
        thread = await h.thread()
        await h.record(thread, *(turn(f"turn {i}", tokens=1) for i in range(1, 6)))
        await h.store.archive_thread(CHANNEL, ROOT_TS)
        before = await h.store.message_count(thread.id)

        await h.record(thread, turn("after reset"))

        assert await h.store.message_count(thread.id) == before
        assert h.llm.completion_calls == []
        assert (await h.store.get_thread(CHANNEL, ROOT_TS)).archived is True


async def test_bot_message_links_go_through_the_queue():
    async with Harness(llm=ScriptedLLM("unused")) as harness:
        thread = await harness.thread()

        await harness.writer.link(thread, ["111.111", "222.222"])

        found = await harness.store.thread_for_bot_message("222.222")
        assert found is not None and found.id == thread.id
