"""In-memory implementation of the Store protocol.

This is a test double, but an unusually load-bearing one: handler and memory
tests use it, so it is held to the same contract suite as SqliteStore.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from slackbot_vaultnel.history.store import Thread, Turn


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryStore:
    def __init__(self, clock=_now) -> None:
        self._clock = clock
        self._threads: dict[int, Thread] = {}
        self._by_key: dict[tuple[str, str], int] = {}
        self._messages: dict[int, list[Turn]] = {}
        self._archived_messages: dict[int, set[int]] = {}
        self._next_thread_id = 1
        self._next_message_id = 1

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def get_thread(self, channel_id: str, thread_ts: str) -> Thread | None:
        thread_id = self._by_key.get((channel_id, thread_ts))
        return self._threads.get(thread_id) if thread_id is not None else None

    async def get_or_create_thread(self, channel_id: str, thread_ts: str) -> Thread:
        existing = await self.get_thread(channel_id, thread_ts)
        now = self._clock()
        if existing is None:
            thread = Thread(
                id=self._next_thread_id,
                channel_id=channel_id,
                thread_ts=thread_ts,
                created_at=now,
                updated_at=now,
            )
            self._next_thread_id += 1
            self._threads[thread.id] = thread
            self._by_key[(channel_id, thread_ts)] = thread.id
            self._messages[thread.id] = []
            self._archived_messages[thread.id] = set()
            return thread

        if existing.archived:
            high_water = max((m.id for m in self._messages[existing.id]), default=0)
            reopened = replace(
                existing,
                archived=False,
                summary="",
                summarized_through_msg_id=high_water,
                updated_at=now,
            )
            self._threads[existing.id] = reopened
            return reopened
        return existing

    def _require_thread(self, thread_id: int) -> Thread:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise ValueError(f"unknown thread {thread_id}")
        return thread

    async def append_message(self, thread_id: int, turn: Turn) -> Turn:
        thread = self._require_thread(thread_id)
        now = self._clock()
        stored = replace(turn, id=self._next_message_id, created_at=now)
        self._next_message_id += 1
        self._messages[thread_id].append(stored)
        self._threads[thread_id] = replace(thread, updated_at=now)
        return stored

    async def messages(
        self,
        thread_id: int,
        *,
        after_id: int = 0,
        limit: int | None = None,
        include_archived: bool = True,
    ) -> list[Turn]:
        self._require_thread(thread_id)
        archived = self._archived_messages[thread_id]
        selected = [
            m
            for m in self._messages[thread_id]
            if m.id > after_id and (include_archived or m.id not in archived)
        ]
        if limit is not None:
            selected = selected[-limit:]
        return list(selected)

    async def message_count(self, thread_id: int) -> int:
        self._require_thread(thread_id)
        return len(self._messages[thread_id])

    async def set_summary(self, thread_id: int, summary: str, through_msg_id: int) -> Thread:
        thread = self._require_thread(thread_id)
        self._validate_watermark(thread, through_msg_id, field="summarized_through_msg_id")
        updated = replace(
            thread,
            summary=summary,
            summarized_through_msg_id=through_msg_id,
            updated_at=self._clock(),
        )
        self._threads[thread_id] = updated
        return updated

    async def mark_extracted(self, thread_id: int, through_msg_id: int) -> Thread:
        thread = self._require_thread(thread_id)
        self._validate_watermark(thread, through_msg_id, field="last_extracted_msg_id")
        updated = replace(
            thread,
            last_extracted_msg_id=through_msg_id,
            updated_at=self._clock(),
        )
        self._threads[thread_id] = updated
        return updated

    def _validate_watermark(self, thread: Thread, through_msg_id: int, *, field: str) -> None:
        current = getattr(thread, field)
        if through_msg_id < current:
            raise ValueError(f"{field} cannot decrease ({through_msg_id} < {current})")
        known = {m.id for m in self._messages[thread.id]}
        if through_msg_id != 0 and through_msg_id not in known:
            raise ValueError(
                f"{field} must reference a message in thread {thread.id}: {through_msg_id}"
            )

    async def archive_thread(self, channel_id: str, thread_ts: str) -> Thread:
        thread = await self.get_thread(channel_id, thread_ts)
        if thread is None:
            raise ValueError(f"unknown thread {channel_id}/{thread_ts}")
        self._archived_messages[thread.id].update(m.id for m in self._messages[thread.id])
        updated = replace(thread, archived=True, updated_at=self._clock())
        self._threads[thread.id] = updated
        return updated


@dataclass(frozen=True)
class SlackCall:
    method: str
    channel: str
    thread_ts: str | None
    ts: str
    text: str
    when: datetime


class ManualClock:
    """Virtual time: tests advance it instead of sleeping."""

    def __init__(self, start: datetime = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> datetime:
        self._now += timedelta(seconds=seconds)
        return self._now


class FakeSlackClient:
    """Records Slack calls made by the renderer and handlers."""

    def __init__(
        self,
        *,
        clock: ManualClock | None = None,
        fail_updates: int = 0,
        users: dict[str, str] | None = None,
        fail_reactions: bool = False,
    ) -> None:
        self._clock = clock
        self.fail_updates = fail_updates
        self.users = users or {}
        self.fail_reactions = fail_reactions
        self.posts: list[SlackCall] = []
        self.updates: list[SlackCall] = []
        self.reactions: list[tuple[str, str, str]] = []
        self.ephemerals: list[tuple[str, str, str]] = []
        self.texts: dict[str, str] = {}
        self._next_ts = 1

    def _when(self) -> datetime:
        return self._clock.now() if self._clock else datetime.now(UTC)

    async def chat_postMessage(
        self, *, channel: str, thread_ts: str | None = None, text: str
    ) -> dict[str, object]:
        ts = f"1700000000.000{self._next_ts:03d}"
        self._next_ts += 1
        call = SlackCall("post", channel, thread_ts, ts, text, self._when())
        self.posts.append(call)
        self.texts[ts] = text
        return {"ok": True, "ts": ts}

    async def chat_update(self, *, channel: str, ts: str, text: str) -> dict[str, object]:
        if self.fail_updates > 0:
            self.fail_updates -= 1
            raise RuntimeError("slack is unhappy")
        call = SlackCall("update", channel, None, ts, text, self._when())
        self.updates.append(call)
        self.texts[ts] = text
        return {"ok": True, "ts": ts}

    async def reactions_add(self, *, channel: str, timestamp: str, name: str) -> dict[str, object]:
        if self.fail_reactions:
            raise RuntimeError("reactions are unavailable")
        self.reactions.append(("add", name, timestamp))
        return {"ok": True}

    async def reactions_remove(
        self, *, channel: str, timestamp: str, name: str
    ) -> dict[str, object]:
        if self.fail_reactions:
            raise RuntimeError("reactions are unavailable")
        self.reactions.append(("remove", name, timestamp))
        return {"ok": True}

    async def users_info(self, *, user: str) -> dict[str, object]:
        name = self.users.get(user)
        if name is None:
            raise RuntimeError(f"unknown user {user}")
        return {
            "ok": True,
            "user": {
                "id": user,
                "name": name,
                "real_name": name,
                "profile": {"display_name": name},
            },
        }

    async def chat_postEphemeral(self, *, channel: str, user: str, text: str) -> dict[str, object]:
        self.ephemerals.append((channel, user, text))
        return {"ok": True}


class ScriptedLLM:
    """Async LLM fake: canned reply, optional delay, records every prompt."""

    def __init__(
        self,
        *replies: str,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.replies = list(replies) or ["scripted reply"]
        self.delay = delay
        self.error = error
        self.calls: list[list[dict[str, str]]] = []
        self.active = 0
        self.max_active = 0

    def stream(self, messages, *, model=None, max_tokens=None, temperature=None):
        return self._stream(list(messages))

    async def _stream(self, messages):
        self.calls.append(messages)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.error is not None:
                raise self.error
            reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
            for chunk in reply:
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield chunk
        finally:
            self.active -= 1

    async def complete(self, messages, *, model=None, max_tokens=None, temperature=None):
        from slackbot_vaultnel.llm.client import Completion

        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return Completion(text=text, model="fake", tokens_in=1, tokens_out=1)
