"""In-memory implementation of the Store protocol.

This is a test double, but an unusually load-bearing one: handler and memory
tests use it, so it is held to the same contract suite as SqliteStore.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

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
