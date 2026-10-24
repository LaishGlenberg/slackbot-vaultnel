"""The single write path for conversation memory (spec §5.2, §5.5).

All writes are serialized through one background task. That removes sqlite
write contention, keeps compaction from racing a reply, and gives failures one
place to be logged and swallowed: memory bookkeeping must never break a reply.

Compaction is rolling. Once the unsummarized turns in a thread pass
``COMPACT_TRIGGER_TOKENS``, everything except the newest
``KEEP_RECENT_TURNS`` is folded into ``threads.summary`` by one utility-model
call, and ``summarized_through_msg_id`` moves forward. An empty, oversized or
failed summary is rejected and retried on the next turn — the previous summary
and watermark stay put.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from slackbot_vaultnel.config import Settings
from slackbot_vaultnel.history.store import Store, Thread, Turn, utcnow
from slackbot_vaultnel.llm.client import LLMClient
from slackbot_vaultnel.llm.prompts import (
    FACT_EXTRACTION_MAX_TOKENS,
    build_extraction_messages,
    build_summary_messages,
)
from slackbot_vaultnel.memory.facts import parse_fact_entries, store_candidates
from slackbot_vaultnel.memory.tokens import count_tokens

logger = logging.getLogger(__name__)

#: A summary may be sloppy by this much; beyond it the previous one is kept.
SUMMARY_SIZE_GUARD = 1.5


class TurnWriter(Protocol):
    """What handlers need from memory: persist a batch of turns."""

    async def record(self, thread: Thread, turns: Sequence[Turn]) -> None: ...

    async def flush(self) -> None: ...


@dataclass(frozen=True)
class _Job:
    kind: Literal["turns", "extract"]
    thread: Thread
    turns: tuple[Turn, ...] = ()
    future: asyncio.Future[None] | None = None


class DirectTurnWriter:
    """Synchronous fallback used by tests and by callers without a queue."""

    def __init__(self, store: Store) -> None:
        self._store = store

    async def record(self, thread: Thread, turns: Sequence[Turn]) -> None:
        for turn in turns:
            await self._store.append_message(thread.id, turn)

    async def flush(self) -> None:
        return None


class MemoryWriter:
    def __init__(
        self,
        *,
        store: Store,
        llm: LLMClient,
        settings: Settings,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._store = store
        self._llm = llm
        self._clock = clock
        self._utility_model = settings.resolved_utility_model
        self._trigger = settings.compact_trigger_tokens
        self._keep_recent = settings.keep_recent_turns
        self._summary_max = settings.summary_max_tokens
        self._summary_limit = int(self._summary_max * SUMMARY_SIZE_GUARD)
        self._extract_every = settings.fact_extraction_every
        self._fact_max = settings.fact_max_per_scope
        self._queue: asyncio.Queue[_Job] = asyncio.Queue(settings.memory_queue_max)
        self._task: asyncio.Task[None] | None = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="memory-writer")

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        self._task = None
        await self._queue.join()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def flush(self) -> None:
        """Wait for queued writes and extraction jobs to finish."""

        await self._queue.join()

    async def record(self, thread: Thread, turns: Sequence[Turn]) -> None:
        """Enqueue a batch and wait for it (including any compaction) to finish."""

        if self._task is None:
            raise RuntimeError("memory writer is not running")
        job = _Job(
            kind="turns",
            thread=thread,
            turns=tuple(turns),
            future=asyncio.get_running_loop().create_future(),
        )
        await self._queue.put(job)
        assert job.future is not None
        await job.future

    # -- worker ------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job.kind == "turns":
                    await self._apply_turns(job)
                else:
                    await self._apply_extraction(job.thread)
            except Exception as exc:  # noqa: BLE001 - log, report to the caller
                logger.exception(
                    "memory job failed for %s/%s",
                    job.thread.channel_id,
                    job.thread.thread_ts,
                )
                if job.future is not None and not job.future.done():
                    job.future.set_exception(exc)
            else:
                if job.future is not None and not job.future.done():
                    job.future.set_result(None)
            finally:
                self._queue.task_done()

    async def _apply_turns(self, job: _Job) -> None:
        thread = await self._store.get_thread(job.thread.channel_id, job.thread.thread_ts)
        if thread is None or thread.archived:
            return  # a reset won the race; do not resurrect the thread
        for turn in job.turns:
            await self._store.append_message(thread.id, turn)
        await self._maybe_compact(thread)
        self._schedule_extraction(thread)

    # -- compaction --------------------------------------------------------

    async def _maybe_compact(self, thread: Thread) -> None:
        current = await self._store.get_thread(thread.channel_id, thread.thread_ts)
        if current is None or current.archived:
            return

        pending = await self._store.messages(
            current.id,
            after_id=current.summarized_through_msg_id,
            include_archived=False,
        )
        if sum(turn.tokens for turn in pending) <= self._trigger:
            return

        foldable = pending[: -self._keep_recent] if self._keep_recent else pending
        if not foldable:
            return

        try:
            completion = await self._llm.complete(
                build_summary_messages(current.summary, foldable),
                model=self._utility_model,
                max_tokens=self._summary_max,
            )
        except Exception:  # noqa: BLE001 - retried on the next turn
            logger.warning(
                "summary call failed for %s/%s",
                current.channel_id,
                current.thread_ts,
                exc_info=True,
            )
            return

        summary = (completion.text or "").strip()
        if not summary:
            logger.debug("rejected an empty summary for thread %s", current.id)
            return
        if count_tokens(summary) > self._summary_limit:
            logger.debug(
                "rejected an oversized summary for thread %s (%s > %s tokens)",
                current.id,
                count_tokens(summary),
                self._summary_limit,
            )
            return

        await self._store.set_summary(current.id, summary, foldable[-1].id)

    # -- fact extraction ---------------------------------------------------

    def _schedule_extraction(self, thread: Thread) -> None:
        """Low priority: drop it rather than make a persist job wait."""

        try:
            self._queue.put_nowait(_Job(kind="extract", thread=thread))
        except asyncio.QueueFull:
            logger.warning(
                "dropped fact extraction for %s/%s (memory queue full)",
                thread.channel_id,
                thread.thread_ts,
            )

    async def _apply_extraction(self, thread: Thread) -> None:
        current = await self._store.get_thread(thread.channel_id, thread.thread_ts)
        if current is None or current.archived:
            return

        pending = await self._store.messages(
            current.id,
            after_id=current.last_extracted_msg_id,
            include_archived=False,
        )
        assistant_turns = sum(1 for turn in pending if turn.role == "assistant")
        if assistant_turns < self._extract_every:
            return

        try:
            completion = await self._llm.complete(
                build_extraction_messages(pending),
                model=self._utility_model,
                max_tokens=FACT_EXTRACTION_MAX_TOKENS,
            )
        except Exception:  # noqa: BLE001 - retried on a later turn
            logger.warning(
                "fact extraction call failed for %s/%s",
                current.channel_id,
                current.thread_ts,
                exc_info=True,
            )
            return

        candidates = parse_fact_entries(completion.text or "")
        if candidates is None:
            logger.debug("fact extraction returned invalid JSON for thread %s", current.id)
            return  # do not advance the watermark; retry later

        user_id = next(
            (
                turn.slack_user
                for turn in reversed(pending)
                if turn.role == "user" and turn.slack_user
            ),
            None,
        )
        try:
            await store_candidates(
                self._store,
                candidates,
                channel_id=current.channel_id,
                user_id=user_id,
                source_msg_id=pending[-1].id,
                max_per_scope=self._fact_max,
            )
            await self._store.mark_extracted(current.id, pending[-1].id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "could not store facts for %s/%s",
                current.channel_id,
                current.thread_ts,
                exc_info=True,
            )
