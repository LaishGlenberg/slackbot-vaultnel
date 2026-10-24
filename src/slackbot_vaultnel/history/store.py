"""Persistence for threads, turns and rolling summaries.

The store is deliberately dumb: it never tokenizes, summarizes or decides what
belongs in a prompt. It guarantees ordering, monotonic watermarks and
idempotent thread identity, and nothing else. Everything policy-shaped lives in
``memory/``.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import aiosqlite

Role = Literal["user", "assistant", "system"]
FactScope = Literal["user", "channel", "workspace"]
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
#: Bumped whenever the on-disk index needs rebuilding from the content tables.
FTS_SCHEMA_VERSION = 1


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Turn:
    """One message in a thread. ``id`` is 0 until it is persisted."""

    role: Role
    content: str
    tokens: int
    slack_user: str | None = None
    display_name: str | None = None
    model: str | None = None
    id: int = 0
    created_at: datetime | None = None


@dataclass(frozen=True)
class StoredFact:
    """A durable long-term memory. ``id`` is 0 until it is persisted."""

    scope: FactScope
    scope_key: str
    content: str
    source_msg_id: int | None = None
    id: int = 0
    content_hash: str = ""
    created_at: datetime | None = None


@dataclass(frozen=True)
class Thread:
    id: int
    channel_id: str
    thread_ts: str
    summary: str = ""
    summarized_through_msg_id: int = 0
    last_extracted_msg_id: int = 0
    archived: bool = False
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True)
class MessageMatch:
    """One keyword-recall hit, already ranked best-first by the store."""

    content: str
    role: Role
    channel_id: str
    thread_ts: str
    slack_user: str | None = None
    display_name: str | None = None
    created_at: datetime | None = None


class Store(Protocol):
    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def get_thread(self, channel_id: str, thread_ts: str) -> Thread | None: ...

    async def get_or_create_thread(self, channel_id: str, thread_ts: str) -> Thread:
        """Identity lookup, with one twist: an archived thread is revived as an
        empty context whose watermark sits at its newest message, so nothing
        from before a reset can re-enter the prompt."""
        ...

    async def append_message(self, thread_id: int, turn: Turn) -> Turn: ...

    async def messages(
        self,
        thread_id: int,
        *,
        after_id: int = 0,
        limit: int | None = None,
        include_archived: bool = True,
    ) -> list[Turn]: ...

    async def message_count(self, thread_id: int) -> int: ...

    async def set_summary(self, thread_id: int, summary: str, through_msg_id: int) -> Thread: ...

    async def mark_extracted(self, thread_id: int, through_msg_id: int) -> Thread: ...

    async def archive_thread(self, channel_id: str, thread_ts: str) -> Thread: ...

    async def link_bot_messages(self, thread_id: int, slack_ts: Sequence[str]) -> None:
        """Remember which thread a bot message belongs to, so a reaction can reset it."""
        ...

    async def thread_for_bot_message(self, slack_ts: str) -> Thread | None: ...

    async def add_fact(self, fact: StoredFact) -> bool:
        """Insert unless an equivalent fact exists. Returns whether it was new."""
        ...

    async def facts(self, scope: FactScope, scope_key: str) -> list[StoredFact]: ...

    async def trim_facts(self, scope: FactScope, scope_key: str, keep: int) -> int:
        """Drop the oldest facts past ``keep``; returns how many were removed."""
        ...

    async def delete_facts(self, scope: FactScope, scope_key: str) -> int: ...

    async def search_messages(
        self,
        *,
        channel_id: str,
        terms: Sequence[str],
        exclude_thread_ts: str | None = None,
        limit: int = 3,
    ) -> list[MessageMatch]:
        """Keyword recall across other live threads in one channel.

        Results are best-match first. Archived threads and archived messages are
        invisible, and the caller's own thread is excluded because it is already
        in the short-term window.
        """
        ...


class SqliteStore:
    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] = utcnow) -> None:
        self._path = Path(path)
        self._clock = clock
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------

    async def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(self._path)
        conn.row_factory = aiosqlite.Row
        await conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await _migrate(conn)
        await conn.commit()
        self._conn = conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("store is not open")
        return self._conn

    # -- threads -----------------------------------------------------------

    async def get_thread(self, channel_id: str, thread_ts: str) -> Thread | None:
        cursor = await self._db.execute(
            "SELECT * FROM threads WHERE channel_id = ? AND thread_ts = ?",
            (channel_id, thread_ts),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _to_thread(row) if row is not None else None

    async def get_or_create_thread(self, channel_id: str, thread_ts: str) -> Thread:
        async with self._write_lock:
            thread = await self.get_thread(channel_id, thread_ts)
            now = self._clock()

            if thread is None:
                cursor = await self._db.execute(
                    "INSERT INTO threads (channel_id, thread_ts, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?)",
                    (channel_id, thread_ts, _iso(now), _iso(now)),
                )
                await self._db.commit()
                thread_id = cursor.lastrowid
                await cursor.close()
                assert thread_id is not None
                return Thread(
                    id=thread_id,
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    summary="",
                    summarized_through_msg_id=0,
                    last_extracted_msg_id=0,
                    archived=False,
                    created_at=now,
                    updated_at=now,
                )

            if not thread.archived:
                return thread

            cursor = await self._db.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages WHERE thread_id = ?",
                (thread.id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            high_water = int(row[0])

            await self._db.execute(
                "UPDATE threads SET archived = 0, summary = '',"
                " summarized_through_msg_id = ?, updated_at = ? WHERE id = ?",
                (high_water, _iso(now), thread.id),
            )
            await self._db.commit()
            return replace(
                thread,
                archived=False,
                summary="",
                summarized_through_msg_id=high_water,
                updated_at=now,
            )

    async def archive_thread(self, channel_id: str, thread_ts: str) -> Thread:
        async with self._write_lock:
            thread = await self.get_thread(channel_id, thread_ts)
            if thread is None:
                raise ValueError(f"unknown thread {channel_id}/{thread_ts}")
            now = self._clock()
            await self._db.execute(
                "UPDATE messages SET archived = 1 WHERE thread_id = ? AND archived = 0",
                (thread.id,),
            )
            await self._db.execute(
                "UPDATE threads SET archived = 1, summary = '', updated_at = ? WHERE id = ?",
                (_iso(now), thread.id),
            )
            await self._db.commit()
            return replace(thread, archived=True, summary="", updated_at=now)

    # -- bot messages ------------------------------------------------------

    async def link_bot_messages(self, thread_id: int, slack_ts: Sequence[str]) -> None:
        async with self._write_lock:
            await self._require_thread(thread_id)
            now = self._clock()
            await self._db.executemany(
                "INSERT OR REPLACE INTO bot_messages (slack_ts, thread_id, created_at)"
                " VALUES (?, ?, ?)",
                [(ts, thread_id, _iso(now)) for ts in slack_ts if ts],
            )
            await self._db.commit()

    async def thread_for_bot_message(self, slack_ts: str) -> Thread | None:
        cursor = await self._db.execute(
            "SELECT t.* FROM bot_messages b JOIN threads t ON t.id = b.thread_id"
            " WHERE b.slack_ts = ?",
            (slack_ts,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _to_thread(row) if row is not None else None

    # -- facts -------------------------------------------------------------

    async def add_fact(self, fact: StoredFact) -> bool:
        async with self._write_lock:
            now = self._clock()
            cursor = await self._db.execute(
                "INSERT OR IGNORE INTO facts"
                " (scope, scope_key, content, content_hash, source_msg_id, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fact.scope,
                    fact.scope_key,
                    fact.content,
                    fact_content_hash(fact.scope, fact.content),
                    fact.source_msg_id,
                    _iso(now),
                ),
            )
            inserted = cursor.rowcount > 0
            await cursor.close()
            await self._db.commit()
            return inserted

    async def facts(self, scope: FactScope, scope_key: str) -> list[StoredFact]:
        cursor = await self._db.execute(
            "SELECT * FROM facts WHERE scope = ? AND scope_key = ? ORDER BY id ASC",
            (scope, scope_key),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_to_fact(row) for row in rows]

    async def trim_facts(self, scope: FactScope, scope_key: str, keep: int) -> int:
        if keep < 0:
            raise ValueError("keep must be >= 0")
        async with self._write_lock:
            cursor = await self._db.execute(
                "DELETE FROM facts WHERE id IN ("
                " SELECT id FROM facts WHERE scope = ? AND scope_key = ?"
                " ORDER BY id DESC LIMIT -1 OFFSET ?)",
                (scope, scope_key, keep),
            )
            removed = cursor.rowcount
            await cursor.close()
            await self._db.commit()
            return max(removed, 0)

    async def delete_facts(self, scope: FactScope, scope_key: str) -> int:
        async with self._write_lock:
            cursor = await self._db.execute(
                "DELETE FROM facts WHERE scope = ? AND scope_key = ?",
                (scope, scope_key),
            )
            removed = cursor.rowcount
            await cursor.close()
            await self._db.commit()
            return max(removed, 0)

    # -- messages ----------------------------------------------------------

    async def append_message(self, thread_id: int, turn: Turn) -> Turn:
        async with self._write_lock:
            thread = await self._require_thread(thread_id)
            now = self._clock()
            cursor = await self._db.execute(
                "INSERT INTO messages"
                " (thread_id, role, content, slack_user, display_name, model, tokens,"
                "  created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    turn.role,
                    turn.content,
                    turn.slack_user,
                    turn.display_name,
                    turn.model,
                    turn.tokens,
                    _iso(now),
                ),
            )
            message_id = cursor.lastrowid
            await cursor.close()
            await self._db.execute(
                "UPDATE threads SET updated_at = ? WHERE id = ?",
                (_iso(now), thread.id),
            )
            await self._db.commit()
            assert message_id is not None
            return replace(turn, id=message_id, created_at=now)

    async def messages(
        self,
        thread_id: int,
        *,
        after_id: int = 0,
        limit: int | None = None,
        include_archived: bool = True,
    ) -> list[Turn]:
        await self._require_thread(thread_id)
        archived_clause = "" if include_archived else " AND archived = 0"
        if limit is None:
            sql = (
                "SELECT * FROM messages WHERE thread_id = ? AND id > ?"
                f"{archived_clause} ORDER BY id ASC"
            )
            params: tuple[object, ...] = (thread_id, after_id)
        else:
            sql = (
                "SELECT * FROM (SELECT * FROM messages"
                " WHERE thread_id = ? AND id > ?" + archived_clause + ""
                " ORDER BY id DESC LIMIT ?) ORDER BY id ASC"
            )
            params = (thread_id, after_id, limit)
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return [_to_turn(row) for row in rows]

    async def message_count(self, thread_id: int) -> int:
        await self._require_thread(thread_id)
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0])

    # -- keyword recall ----------------------------------------------------

    async def search_messages(
        self,
        *,
        channel_id: str,
        terms: Sequence[str],
        exclude_thread_ts: str | None = None,
        limit: int = 3,
    ) -> list[MessageMatch]:
        match = _match_expression(terms)
        if limit <= 0 or not match:
            return []
        exclude = "" if exclude_thread_ts is None else " AND t.thread_ts <> ?"
        params: tuple[object, ...] = (match, channel_id)
        if exclude_thread_ts is not None:
            params += (exclude_thread_ts,)
        sql = (
            "SELECT m.content, m.role, m.slack_user, m.display_name, m.created_at,"
            " t.channel_id, t.thread_ts"
            " FROM messages_fts"
            " JOIN messages m ON m.id = messages_fts.rowid"
            " JOIN threads t ON t.id = m.thread_id"
            " WHERE messages_fts MATCH ?"
            " AND t.channel_id = ?"
            " AND t.archived = 0 AND m.archived = 0"
            " AND m.role IN ('user', 'assistant')"
            f"{exclude}"
            " ORDER BY bm25(messages_fts), m.id DESC"
            " LIMIT ?"
        )
        cursor = await self._db.execute(sql, (*params, limit))
        rows = await cursor.fetchall()
        await cursor.close()
        return [_to_match(row) for row in rows]

    # -- watermarks --------------------------------------------------------

    async def set_summary(self, thread_id: int, summary: str, through_msg_id: int) -> Thread:
        async with self._write_lock:
            thread = await self._require_thread(thread_id)
            await self._validate_watermark(thread, through_msg_id, "summarized_through_msg_id")
            now = self._clock()
            await self._db.execute(
                "UPDATE threads SET summary = ?, summarized_through_msg_id = ?,"
                " updated_at = ? WHERE id = ?",
                (summary, through_msg_id, _iso(now), thread_id),
            )
            await self._db.commit()
            return replace(
                thread,
                summary=summary,
                summarized_through_msg_id=through_msg_id,
                updated_at=now,
            )

    async def mark_extracted(self, thread_id: int, through_msg_id: int) -> Thread:
        async with self._write_lock:
            thread = await self._require_thread(thread_id)
            await self._validate_watermark(thread, through_msg_id, "last_extracted_msg_id")
            now = self._clock()
            await self._db.execute(
                "UPDATE threads SET last_extracted_msg_id = ?, updated_at = ? WHERE id = ?",
                (through_msg_id, _iso(now), thread_id),
            )
            await self._db.commit()
            return replace(thread, last_extracted_msg_id=through_msg_id, updated_at=now)

    # -- helpers -----------------------------------------------------------

    async def _require_thread(self, thread_id: int) -> Thread:
        cursor = await self._db.execute("SELECT * FROM threads WHERE id = ?", (thread_id,))
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise ValueError(f"unknown thread {thread_id}")
        return _to_thread(row)

    async def _validate_watermark(self, thread: Thread, through_msg_id: int, field: str) -> None:
        current = getattr(thread, field)
        if through_msg_id < current:
            raise ValueError(f"{field} cannot decrease ({through_msg_id} < {current})")
        if through_msg_id == 0:
            return
        cursor = await self._db.execute(
            "SELECT 1 FROM messages WHERE id = ? AND thread_id = ?",
            (through_msg_id, thread.id),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise ValueError(
                f"{field} must reference a message in thread {thread.id}: {through_msg_id}"
            )


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


async def _migrate(conn: aiosqlite.Connection) -> None:
    """Populate the full-text index for rows written before it existed."""

    cursor = await conn.execute("PRAGMA user_version")
    row = await cursor.fetchone()
    await cursor.close()
    version = int(row[0]) if row is not None else 0
    if version < FTS_SCHEMA_VERSION:
        await conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        await conn.execute(f"PRAGMA user_version = {FTS_SCHEMA_VERSION}")


def _match_expression(terms: Sequence[str]) -> str:
    """OR together quoted terms so nothing in a user message is read as syntax."""

    quoted = [f'"{term.replace(chr(34), "")}"' for term in terms if term]
    return " OR ".join(quoted)


def _to_thread(row: aiosqlite.Row) -> Thread:
    return Thread(
        id=row["id"],
        channel_id=row["channel_id"],
        thread_ts=row["thread_ts"],
        summary=row["summary"],
        summarized_through_msg_id=row["summarized_through_msg_id"],
        last_extracted_msg_id=row["last_extracted_msg_id"],
        archived=bool(row["archived"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _to_fact(row: aiosqlite.Row) -> StoredFact:
    return StoredFact(
        id=row["id"],
        scope=row["scope"],
        scope_key=row["scope_key"],
        content=row["content"],
        content_hash=row["content_hash"],
        source_msg_id=row["source_msg_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def fact_content_hash(scope: str, content: str) -> str:
    normalized = re.sub(r"\s+", " ", content).strip().lower()
    return hashlib.sha256(f"{scope}{normalized}".encode()).hexdigest()


def _to_turn(row: aiosqlite.Row) -> Turn:
    return Turn(
        id=row["id"],
        role=row["role"],
        content=row["content"],
        tokens=row["tokens"],
        slack_user=row["slack_user"],
        display_name=row["display_name"],
        model=row["model"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _to_match(row: aiosqlite.Row) -> MessageMatch:
    return MessageMatch(
        content=row["content"],
        role=row["role"],
        channel_id=row["channel_id"],
        thread_ts=row["thread_ts"],
        slack_user=row["slack_user"],
        display_name=row["display_name"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
