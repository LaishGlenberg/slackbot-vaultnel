"""Streaming replies: placeholder, throttled updates, chunking (spec §4.2–4.3).

Slack's `chat.update` is Tier 3 (~1 request/second per channel) and message
text is capped at 4000 characters. This module owns both constraints:

* deltas accumulate; an update happens at most once per ``STREAM_INTERVAL``,
  or immediately when more than ``BURST_CHARS`` characters are pending;
* a long answer is split at paragraph → sentence → hard boundaries, with code
  fences closed and reopened so every message renders on its own.

The client and clock are injected. Nothing here knows about Bolt, the LLM or
background tasks — the handler drives it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from slackbot_vaultnel.history.store import utcnow

logger = logging.getLogger(__name__)

#: Soft cap: leaves room for fence repair and status markers.
CHUNK_CHARS = 3800
#: Slack rejects text longer than this.
SLACK_TEXT_LIMIT = 4000
#: Flushing sooner than the interval is allowed only past this buffer size.
BURST_CHARS = 120

PLACEHOLDER = ":eyes: _Thinking…_"
SUCCESS_MARKER = ":white_check_mark:"
WARNING_MARKER = ":warning:"
MIN_CHUNK_CHARS = 32

_FENCE_RE = re.compile(r"^ {0,3}(`{3,})[^\n]*$", re.MULTILINE)


class SlackReplyClient(Protocol):
    async def chat_postMessage(
        self, *, channel: str, thread_ts: str | None = None, text: str
    ) -> dict[str, object]: ...

    async def chat_update(self, *, channel: str, ts: str, text: str) -> dict[str, object]: ...


@dataclass(frozen=True)
class StreamResult:
    ts: str
    text: str
    chunk_ts: list[str] = field(default_factory=list)
    updates: int = 0


# -- chunking ---------------------------------------------------------------


def split_message(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split ``text`` into messages of at most ``limit`` characters.

    Boundaries are preferred in this order: paragraph, newline, sentence, then
    a hard cut. Code fences are tracked across the split so no chunk contains
    an unclosed fence. Concatenating the result without the repairs yields the
    original text, minus any characters the hard cut had to move.
    """

    if limit < MIN_CHUNK_CHARS:
        raise ValueError(f"limit must be at least {MIN_CHUNK_CHARS} characters")
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    pos = 0
    fence: str | None = None
    while pos < len(text):
        prefix = f"{fence}\n" if fence else ""
        budget = limit - len(prefix)
        if budget < MIN_CHUNK_CHARS:
            raise ValueError("limit leaves no room once the code fence is reopened")

        end = min(pos + budget, len(text))
        if end < len(text):
            end = _cut_point(text, pos, end)
            end = _avoid_partial_fence(text, pos, end)

        piece = text[pos:end]
        new_fence = _advance_fence(fence, piece)
        closing = "\n```" if new_fence else ""
        while len(prefix) + len(piece) + len(closing) > limit:
            end -= 1
            if end <= pos:
                raise ValueError("limit is too small to hold repaired text")
            piece = text[pos:end]
            new_fence = _advance_fence(fence, piece)
            closing = "\n```" if new_fence else ""

        chunks.append(prefix + piece + closing)
        fence = new_fence
        pos = end
    return chunks


def _cut_point(text: str, start: int, end: int) -> int:
    window = text[start:end]
    floor = max(len(window) // 2, 1)
    for separator in ("\n\n", "\n", ". ", "? ", "! ", "; "):
        index = window.rfind(separator, floor)
        if index > 0:
            return start + index + len(separator)
    return end


def _avoid_partial_fence(text: str, start: int, end: int) -> int:
    """Never cut inside a fence marker line; back off to its start instead."""

    line_start = text.rfind("\n", start, end) + 1
    if line_start <= start:
        return end
    fragment = text[line_start:end]
    if fragment.lstrip().startswith("`"):
        return line_start
    return end


def _advance_fence(fence: str | None, piece: str) -> str | None:
    for match in _FENCE_RE.finditer(piece):
        fence = None if fence else match.group(0).strip()
    return fence


# -- status decoration ------------------------------------------------------


def decorate(text: str, marker: str) -> str:
    return f"{text}\n\n{marker}" if text else marker


def strip_status(text: str) -> str:
    """The answer as the user would read it, without the trailing marker."""

    if text == PLACEHOLDER:
        return ""
    for marker in (SUCCESS_MARKER, WARNING_MARKER):
        index = text.rfind(f"\n\n{marker}")
        if index >= 0:
            return text[:index]
        if text.startswith(marker):
            return text[len(marker) :].lstrip()
    return text


# -- renderer ---------------------------------------------------------------


class Renderer:
    """One streamed reply: post the placeholder, update it, split when long."""

    def __init__(
        self,
        client: SlackReplyClient,
        *,
        channel: str,
        thread_ts: str | None,
        interval: float = 1.0,
        clock: Callable[[], datetime] = utcnow,
        limit: int = CHUNK_CHARS,
        burst_chars: int = BURST_CHARS,
    ) -> None:
        self._client = client
        self._channel = channel
        self._thread_ts = thread_ts
        self._interval = interval
        self._clock = clock
        self._limit = limit
        self._burst_chars = burst_chars

        self._state = "new"
        self._text = ""
        self._shown_chars = 0
        self._chunk_ts: list[str] = []
        self._chunk_text: dict[int, str] = {}
        self._last_update: datetime | None = None
        self._updates = 0

    @property
    def last_ts(self) -> str:
        if not self._chunk_ts:
            raise RuntimeError("renderer has not started")
        return self._chunk_ts[-1]

    @property
    def text(self) -> str:
        return self._text

    async def start(self) -> str:
        self._require_state("new")
        response = await self._client.chat_postMessage(
            channel=self._channel,
            thread_ts=self._thread_ts,
            text=PLACEHOLDER,
        )
        ts = str(response["ts"])
        self._chunk_ts = [ts]
        self._last_update = self._clock()
        self._state = "streaming"
        return ts

    async def push(self, delta: str) -> None:
        self._require_state("streaming")
        if not delta:
            return
        self._text += delta
        pending = len(self._text) - self._shown_chars
        if pending >= self._burst_chars or self._is_due():
            await self._flush()

    async def tick(self) -> None:
        self._require_state("streaming")
        if self._text and self._is_due():
            await self._flush()

    async def finish(self, *, status: str = "success") -> StreamResult:
        self._require_state("streaming")
        if not self._text.strip():
            return await self.fail("no response generated")

        marker = SUCCESS_MARKER if status == "success" else WARNING_MARKER
        chunks = split_message(self._text, self._limit - len(marker) - 2)
        await self._sync_chunks(chunks)
        final = decorate(chunks[-1], marker)
        if not await self._safe_update(self.last_ts, final):
            await self._post_fallback(final)

        self._state = "finished"
        return StreamResult(
            ts=self._chunk_ts[0],
            text=self._text,
            chunk_ts=list(self._chunk_ts),
            updates=self._updates,
        )

    async def fail(self, message: str) -> StreamResult:
        self._require_state("streaming")
        marker = f"{WARNING_MARKER} {message}".strip()
        if self._text.strip():
            final = f"{strip_status(self._text)}\n\n{marker}"
            if len(final) > self._limit:
                final = final[: self._limit - 1] + "…"
        else:
            final = marker

        if not await self._safe_update(self.last_ts, final):
            await self._post_fallback(final)

        self._state = "finished"
        return StreamResult(
            ts=self._chunk_ts[0],
            text=self._text,
            chunk_ts=list(self._chunk_ts),
            updates=self._updates,
        )

    # -- internals ---------------------------------------------------------

    async def _post_fallback(self, text: str) -> None:
        ts = await self._safe_post(text)
        if ts is not None:
            self._chunk_ts.append(ts)
            self._chunk_text[len(self._chunk_ts) - 1] = text

    def _is_due(self) -> bool:
        if self._last_update is None:
            return True
        return (self._clock() - self._last_update).total_seconds() >= self._interval

    async def _flush(self) -> None:
        chunks = split_message(self._text, self._limit)
        await self._sync_chunks(chunks)
        self._shown_chars = len(self._text)
        self._last_update = self._clock()

    async def _sync_chunks(self, chunks: Sequence[str]) -> None:
        for index, chunk in enumerate(chunks):
            if index < len(self._chunk_ts):
                if self._chunk_text.get(index) == chunk:
                    continue
                if await self._safe_update(self._chunk_ts[index], chunk):
                    self._chunk_text[index] = chunk
            else:
                ts = await self._safe_post(chunk)
                if ts is not None:
                    self._chunk_ts.append(ts)
                    self._chunk_text[len(self._chunk_ts) - 1] = chunk

    async def _safe_update(self, ts: str, text: str) -> bool:
        try:
            await self._client.chat_update(channel=self._channel, ts=ts, text=text)
        except Exception as exc:  # noqa: BLE001 - Slack failures must not kill the reply
            logger.warning("chat.update failed for %s: %s", ts, exc)
            return False
        self._updates += 1
        return True

    async def _safe_post(self, text: str) -> str | None:
        try:
            response = await self._client.chat_postMessage(
                channel=self._channel,
                thread_ts=self._thread_ts,
                text=text,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat.postMessage failed: %s", exc)
            return None
        return str(response["ts"])

    def _require_state(self, expected: str) -> None:
        if self._state != expected:
            raise RuntimeError(f"renderer is {self._state}, expected {expected}")
