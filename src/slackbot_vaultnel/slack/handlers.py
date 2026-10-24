"""Slack event handling: filtering, threading, memory and streaming (spec §4).

Handlers receive every dependency as an argument, so this module never imports
Bolt, never reads the environment and never sleeps on its own. ``slack/app.py``
is the only place that knows about the Socket Mode adapter.

The flow for one message is deliberately linear:

1. filter (bots, subtypes, our own messages) and dedupe by event id
2. strip our mention, resolve the speaker, acknowledge with :eyes:
3. take the per-thread lock so a busy thread answers in order
4. assemble the prompt and persist the user turn
5. stream the reply into a placeholder, then swap the reaction
6. persist the assistant turn

Persistence failures are logged and never break a reply. The background
writer replaces the direct ``append_message`` calls in a later slice; the
handler contract stays the same.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Protocol

from slackbot_vaultnel.config import Settings
from slackbot_vaultnel.history.store import Store, Thread, Turn, utcnow
from slackbot_vaultnel.llm.client import AuthError, LLMError, RateLimited
from slackbot_vaultnel.memory.context import PromptBudget, PromptRequest, build
from slackbot_vaultnel.memory.facts import facts_for_prompt
from slackbot_vaultnel.memory.recall import recall_for_prompt
from slackbot_vaultnel.memory.tokens import count_tokens
from slackbot_vaultnel.memory.writer import DirectTurnWriter, TurnWriter
from slackbot_vaultnel.slack.render import Renderer, SlackReplyClient

logger = logging.getLogger(__name__)

MENTION_HINT = "Hi! Ask me a question after the mention, e.g. `@vaultnel how do I …`."
ASK_USAGE = "Usage: `/ask <question>`, `/ask reset`, or `/ask forget me`."
GENERIC_FAILURE = "Something went wrong while I was answering; the details are in the logs."

#: Enough to cover a Slack retry storm without growing without bound.
SEEN_EVENTS_MAX = 1024


class SlackAPI(SlackReplyClient, Protocol):
    async def reactions_add(self, *, channel: str, timestamp: str, name: str) -> object: ...

    async def reactions_remove(self, *, channel: str, timestamp: str, name: str) -> object: ...

    async def users_info(self, *, user: str) -> Mapping[str, object]: ...

    async def chat_postEphemeral(self, *, channel: str, user: str, text: str) -> object: ...


RendererFactory = Callable[..., Renderer]


class Handlers:
    def __init__(
        self,
        *,
        client: SlackAPI,
        store: Store,
        llm,
        settings: Settings,
        bot_user_id: str | None = None,
        clock: Callable[[], datetime] = utcnow,
        renderer_factory: RendererFactory = Renderer,
        llm_semaphore: asyncio.Semaphore | None = None,
        writer: TurnWriter | None = None,
    ) -> None:
        self.client = client
        self.store = store
        self.writer = writer or DirectTurnWriter(store)
        self.llm = llm
        self.settings = settings
        self.bot_user_id = bot_user_id
        self.clock = clock
        self.renderer_factory = renderer_factory
        self.llm_semaphore = llm_semaphore

        self._budget = PromptBudget.from_settings(settings)
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._names: dict[str, str | None] = {}

    # -- event entrypoints -------------------------------------------------

    async def handle_event(
        self,
        event: Mapping[str, object],
        *,
        event_id: str | None = None,
    ) -> bool:
        """Dispatch one Socket Mode event. Returns False when it was ignored."""

        if event_id is not None and not self._first_sighting(event_id):
            return False

        kind = event.get("type")
        if kind == "app_mention":
            return await self._handle_mention(event)
        if kind == "message" and event.get("channel_type") == "im":
            return await self._handle_dm(event)
        return False

    async def handle_ask(self, command: Mapping[str, object]) -> None:
        """`/ask <question>`, `/ask reset`, `/ask forget me` (spec §4.4)."""

        channel = _text(command.get("channel_id"))
        user = _text(command.get("user_id"))
        if not channel or not user:
            return

        text = _text(command.get("text")).strip()
        thread_ts = _text(command.get("thread_ts")) or None
        lowered = text.lower()

        if lowered == "reset":
            memory_key = thread_ts or self._slash_key(channel)
            try:
                await self.store.archive_thread(channel, memory_key)
            except ValueError:
                await self._ephemeral(channel, user, "There is no thread memory to clear here.")
            else:
                await self._ephemeral(channel, user, "Thread memory cleared.")
            return

        if lowered == "forget me":
            try:
                # Let queued extraction finish first, so it cannot re-add what we delete.
                await self.writer.flush()
                removed = await self.store.delete_facts("user", user)
            except Exception:  # noqa: BLE001
                logger.exception("could not delete user facts for %s", user)
                await self._ephemeral(
                    channel, user, "I couldn't delete your facts; the details are in the logs."
                )
                return
            if removed:
                noun = "fact" if removed == 1 else "facts"
                await self._ephemeral(channel, user, f"Deleted {removed} {noun} about you.")
            else:
                await self._ephemeral(channel, user, "No facts about you are stored.")
            return

        if not text:
            await self._ephemeral(channel, user, ASK_USAGE)
            return

        await self._answer(
            channel=channel,
            reply_thread_ts=thread_ts,
            memory_thread_ts=thread_ts or self._slash_key(channel),
            text=text,
            user=user,
            ack_ts=None,
        )

    # -- filtering ---------------------------------------------------------

    async def _handle_mention(self, event: Mapping[str, object]) -> bool:
        if self._ignored(event):
            return False
        channel = _text(event.get("channel"))
        ts = _text(event.get("ts"))
        if not channel or not ts:
            return False

        thread_ts = _text(event.get("thread_ts")) or ts
        text = self._strip_mentions(_text(event.get("text")))
        user = _text(event.get("user"))
        if not text:
            await self._post_hint(channel, thread_ts)
            return True

        await self._answer(
            channel=channel,
            reply_thread_ts=thread_ts,
            memory_thread_ts=thread_ts,
            text=text,
            user=user,
            ack_ts=ts,
        )
        return True

    async def _handle_dm(self, event: Mapping[str, object]) -> bool:
        if self._ignored(event):
            return False
        channel = _text(event.get("channel"))
        ts = _text(event.get("ts"))
        if not channel or not ts:
            return False

        thread_ts = _text(event.get("thread_ts")) or ts
        text = self._strip_mentions(_text(event.get("text")))
        if not text:
            await self._post_hint(channel, thread_ts)
            return True

        await self._answer(
            channel=channel,
            reply_thread_ts=thread_ts,
            memory_thread_ts=thread_ts,
            text=text,
            user=_text(event.get("user")),
            ack_ts=ts,
        )
        return True

    def _ignored(self, event: Mapping[str, object]) -> bool:
        if event.get("subtype") or event.get("bot_id"):
            return True
        user = _text(event.get("user"))
        if not user or not _text(event.get("channel")):
            return True
        return self.bot_user_id is not None and user == self.bot_user_id

    def _strip_mentions(self, text: str) -> str:
        if not self.bot_user_id:
            return text.strip()
        pattern = re.compile(rf"<@{re.escape(self.bot_user_id)}(?:\|[^>]*)?>")
        return pattern.sub("", text).strip()

    # -- one answer --------------------------------------------------------

    async def _answer(
        self,
        *,
        channel: str,
        reply_thread_ts: str | None,
        memory_thread_ts: str,
        text: str,
        user: str | None,
        ack_ts: str | None,
    ) -> None:
        display_name = await self._display_name(user)
        if ack_ts is not None:
            await self._react(channel, ack_ts, "eyes", add=True)

        async with self._thread_lock(channel, memory_thread_ts):
            try:
                facts = await facts_for_prompt(self.store, channel_id=channel, user_id=user)
            except Exception:  # noqa: BLE001 - missing memory must not block a reply
                logger.exception("could not load facts for %s/%s", channel, memory_thread_ts)
                facts = []

            try:
                snippets = await recall_for_prompt(
                    self.store,
                    channel_id=channel,
                    thread_ts=memory_thread_ts,
                    text=text,
                    limit=self.settings.recall_top_k,
                )
            except Exception:  # noqa: BLE001 - recall must not block a reply
                logger.exception("could not recall snippets for %s/%s", channel, memory_thread_ts)
                snippets = []

            request = PromptRequest(
                channel_id=channel,
                thread_ts=memory_thread_ts,
                text=text,
                slack_user=user,
                display_name=display_name,
                persona=self.settings.resolved_system_prompt,
                include_speaker_names=self.settings.include_speaker_names,
                now=self.clock(),
                facts=facts,
                snippets=snippets,
            )
            try:
                prompt = await build(request, self.store, self._budget)
            except Exception:  # noqa: BLE001 - memory must not block a reply
                logger.exception("prompt assembly failed for %s/%s", channel, memory_thread_ts)
                prompt = [
                    {"role": "system", "content": self.settings.resolved_system_prompt},
                    {"role": "user", "content": text},
                ]

            thread = await self.store.get_or_create_thread(channel, memory_thread_ts)
            user_turn = Turn(
                role="user",
                content=text,
                tokens=count_tokens(text),
                slack_user=user,
                display_name=display_name,
            )

            renderer = self.renderer_factory(
                self.client,
                channel=channel,
                thread_ts=reply_thread_ts,
                interval=self.settings.stream_interval,
                clock=self.clock,
            )
            try:
                await renderer.start()
            except Exception:  # noqa: BLE001
                logger.exception("could not post a placeholder to %s/%s", channel, memory_thread_ts)
                await self._record(thread, [user_turn])
                return

            try:
                async with self._llm_slot():
                    async for delta in self._stream(prompt):
                        await renderer.push(delta)
                result = await renderer.finish()
            except LLMError as exc:
                logger.warning("model call failed for %s/%s: %s", channel, memory_thread_ts, exc)
                await renderer.fail(_friendly_error(exc))
                await self._settle_reaction(channel, ack_ts, success=False)
                await self._record(thread, [user_turn])
                return
            except Exception:  # noqa: BLE001 - one bad reply must not kill the socket
                logger.exception("unhandled reply failure for %s/%s", channel, memory_thread_ts)
                await renderer.fail(GENERIC_FAILURE)
                await self._settle_reaction(channel, ack_ts, success=False)
                await self._record(thread, [user_turn])
                return

            if result.text.strip():
                await self._settle_reaction(channel, ack_ts, success=True)
                await self._record(
                    thread,
                    [
                        user_turn,
                        Turn(
                            role="assistant",
                            content=result.text,
                            tokens=count_tokens(result.text),
                            model=self.settings.resolved_model,
                        ),
                    ],
                )
            else:
                # The renderer already marked the empty answer with :warning:.
                await self._settle_reaction(channel, ack_ts, success=False)
                await self._record(thread, [user_turn])

    def _stream(self, prompt: list[dict[str, str]]) -> AsyncIterator[str]:
        return self.llm.stream(
            prompt,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
        )

    @asynccontextmanager
    async def _llm_slot(self):
        if self.llm_semaphore is None:
            yield
        else:
            async with self.llm_semaphore:
                yield

    # -- persistence -------------------------------------------------------

    async def _record(self, thread: Thread, turns: Sequence[Turn]) -> None:
        try:
            await self.writer.record(thread, turns)
        except Exception:  # noqa: BLE001
            logger.exception("could not record turns in thread %s", thread.id)

    # -- Slack side effects ------------------------------------------------

    async def _display_name(self, user: str | None) -> str | None:
        if not user or not self.settings.include_speaker_names:
            return None
        if user in self._names:
            return self._names[user]
        name: str | None = None
        try:
            response = await self.client.users_info(user=user)
            profile = response.get("user", {})
            if isinstance(profile, Mapping):
                details = profile.get("profile", {})
                if isinstance(details, Mapping):
                    name = _optional_text(details.get("display_name"))
                name = (
                    name
                    or _optional_text(profile.get("real_name"))
                    or _optional_text(profile.get("name"))
                )
        except Exception:  # noqa: BLE001 - names are decoration
            logger.debug("could not resolve display name for %s", user, exc_info=True)
        self._names[user] = name
        return name

    async def _post_hint(self, channel: str, thread_ts: str) -> None:
        try:
            await self.client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=MENTION_HINT,
            )
        except Exception:  # noqa: BLE001
            logger.exception("could not post the usage hint to %s/%s", channel, thread_ts)

    async def _ephemeral(self, channel: str, user: str, text: str) -> None:
        try:
            await self.client.chat_postEphemeral(channel=channel, user=user, text=text)
        except Exception:  # noqa: BLE001
            logger.exception("could not send an ephemeral message to %s", user)

    async def _react(self, channel: str, ts: str, name: str, *, add: bool) -> None:
        try:
            if add:
                await self.client.reactions_add(channel=channel, timestamp=ts, name=name)
            else:
                await self.client.reactions_remove(channel=channel, timestamp=ts, name=name)
        except Exception:  # noqa: BLE001 - reactions are best-effort
            logger.debug("reaction %s on %s failed", name, ts, exc_info=True)

    async def _settle_reaction(self, channel: str, ts: str | None, *, success: bool) -> None:
        if ts is None:
            return
        await self._react(channel, ts, "eyes", add=False)
        await self._react(
            channel,
            ts,
            "white_check_mark" if success else "warning",
            add=True,
        )

    # -- small helpers -----------------------------------------------------

    def _thread_lock(self, channel: str, thread_ts: str) -> asyncio.Lock:
        key = (channel, thread_ts)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def _first_sighting(self, event_id: str) -> bool:
        if event_id in self._seen:
            return False
        self._seen[event_id] = None
        while len(self._seen) > SEEN_EVENTS_MAX:
            self._seen.popitem(last=False)
        return True

    @staticmethod
    def _slash_key(channel: str) -> str:
        return f"slash-{channel}"


def _friendly_error(exc: LLMError) -> str:
    if isinstance(exc, AuthError):
        return "I can't reach the model: check that LLM_API_KEY is valid."
    if isinstance(exc, RateLimited):
        return "The model provider is rate limiting us; try again in a moment."
    return f"The model call failed: {exc}"


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
