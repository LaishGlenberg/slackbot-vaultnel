"""Bolt wiring: the only module that knows Socket Mode exists.

Event and command listeners are acknowledged immediately and the real work
runs as a Bolt lazy listener. Slack retries an event that is not acknowledged
within three seconds, and a streamed answer routinely takes longer than that;
acking first is what keeps one question from becoming several.

``__main__.py`` owns the lifecycle: authenticate, build the app, register the
handlers, start the socket, close cleanly.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from slack_bolt.async_app import AsyncApp

from slackbot_vaultnel.config import Settings
from slackbot_vaultnel.slack.handlers import Handlers


def create_app(settings: Settings, *, logger: logging.Logger | None = None) -> AsyncApp:
    signing_secret = settings.slack_signing_secret.get_secret_value() or None
    return AsyncApp(
        token=settings.slack_bot_token.get_secret_value(),
        signing_secret=signing_secret,
        logger=logger,
    )


def register_handlers(app: AsyncApp, handlers: Handlers) -> None:
    async def acknowledge(ack=None, **_kwargs: object) -> None:
        if ack is not None:
            await ack()

    async def on_mention(body: Mapping[str, object], event: Mapping[str, object]) -> None:
        await handlers.handle_event(event, event_id=_event_id(body))

    async def on_message(body: Mapping[str, object], event: Mapping[str, object]) -> None:
        await handlers.handle_event(event, event_id=_event_id(body))

    async def on_ask(command: Mapping[str, object]) -> None:
        await handlers.handle_ask(command)

    app.event("app_mention")(ack=acknowledge, lazy=[on_mention])
    app.event("message")(ack=acknowledge, lazy=[on_message])
    app.command("/ask")(ack=acknowledge, lazy=[on_ask])


def _event_id(body: Mapping[str, object]) -> str | None:
    value = body.get("event_id")
    return value if isinstance(value, str) else None
