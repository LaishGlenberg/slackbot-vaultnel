"""Listener wiring: every subscribed event must have a handler.

Slack sends a subscribed event even when the bot does not care about it, and
Bolt logs ``Unhandled request`` for each one. That noise is what these tests
protect against: the manifest and ``register_handlers`` have to agree, and the
ack-only listeners must not touch the message pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from slackbot_vaultnel.slack.app import register_handlers

MANIFEST = Path(__file__).resolve().parent.parent / "slack" / "manifest.yaml"


class RecordingApp:
    """Stands in for ``AsyncApp``: records what gets registered."""

    def __init__(self) -> None:
        self.events: dict[str, dict[str, Any]] = {}
        self.commands: dict[str, dict[str, Any]] = {}

    def event(self, name: str) -> Callable[..., None]:
        def register(**kwargs: Any) -> None:
            self.events[name] = kwargs

        return register

    def command(self, name: str) -> Callable[..., None]:
        def register(**kwargs: Any) -> None:
            self.commands[name] = kwargs

        return register


class RecordingHandlers:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def handle_event(self, event: Mapping[str, object], *, event_id: str | None) -> bool:
        self.calls.append("event")
        return True

    async def handle_ask(self, command: Mapping[str, object]) -> None:
        self.calls.append("ask")


def _manifest_bot_events() -> list[str]:
    manifest = yaml.safe_load(MANIFEST.read_text())
    return list(manifest["settings"]["event_subscriptions"]["bot_events"])


def test_every_manifest_event_has_a_listener() -> None:
    app = RecordingApp()
    register_handlers(app, RecordingHandlers())  # type: ignore[arg-type]

    # message.im and message.channels both arrive as "message"; any other bot
    # event must be registered under its own name.
    subscribed = {name for name in _manifest_bot_events() if not name.startswith("message.")}
    assert subscribed <= set(app.events)


def test_manifest_has_no_listenerless_events() -> None:
    app = RecordingApp()
    register_handlers(app, RecordingHandlers())  # type: ignore[arg-type]

    assert set(app.events) - {"message"} <= set(_manifest_bot_events())
    assert set(app.commands) == {"/ask"}


@pytest.mark.asyncio
async def test_app_home_opened_is_ack_only() -> None:
    app = RecordingApp()
    handlers = RecordingHandlers()
    register_handlers(app, handlers)  # type: ignore[arg-type]

    # Bolt injects only the arguments a listener declares; this one declares none.
    (listener,) = app.events["app_home_opened"]["lazy"]
    await listener()

    assert handlers.calls == []
