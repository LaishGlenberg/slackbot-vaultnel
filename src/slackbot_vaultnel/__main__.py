"""Entrypoint: wire real dependencies and hold the Socket Mode connection."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from slackbot_vaultnel.config import ConfigError, Settings, load_settings
from slackbot_vaultnel.history.store import SqliteStore, utcnow
from slackbot_vaultnel.llm.client import OpenAICompatClient
from slackbot_vaultnel.slack.app import create_app, register_handlers
from slackbot_vaultnel.slack.handlers import Handlers

logger = logging.getLogger("slackbot_vaultnel")


async def run() -> int:
    settings = load_settings()
    _configure_logging(settings)
    logger.info("starting Vaultnel %s", settings.redacted_summary())

    settings.ensure_db_dir()
    store = SqliteStore(settings.db_path, clock=utcnow)
    await store.open()
    try:
        app = create_app(settings)
        auth = await app.client.auth_test()
        bot_user_id = _string(auth.get("user_id"))
        logger.info("authenticated as %s (%s)", auth.get("user"), bot_user_id)

        handlers = Handlers(
            client=app.client,
            store=store,
            llm=_build_llm(settings),
            settings=settings,
            bot_user_id=bot_user_id,
            clock=utcnow,
            llm_semaphore=asyncio.Semaphore(settings.max_concurrent_llm),
        )
        register_handlers(app, handlers)

        socket = AsyncSocketModeHandler(app, settings.slack_app_token.get_secret_value())
        await _serve(socket)
    finally:
        await store.close()
    return 0


def _build_llm(settings: Settings) -> OpenAICompatClient:
    return OpenAICompatClient(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.resolved_base_url,
        model=settings.resolved_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        reasoning_effort=settings.reasoning_effort,
    )


async def _serve(socket: AsyncSocketModeHandler) -> None:
    """Run the socket until it fails or a shutdown signal arrives."""

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    running = asyncio.create_task(socket.start_async(), name="slack-socket")
    signalled = asyncio.create_task(stop.wait(), name="shutdown")
    try:
        done, _ = await asyncio.wait({running, signalled}, return_when=asyncio.FIRST_COMPLETED)
        if running in done:
            running.result()
        else:
            logger.info("shutdown signal received")
    finally:
        signalled.cancel()
        await socket.close_async()
        if not running.done():
            running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running


def _configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def main() -> int:
    try:
        return asyncio.run(run())
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
