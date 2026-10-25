"""Entrypoint: wire real dependencies and hold the Socket Mode connection."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from typing import Any

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from slackbot_vaultnel.config import ConfigError, Settings, load_settings
from slackbot_vaultnel.history.store import SqliteStore, utcnow
from slackbot_vaultnel.integrations.mcp import (
    MCP_CONFIG_FILENAME,
    HttpMCPClient,
    MCPError,
    load_mcp_config,
)
from slackbot_vaultnel.integrations.tools import build_tools
from slackbot_vaultnel.llm.agent import Tooling
from slackbot_vaultnel.llm.client import OpenAICompatClient
from slackbot_vaultnel.memory.writer import MemoryWriter
from slackbot_vaultnel.slack.app import create_app, register_handlers
from slackbot_vaultnel.slack.handlers import Handlers

logger = logging.getLogger("slackbot_vaultnel")

#: Socket Mode has no inbound health check, so the log line is the liveness signal.
HEARTBEAT_SECONDS = 300


async def run() -> int:
    settings = load_settings()
    _configure_logging(settings)
    logger.info("starting Vaultnel %s", settings.redacted_summary())

    settings.ensure_db_dir()
    store = SqliteStore(settings.db_path, clock=utcnow)
    await store.open()
    llm = _build_llm(settings)
    writer = MemoryWriter(store=store, llm=llm, settings=settings, clock=utcnow)
    await writer.start()
    servers = load_mcp_config(settings.db_path.parent / MCP_CONFIG_FILENAME)
    mcp = HttpMCPClient(servers, timeout=settings.mcp_timeout_seconds)
    try:
        app = create_app(settings)
        auth = await app.client.auth_test()
        bot_user_id = _string(auth.get("user_id"))
        logger.info("authenticated as %s (%s)", auth.get("user"), bot_user_id)

        tooling = await _discover_tools(mcp, servers)
        handlers = Handlers(
            client=app.client,
            store=store,
            llm=llm,
            settings=settings,
            bot_user_id=bot_user_id,
            clock=utcnow,
            llm_semaphore=asyncio.Semaphore(settings.max_concurrent_llm),
            writer=writer,
            tooling=tooling,
        )
        register_handlers(app, handlers)

        socket = AsyncSocketModeHandler(app, settings.slack_app_token.get_secret_value())
        logger.info("connecting to Slack over Socket Mode")
        heartbeat = asyncio.create_task(_heartbeat(), name="heartbeat")
        try:
            await _serve(socket)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
    finally:
        await mcp.aclose()
        await writer.stop()
        await store.close()
    return 0


async def _discover_tools(mcp: HttpMCPClient, servers: dict[str, Any]) -> Tooling | None:
    """One discovery pass at startup; a dead integration never blocks the bot."""

    if not servers:
        return None
    try:
        discovered = await mcp.list_tools()
    except MCPError as exc:
        logger.warning("MCP integrations unavailable, continuing without tools: %s", exc)
        return None
    if not discovered:
        logger.info("MCP configured but no tools matched the allowlists")
        return None
    tools = build_tools(discovered)
    logger.info("MCP tools available: %s", ", ".join(tool.name for tool in tools))
    return Tooling(tools=tools, call=mcp.call_tool)


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


async def _heartbeat(interval: float = HEARTBEAT_SECONDS) -> None:
    while True:
        await asyncio.sleep(interval)
        logger.info("heartbeat: process alive")


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
