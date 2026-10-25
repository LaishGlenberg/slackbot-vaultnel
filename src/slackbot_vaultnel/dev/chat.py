"""Chat with Vaultnel from the terminal, with Slack faked out.

This runs the same ``Handlers``, SQLite store, LLM client and MCP tools as the
deployed bot. Only Slack is replaced, so a tool call that works here works in
production.

    uv run python -m slackbot_vaultnel.dev.chat
    uv run python -m slackbot_vaultnel.dev.chat "list open PRs in owner/repo"

Slack tokens are not needed. The LLM key comes from ``.env``; integrations come
from ``mcp.json`` next to the database, exactly like the bot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

from slackbot_vaultnel.config import (
    ConfigError,
    Settings,
    load_environment,
    load_settings,
)
from slackbot_vaultnel.dev.slack import FakeSlack
from slackbot_vaultnel.history.store import SqliteStore, utcnow
from slackbot_vaultnel.integrations.discovery import discover_tooling
from slackbot_vaultnel.integrations.mcp import (
    MCP_CONFIG_FILENAME,
    HttpMCPClient,
    MCPResult,
    MCPServerConfig,
    load_mcp_config,
)
from slackbot_vaultnel.integrations.tools import FunctionTool
from slackbot_vaultnel.llm.agent import Tooling
from slackbot_vaultnel.llm.client import build_llm
from slackbot_vaultnel.memory.writer import DirectTurnWriter
from slackbot_vaultnel.slack.handlers import Handlers

DEV_CHANNEL = "C_DEV"
DEV_USER = "U_DEV"
DEV_BOT = "U_BOT"
DEV_THREAD = "dev-thread"


def build_event(text: str, *, ts: str, thread_ts: str = DEV_THREAD) -> dict[str, object]:
    """A mention event shaped like the one Socket Mode would deliver."""

    return {
        "type": "app_mention",
        "channel": DEV_CHANNEL,
        "ts": ts,
        "thread_ts": thread_ts,
        "text": f"<@{DEV_BOT}> {text}",
        "user": DEV_USER,
    }


async def run(prompt: str | None, *, db: Path | None = None) -> int:
    # Slack is faked, so the two Slack tokens must not gate a local run.
    overrides: dict[str, object] = {
        "slack_bot_token": "xoxb-dev",
        "slack_app_token": "xapp-dev",
    }
    if db is not None:
        overrides["db_path"] = db
    settings = load_settings(**overrides)
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    # The transport's request/session chatter buries the answer and tool calls.
    for noisy in ("httpx2", "mcp.client.streamable_http"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings.ensure_db_dir()

    store = SqliteStore(settings.db_path, clock=utcnow)
    await store.open()
    servers = load_mcp_config(settings.db_path.parent / MCP_CONFIG_FILENAME)
    mcp = HttpMCPClient(
        servers,
        timeout=settings.mcp_timeout_seconds,
        environ=load_environment(),
    )
    slack = FakeSlack()
    try:
        print(f"DB: {settings.db_path}")
        tooling = await _load_tooling(settings, mcp, servers)
        handlers = Handlers(
            client=slack,
            store=store,
            llm=build_llm(settings),
            settings=settings,
            bot_user_id=DEV_BOT,
            clock=utcnow,
            writer=DirectTurnWriter(store),
            tooling=tooling,
        )
        if prompt:
            await _say(handlers, slack, prompt, counter=1)
        else:
            await _repl(handlers, slack)
    finally:
        await mcp.aclose()
        await store.close()
    return 0


async def _load_tooling(
    settings: Settings,
    mcp: HttpMCPClient,
    servers: Mapping[str, MCPServerConfig],
) -> Tooling | None:
    path = settings.db_path.parent / MCP_CONFIG_FILENAME
    if not servers:
        print(f"MCP: no {path}; running without tools.")
        print(f"     cp mcp.example.json {path}")
        return None

    print(f"MCP: {', '.join(servers)}")
    discovered = await discover_tooling(mcp, servers)
    if discovered is None:
        print("MCP: no usable tools; see the warnings above.")
        return None

    print("MCP tools: " + ", ".join(tool.tool for tool in discovered.tools))
    return Tooling(tools=discovered.tools, call=_instrumented(mcp))


def _instrumented(mcp: HttpMCPClient):
    async def call(tool: FunctionTool, arguments: Mapping[str, object]) -> MCPResult:
        print(f"  tool -> {tool.server}/{tool.tool} {json.dumps(arguments, ensure_ascii=False)}")
        try:
            result = await mcp.call_tool(tool.server, tool.tool, arguments)
        except Exception as exc:  # noqa: BLE001 - re-raised for the agent to report
            print(f"  tool <- failed: {exc}")
            raise
        first = next((line for line in result.text.splitlines() if line.strip()), "")
        status = "error" if result.is_error else "ok"
        print(f"  tool <- {status} ({len(result.text)} chars): {first[:160]}")
        return result

    return call


async def _repl(handlers: Handlers, slack: FakeSlack) -> None:
    print("Vaultnel local chat (Slack is faked). Type 'exit' to quit.")
    counter = 0
    while True:
        try:
            line = await asyncio.to_thread(input, "\nyou > ")
        except EOFError:
            print()
            return
        text = line.strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            return
        counter += 1
        await _say(handlers, slack, text, counter)


async def _say(handlers: Handlers, slack: FakeSlack, prompt: str, counter: int) -> None:
    print(f"\nyou > {prompt}")
    event = build_event(prompt, ts=f"{counter}.000")
    await handlers.handle_event(event, event_id=f"dev-{counter}")
    for message in slack.take_messages():
        print(f"bot > {message}")
    for message in slack.take_ephemeral():
        print(f"bot (ephemeral) > {message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chat with Vaultnel locally; Slack is faked, LLM and MCP are real."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="one prompt (omit for a REPL)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="SQLite file, keeping a test conversation out of the bot's database",
    )
    args = parser.parse_args(argv)
    prompt = " ".join(args.prompt) if args.prompt else None
    try:
        return asyncio.run(run(prompt, db=args.db))
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
