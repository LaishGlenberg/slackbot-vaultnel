"""Remote MCP servers as tools (spec §5.7).

Integrations are configuration, not code: each entry in ``MCP_SERVERS`` names a
Streamable HTTP endpoint, an optional environment variable holding a bearer
token, and an optional tool allowlist. Adding Linear or Sentry later is another
entry, not another module.

This module owns the parts the SDK does not: lazy per-server connection, header
construction, tool/result mapping, and the rule that one dead integration must
not take the bot down. ``mcp`` itself is imported in exactly one place.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

from slackbot_vaultnel.config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPError(RuntimeError):
    """An MCP server could not be used. Never fatal to a reply."""


@dataclass(frozen=True)
class MCPTool:
    server: str
    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True)
class MCPResult:
    text: str
    is_error: bool = False


class MCPSession(Protocol):
    async def list_tools(self) -> Sequence[MCPTool]: ...

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> MCPResult: ...


class MCPClient(Protocol):
    async def list_tools(self) -> Sequence[MCPTool]: ...

    async def call_tool(
        self, server: str, tool: str, arguments: Mapping[str, Any]
    ) -> MCPResult: ...

    async def aclose(self) -> None: ...


SessionFactory = Callable[[str, MCPServerConfig, Mapping[str, str]], Any]


def _headers_for(name: str, config: MCPServerConfig, environ: Mapping[str, str]) -> dict[str, str]:
    if config.token_env is None:
        return {}
    token = environ.get(config.token_env)
    if not token:
        raise MCPError(f"MCP server {name!r} needs {config.token_env} to be set; it was not found")
    return {"Authorization": f"Bearer {token}"}


class HttpMCPClient:
    """One lazy session per configured server, opened on first use."""

    def __init__(
        self,
        servers: Mapping[str, MCPServerConfig],
        *,
        timeout: float = 30.0,
        environ: Mapping[str, str] | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._servers = dict(servers)
        self._environ = os.environ if environ is None else environ
        self._factory: SessionFactory = session_factory or partial(
            _sdk_session, timeout_seconds=timeout
        )
        self._sessions: dict[str, tuple[AsyncExitStack, MCPSession]] = {}

    # -- public API --------------------------------------------------------

    async def list_tools(self) -> Sequence[MCPTool]:
        """Every configured tool, or a partial list when a server is down."""

        if not self._servers:
            return []

        found: list[MCPTool] = []
        failures: list[str] = []
        for name, config in self._servers.items():
            try:
                session = await self._open(name)
                tools = list(await session.list_tools())
            except Exception as exc:  # noqa: BLE001 - one dead server is not fatal
                logger.warning("MCP server %s unavailable: %s", name, exc)
                failures.append(f"{name}: {exc}")
                continue
            allowed = set(config.tools)
            if allowed:
                tools = [it for it in tools if it.name in allowed]
            found.extend(tools)

        if failures and len(failures) == len(self._servers):
            raise MCPError("no MCP server could be reached (" + "; ".join(failures) + ")")
        return found

    async def call_tool(self, server: str, tool: str, arguments: Mapping[str, Any]) -> MCPResult:
        session = await self._open(server)
        try:
            return await session.call_tool(tool, dict(arguments))
        except MCPError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the model, not raised
            raise MCPError(f"MCP tool {server}/{tool} failed: {exc}") from exc

    async def aclose(self) -> None:
        for _name, (stack, _session) in list(self._sessions.items()):
            with suppress(Exception):
                await stack.aclose()
            self._sessions.pop(_name, None)

    # -- internals ---------------------------------------------------------

    async def _open(self, name: str) -> MCPSession:
        existing = self._sessions.get(name)
        if existing is not None:
            return existing[1]

        config = self._servers.get(name)
        if config is None:
            raise MCPError(f"no MCP server named {name!r}; check MCP_SERVERS")

        headers = _headers_for(name, config, self._environ)
        stack = AsyncExitStack()
        try:
            session = await stack.enter_async_context(self._factory(name, config, headers))
        except Exception as exc:  # noqa: BLE001 - surfaced as MCPError below
            with suppress(Exception):
                await stack.aclose()
            raise MCPError(f"MCP server {name!r} could not be reached: {exc}") from exc

        self._sessions[name] = (stack, session)
        return session


# -- mcp SDK adapter ----------------------------------------------------------
#
# Everything below is the only code that knows about the ``mcp`` package. The
# mapping helpers are pure so they can be tested against real ``mcp.types``
# without a server.


def tool_from_sdk(server: str, tool: object) -> MCPTool:
    return MCPTool(
        server=server,
        name=str(getattr(tool, "name", "")),
        description=str(getattr(tool, "description", None) or ""),
        input_schema=dict(getattr(tool, "input_schema", None) or {}),
    )


def result_text(result: object) -> MCPResult:
    """Flatten a ``CallToolResult`` into text the model can read."""

    blocks = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    if not parts:
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            parts.append(json.dumps(structured, ensure_ascii=False))
    return MCPResult(
        text="\n".join(parts),
        is_error=bool(getattr(result, "is_error", False)),
    )


class _SdkSession:
    """Thin adapter so the client never imports ``mcp`` directly."""

    def __init__(self, session: Any, server: str) -> None:
        self._session = session
        self._server = server

    async def list_tools(self) -> Sequence[MCPTool]:
        result = await self._session.list_tools()
        return [tool_from_sdk(self._server, tool) for tool in result.tools]

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> MCPResult:
        return result_text(await self._session.call_tool(name, dict(arguments)))


@asynccontextmanager
async def _sdk_session(
    name: str,
    config: MCPServerConfig,
    headers: Mapping[str, str],
    *,
    timeout_seconds: float = 30.0,
) -> AsyncIterator[MCPSession]:
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with (
        httpx2.AsyncClient(headers=dict(headers), timeout=timeout_seconds) as http,
        streamable_http_client(config.url, http_client=http) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield _SdkSession(session, name)
