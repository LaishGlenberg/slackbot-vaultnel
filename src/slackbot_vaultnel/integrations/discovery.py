"""Startup tool discovery, shared by the bot and the local terminal harness.

Discovery happens once, at startup. A server that fails is logged and skipped;
a missing integration must never stop a reply before the user has said
anything.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from slackbot_vaultnel.integrations.mcp import (
    HttpMCPClient,
    MCPError,
    MCPResult,
    MCPServerConfig,
)
from slackbot_vaultnel.integrations.tools import FunctionTool, build_tools
from slackbot_vaultnel.llm.agent import Tooling

logger = logging.getLogger(__name__)


async def discover_tooling(
    mcp: HttpMCPClient,
    servers: Mapping[str, MCPServerConfig],
) -> Tooling | None:
    """The model-facing catalogue, or ``None`` when no tools are usable."""

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

    async def call(tool: FunctionTool, arguments: Mapping[str, object]) -> MCPResult:
        """Adapt the client's (server, tool, arguments) shape to the agent's."""

        return await mcp.call_tool(tool.server, tool.tool, arguments)

    return Tooling(tools=tools, call=call)
