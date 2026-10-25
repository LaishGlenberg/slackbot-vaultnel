"""Startup discovery shared by the bot and the local terminal harness."""

from __future__ import annotations

from slackbot_vaultnel.integrations.discovery import discover_tooling
from slackbot_vaultnel.integrations.mcp import (
    MCPError,
    MCPResult,
    MCPServerConfig,
    MCPTool,
)

GITHUB = MCPServerConfig(url="https://api.githubcopilot.com/mcp/")


class FakeMCP:
    def __init__(self, tools: list[MCPTool] | None = None, *, error: bool = False) -> None:
        self.tools = tools or []
        self.error = error
        self.calls: list[tuple[str, str, dict]] = []

    async def list_tools(self) -> list[MCPTool]:
        if self.error:
            raise MCPError("github: 401")
        return self.tools

    async def call_tool(self, server: str, tool: str, arguments: dict) -> MCPResult:
        self.calls.append((server, tool, dict(arguments)))
        return MCPResult(text=f"{server}/{tool}")

    async def aclose(self) -> None:
        return None


def a_tool() -> MCPTool:
    return MCPTool(
        server="github",
        name="get_me",
        description="who am I",
        input_schema={"type": "object", "properties": {}},
    )


async def test_no_servers_means_no_discovery_and_no_connection():
    mcp = FakeMCP([a_tool()])

    assert await discover_tooling(mcp, {}) is None
    assert mcp.calls == []


async def test_tools_are_namespaced_and_dispatch_through_the_client():
    mcp = FakeMCP([a_tool()])

    tooling = await discover_tooling(mcp, {"github": GITHUB})

    assert tooling is not None
    assert [tool.name for tool in tooling.tools] == ["github__get_me"]
    assert await tooling.call(tooling.tools[0], {}) == MCPResult(text="github/get_me")
    assert mcp.calls == [("github", "get_me", {})]


async def test_a_dead_server_degrades_to_no_tools():
    assert await discover_tooling(FakeMCP(error=True), {"github": GITHUB}) is None


async def test_an_allowlist_that_matches_nothing_degrades_to_no_tools():
    assert await discover_tooling(FakeMCP([]), {"github": GITHUB}) is None
