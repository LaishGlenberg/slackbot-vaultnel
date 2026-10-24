"""MCP client: discovery, routing, allowlists and failure isolation.

The SDK's own HTTP transport is not exercised here (same caveat the decisions
log records for httpx2/respx). What is tested is everything this repo owns:
lazy connection, header construction, tool mapping, result mapping and the
rule that one dead integration must not take the bot down.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest

from slackbot_vaultnel.integrations.mcp import (
    HttpMCPClient,
    MCPError,
    MCPResult,
    MCPServerConfig,
    MCPTool,
    result_text,
    tool_from_sdk,
)

GITHUB = "https://api.githubcopilot.com/mcp/"


class FakeSession:
    def __init__(self, server: str, tools: list[str], *, fail: bool = False) -> None:
        self.server = server
        self.tool_names = tools
        self.fail = fail
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    async def list_tools(self) -> list[MCPTool]:
        if self.fail:
            raise RuntimeError(f"{self.server} refused")
        return [
            MCPTool(
                server=self.server,
                name=name,
                description=f"{name} on {self.server}",
                input_schema={"type": "object", "properties": {}},
            )
            for name in self.tool_names
        ]

    async def call_tool(self, name: str, arguments: dict) -> MCPResult:
        self.calls.append((name, dict(arguments)))
        return MCPResult(text=f"{self.server}:{name}:{json.dumps(arguments)}")


class FakeSessionFactory:
    """Records the servers it was asked to open, like the real transport would."""

    def __init__(self, specs: dict[str, tuple[list[str], bool]]) -> None:
        self.specs = specs
        self.opened: list[str] = []
        self.sessions: list[FakeSession] = []
        self.headers: list[dict[str, str]] = []

    def __call__(self, name: str, config: MCPServerConfig, headers: dict[str, str]):
        self.opened.append(name)
        self.headers.append(dict(headers))
        tools, fail = self.specs.get(name, ([], False))
        session = FakeSession(name, tools, fail=fail)
        self.sessions.append(session)

        @asynccontextmanager
        async def _ctx():
            try:
                yield session
            finally:
                session.closed = True

        return _ctx()


def build(factory: FakeSessionFactory, **servers: MCPServerConfig) -> HttpMCPClient:
    return HttpMCPClient(servers, session_factory=factory, environ={"GITHUB_TOKEN": "tok-123"})


def server(**kwargs) -> MCPServerConfig:
    return MCPServerConfig(url=GITHUB, **kwargs)


async def test_no_servers_means_no_tools_and_no_connections():
    factory = FakeSessionFactory({})

    client = build(factory)

    assert await client.list_tools() == []
    assert factory.opened == []


async def test_lists_tools_from_every_configured_server():
    factory = FakeSessionFactory({"github": (["get_pull_request", "list_issues"], False)})
    client = build(factory, github=server())

    tools = await client.list_tools()

    assert [(t.server, t.name) for t in tools] == [
        ("github", "get_pull_request"),
        ("github", "list_issues"),
    ]
    assert factory.opened == ["github"]


async def test_sessions_are_opened_lazily_once_per_server():
    factory = FakeSessionFactory({"github": (["a"], False)})
    client = build(factory, github=server())

    await client.list_tools()
    await client.list_tools()

    assert factory.opened == ["github"]


async def test_token_env_becomes_a_bearer_header():
    factory = FakeSessionFactory({"github": (["a"], False)})
    client = build(factory, github=server(token_env="GITHUB_TOKEN"))

    await client.list_tools()

    assert factory.headers == [{"Authorization": "Bearer tok-123"}]


async def test_missing_token_env_fails_without_leaking_anything():
    factory = FakeSessionFactory({"github": (["a"], False)})
    client = HttpMCPClient({"github": server(token_env="MISSING_TOKEN")}, session_factory=factory)

    with pytest.raises(MCPError) as exc:
        await client.list_tools()

    assert "MISSING_TOKEN" in str(exc.value)
    assert factory.opened == []


async def test_per_server_allowlist_filters_tools():
    factory = FakeSessionFactory({"github": (["get_pull_request", "delete_repo"], False)})
    client = build(factory, github=server(tools=("get_pull_request",)))

    assert [t.name for t in await client.list_tools()] == ["get_pull_request"]


async def test_one_broken_server_does_not_hide_a_working_one():
    factory = FakeSessionFactory(
        {"github": (["get_pull_request"], False), "jira": (["search"], True)}
    )
    client = build(
        factory, github=server(), jira=MCPServerConfig(url="https://jira.example.com/mcp")
    )

    tools = await client.list_tools()

    assert [t.name for t in tools] == ["get_pull_request"]


async def test_every_server_failing_is_an_error():
    factory = FakeSessionFactory({"github": ([], True)})
    client = build(factory, github=server())

    with pytest.raises(MCPError) as exc:
        await client.list_tools()

    assert "github" in str(exc.value)


async def test_call_tool_routes_to_the_named_server():
    factory = FakeSessionFactory({"github": (["get_pull_request"], False)})
    client = build(factory, github=server())
    await client.list_tools()

    result = await client.call_tool("github", "get_pull_request", {"number": 41})

    assert result.text == 'github:get_pull_request:{"number": 41}'
    assert factory.sessions[0].calls == [("get_pull_request", {"number": 41})]


async def test_call_tool_on_an_unknown_server_is_an_error():
    client = build(FakeSessionFactory({}))

    with pytest.raises(MCPError):
        await client.call_tool("nope", "x", {})


async def test_aclose_closes_every_session_and_is_idempotent():
    factory = FakeSessionFactory({"github": (["a"], False)})
    client = build(factory, github=server())
    await client.list_tools()

    await client.aclose()
    await client.aclose()
    assert [s.closed for s in factory.sessions] == [True]

    # Closing is not terminal: the next use reconnects on demand.
    await client.list_tools()
    assert factory.opened == ["github", "github"]


# -- SDK mapping (pure functions over mcp.types) -------------------------------


def test_tool_from_sdk_maps_name_description_and_schema():
    from mcp.types import Tool

    sdk_tool = Tool(
        name="get_pull_request",
        description="Fetch one pull request",
        input_schema={"type": "object", "properties": {"number": {"type": "integer"}}},
    )

    mapped = tool_from_sdk("github", sdk_tool)

    assert mapped == MCPTool(
        server="github",
        name="get_pull_request",
        description="Fetch one pull request",
        input_schema={"type": "object", "properties": {"number": {"type": "integer"}}},
    )


def test_tool_from_sdk_tolerates_a_missing_description():
    from mcp.types import Tool

    mapped = tool_from_sdk("github", Tool(name="x", input_schema={}))

    assert mapped.description == ""


def test_result_text_joins_text_blocks():
    from mcp.types import CallToolResult, TextContent

    result = CallToolResult(
        content=[
            TextContent(type="text", text="line one"),
            TextContent(type="text", text="line two"),
        ]
    )

    assert result_text(result).text == "line one\nline two"


def test_result_text_marks_tool_errors():
    from mcp.types import CallToolResult, TextContent

    result = CallToolResult(content=[TextContent(type="text", text="403 Forbidden")], is_error=True)

    mapped = result_text(result)

    assert mapped.is_error is True
    assert "403 Forbidden" in mapped.text


def test_result_text_falls_back_to_structured_content():
    from mcp.types import CallToolResult

    result = CallToolResult(content=[], structured_content={"state": "open"})

    assert json.loads(result_text(result).text) == {"state": "open"}
