"""``mcp.json`` loading: the integration config lives in a file, not the env.

Secrets are never in this file. A server names the environment variable that
holds its token (``token_env``), and the name is validated to look like an env
var so a pasted token fails loudly instead of half-working.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from slackbot_vaultnel.config import ConfigError
from slackbot_vaultnel.integrations.mcp import MCPServerConfig, load_mcp_config

VALID = {
    "mcpServers": {
        "github": {
            "url": "https://api.githubcopilot.com/mcp/",
            "token_env": "GITHUB_TOKEN",
            "tools": ["get_pull_request", "list_pull_requests"],
        }
    }
}


def write(tmp_path: Path, payload) -> Path:
    path = tmp_path / "mcp.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


def test_a_missing_file_means_no_integrations(tmp_path):
    assert load_mcp_config(tmp_path / "mcp.json") == {}


def test_servers_are_parsed_with_token_env_and_allowlist(tmp_path):
    servers = load_mcp_config(write(tmp_path, VALID))

    server = servers["github"]
    assert server.url == "https://api.githubcopilot.com/mcp/"
    assert server.token_env == "GITHUB_TOKEN"
    assert server.tools == ("get_pull_request", "list_pull_requests")


def test_a_server_may_have_no_token_and_no_allowlist(tmp_path):
    path = write(tmp_path, {"mcpServers": {"local": {"url": "http://127.0.0.1:8765/mcp"}}})

    server = load_mcp_config(path)["local"]

    assert server.token_env is None
    assert server.tools == ()


def test_the_standard_mcpServers_wrapper_is_required(tmp_path):
    path = write(tmp_path, {"github": {"url": "https://api.example.com/mcp"}})

    with pytest.raises(ConfigError) as exc:
        load_mcp_config(path)

    assert "mcpServers" in str(exc.value)
    assert str(path) in str(exc.value)


def test_invalid_json_names_the_file(tmp_path):
    path = write(tmp_path, "{nope")

    with pytest.raises(ConfigError) as exc:
        load_mcp_config(path)

    assert str(path) in str(exc.value)


def test_mcpServers_must_be_an_object(tmp_path):
    with pytest.raises(ConfigError):
        load_mcp_config(write(tmp_path, {"mcpServers": []}))


@pytest.mark.parametrize("name", ["git hub", "git/hub", "", "x" * 40])
def test_server_names_must_be_tool_safe(tmp_path, name):
    path = write(tmp_path, {"mcpServers": {name: {"url": "https://a.example.com/mcp"}}})

    with pytest.raises(ConfigError) as exc:
        load_mcp_config(path)

    assert "server name" in str(exc.value)


def test_unknown_server_fields_are_rejected(tmp_path):
    path = write(
        tmp_path,
        {"mcpServers": {"github": {"url": "https://a.example.com/mcp", "token": "leak"}}},
    )

    with pytest.raises(ConfigError):
        load_mcp_config(path)


def test_plain_http_is_rejected_outside_localhost(tmp_path):
    path = write(tmp_path, {"mcpServers": {"github": {"url": "http://a.example.com/mcp"}}})

    with pytest.raises(ConfigError) as exc:
        load_mcp_config(path)

    assert "https" in str(exc.value)


def test_a_token_pasted_into_token_env_is_rejected(tmp_path):
    path = write(
        tmp_path,
        {"mcpServers": {"github": {"url": "https://a.example.com/mcp", "token_env": "ghp_abc123"}}},
    )

    with pytest.raises(ConfigError) as exc:
        load_mcp_config(path)

    assert "token_env" in str(exc.value)


def test_blank_allowlist_entries_are_dropped(tmp_path):
    path = write(
        tmp_path,
        {"mcpServers": {"github": {"url": "https://a.example.com/mcp", "tools": ["x", " "]}}},
    )

    assert load_mcp_config(path)["github"].tools == ("x",)


def test_the_model_is_frozen():
    server = MCPServerConfig(url="https://a.example.com/mcp")

    with pytest.raises(ValidationError):
        server.url = "https://b.example.com/mcp"  # type: ignore[misc]
