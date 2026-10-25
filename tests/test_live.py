"""Live smoke tests against the configured provider (spec §9).

Deselected by default. Run with ``uv run pytest -m slow`` and ``LLM_API_KEY``
in ``.env``. Two round trips only: plain text, and one tool call fed back.
"""

from __future__ import annotations

import json

import pytest

from slackbot_vaultnel.config import load_settings
from slackbot_vaultnel.llm.client import OpenAICompatClient, ToolCall

pytestmark = pytest.mark.slow

LOOKUP = {
    "type": "function",
    "function": {
        "name": "lookup_word",
        "description": "Look up the secret word for a key. Use this instead of guessing.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
}


def client() -> OpenAICompatClient:
    # Slack tokens are required by Settings but never used here, so stub them:
    # demanding them made a missing LLM key report as an unrelated config error.
    settings = load_settings(slack_bot_token="live-test", slack_app_token="live-test")
    return OpenAICompatClient(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.resolved_base_url,
        model=settings.resolved_model,
        max_tokens=256,
        temperature=0.0,
        timeout=settings.llm_timeout_seconds,
        reasoning_effort=settings.reasoning_effort,
    )


async def test_live_plain_round_trip():
    llm = client()

    text = "".join(
        [part async for part in llm.stream([{"role": "user", "content": "Reply with exactly: OK"}])]
    )

    assert "OK" in text


async def test_live_model_requests_a_tool_and_uses_its_result():
    llm = client()
    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "Look up the secret word for key 'alpha' using the lookup_word tool, "
                "then tell me the word. Do not guess."
            ),
        }
    ]

    calls: list[ToolCall] = []
    async for item in llm.stream_turn(messages, tools=[LOOKUP]):
        if isinstance(item, ToolCall):
            calls.append(item)

    assert calls, "the provider returned no tool call for a tool-only instruction"
    call = calls[0]
    assert call.name == "lookup_word"
    assert json.loads(call.arguments)["key"] == "alpha"

    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": "the secret word is OTTER",
        }
    )

    answer = "".join([part async for part in llm.stream_turn(messages, tools=[LOOKUP])])

    assert "OTTER" in answer
