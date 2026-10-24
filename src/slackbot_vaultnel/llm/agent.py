"""The bounded tool loop (spec §5.7).

An agent framework is the wrong shape for this bot, so the loop is local and
small: stream a turn, execute whatever tools the model asked for, feed the
results back, repeat. Three bounds keep it predictable — rounds, calls per
turn, and the size of any single tool result.

Only text is yielded. The renderer therefore never learns about tools, and the
"text only grows" invariant from ``slack/render.py`` still holds.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from slackbot_vaultnel.integrations.mcp import MCPError, MCPResult
from slackbot_vaultnel.integrations.tools import FunctionTool, wrap_tool_result
from slackbot_vaultnel.llm.client import StreamItem, ToolCall

logger = logging.getLogger(__name__)

CallTool = Callable[[FunctionTool, Mapping[str, Any]], Awaitable[MCPResult]]


@dataclass(frozen=True)
class Tooling:
    """What the handler needs to offer tools: the catalogue and the dispatcher."""

    tools: Sequence[FunctionTool]
    call: CallTool


class StreamingLLM(Protocol):
    def stream_turn(
        self, messages: Sequence[Mapping[str, Any]], *, tools: object = None
    ) -> AsyncIterator[StreamItem]: ...


class ToolAgent:
    def __init__(
        self,
        llm: StreamingLLM,
        tools: Sequence[FunctionTool],
        call: CallTool,
        *,
        max_rounds: int = 3,
        max_calls: int = 4,
        result_tokens: int = 1200,
    ) -> None:
        self._llm = llm
        self._tools = list(tools)
        self._call = call
        self._max_rounds = max(1, max_rounds)
        self._max_calls = max(0, max_calls)
        self._result_tokens = max(1, result_tokens)

    async def run(self, messages: Sequence[Mapping[str, Any]]) -> AsyncIterator[str]:
        convo = [dict(message) for message in messages]
        schemas = [tool.schema() for tool in self._tools]
        by_name = {tool.name: tool for tool in self._tools}
        calls_left = self._max_calls

        # One extra iteration: the last one runs without tools, so the model has
        # to answer with what it already has instead of looping forever.
        for round_index in range(self._max_rounds + 1):
            terminal = round_index >= self._max_rounds or calls_left <= 0 or not schemas
            requested: list[ToolCall] = []
            said: list[str] = []

            async for item in self._llm.stream_turn(convo, tools=None if terminal else schemas):
                if isinstance(item, str):
                    said.append(item)
                    yield item
                else:
                    requested.append(item)

            if terminal or not requested:
                return

            convo.append(_assistant_turn(said, requested))
            for call in requested:
                if calls_left <= 0:
                    break
                convo.append(await self._execute(call, by_name))
                calls_left -= 1

    # -- internals ---------------------------------------------------------

    async def _execute(self, call: ToolCall, by_name: Mapping[str, FunctionTool]) -> dict:
        tool = by_name.get(call.name)
        if tool is None:
            return _tool_message(
                call.id, f"[tool error] the model asked for unknown tool {call.name}"
            )

        arguments, problem = _decode_arguments(call.arguments)
        if problem is not None:
            return _tool_message(call.id, f"[tool error] {problem}")

        try:
            result = await self._call(tool, arguments)
        except MCPError as exc:
            logger.warning("tool %s/%s failed: %s", tool.server, tool.tool, exc)
            return _tool_message(call.id, self._wrap(tool, str(exc), is_error=True))
        except Exception:  # noqa: BLE001 - a broken integration must not kill a reply
            logger.exception("unexpected tool failure %s/%s", tool.server, tool.tool)
            return _tool_message(
                call.id, self._wrap(tool, "the integration failed unexpectedly", is_error=True)
            )
        return _tool_message(call.id, self._wrap(tool, result.text, is_error=result.is_error))

    def _wrap(self, tool: FunctionTool, text: str, *, is_error: bool) -> str:
        return wrap_tool_result(
            tool.server,
            tool.tool,
            text,
            max_tokens=self._result_tokens,
            is_error=is_error,
        )


def _assistant_turn(said: Sequence[str], requested: Sequence[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "".join(said) or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments or "{}"},
            }
            for call in requested
        ],
    }


def _tool_message(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _decode_arguments(raw: str) -> tuple[dict[str, Any], str | None]:
    """Model-supplied JSON is untrusted input, including its shape."""

    text = (raw or "").strip()
    if not text or text == "null":
        return {}, None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"tool arguments were not valid JSON ({exc.msg})"
    if not isinstance(parsed, dict):
        return {}, "tool arguments must be a JSON object"
    return parsed, None
