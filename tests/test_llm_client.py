import json
from dataclasses import dataclass, field

import httpx2
import pytest

from slackbot_vaultnel.llm.client import (
    AuthError,
    LLMError,
    OpenAICompatClient,
    RateLimited,
    supports_temperature,
)

BASE = "https://api.deepseek.com"
URL = f"{BASE}/chat/completions"
MESSAGES = [{"role": "user", "content": "hello"}]


class Sleeper:
    """Records backoff delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class MockProvider:
    """Scripted transport. The last response repeats once the script runs out."""

    def __init__(self, *scripted) -> None:
        self.requests: list[httpx2.Request] = []
        self._scripted = list(scripted)

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        item = self._scripted[0] if len(self._scripted) == 1 else self._scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


@dataclass
class Harness:
    client: OpenAICompatClient
    provider: MockProvider = field(default_factory=lambda: MockProvider())
    sleeper: Sleeper = field(default_factory=Sleeper)


def make_client(*scripted, **overrides) -> Harness:
    provider = MockProvider(*scripted)
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(provider))
    sleeper = overrides.pop("sleeper", Sleeper())
    reasoning_effort = overrides.pop("reasoning_effort", "none")
    client = OpenAICompatClient(
        api_key="sk-test",
        base_url=BASE,
        model="deepseek-chat",
        max_tokens=64,
        temperature=0.7,
        max_attempts=3,
        sleep=sleeper,
        reasoning_effort=reasoning_effort,
        http_client=http_client,
        **overrides,
    )
    return Harness(client=client, provider=provider, sleeper=sleeper)


def chunk(content: str | None = None, **delta_extra) -> str:
    delta: dict = {"content": content} if content is not None else {}
    delta.update(delta_extra)
    payload = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def sse(*chunks: str) -> httpx2.Response:
    body = "".join(chunks) + "data: [DONE]\n\n"
    return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})


def json_response(payload: dict | None = None, status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, json=payload or {})


def rate_limited() -> httpx2.Response:
    return json_response({"error": {"message": "slow down"}}, status=429)


def completion_json(text: str = "Hello", usage: dict | None = None) -> dict:
    payload = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


async def test_stream_sends_the_expected_request():
    harness = make_client(sse(chunk("Hi")))

    text = "".join([part async for part in harness.client.stream(MESSAGES)])

    assert text == "Hi"
    assert harness.provider.requests
    request = harness.provider.requests[0]
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "deepseek-chat"
    assert body["messages"] == MESSAGES
    assert body["stream"] is True
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.7


async def test_stream_yields_deltas_in_order_and_skips_empty_ones():
    harness = make_client(sse(chunk("Hel"), chunk(), chunk("lo, "), chunk("world")))

    parts = [part async for part in harness.client.stream(MESSAGES)]

    assert parts == ["Hel", "lo, ", "world"]


async def test_reasoning_deltas_are_never_shown_to_users():
    harness = make_client(
        sse(
            chunk(None, reasoning_content="internal monologue"),
            chunk("answer"),
            chunk(None, reasoning_content="more thinking"),
        )
    )

    assert [part async for part in harness.client.stream(MESSAGES)] == ["answer"]


async def test_call_level_overrides_beat_constructor_defaults():
    harness = make_client(sse(chunk("ok")))

    _ = [
        part async for part in harness.client.stream(MESSAGES, model="deepseek-flash", max_tokens=8)
    ]

    body = json.loads(harness.provider.requests[0].content)
    assert body["model"] == "deepseek-flash"
    assert body["max_tokens"] == 8


async def test_temperature_can_be_suppressed_per_call():
    harness = make_client(sse(chunk("ok")))

    _ = [part async for part in harness.client.stream(MESSAGES, temperature=None)]

    assert "temperature" not in json.loads(harness.provider.requests[0].content)


@pytest.mark.parametrize(
    "model, expected",
    [
        ("deepseek-chat", True),
        ("deepseek-flash", True),
        ("deepseek-v4-pro", True),
        ("deepseek-reasoner", False),
        ("o3-mini", False),
        ("o1", False),
        ("some-model-thinking", False),
    ],
)
def test_temperature_support_is_model_aware(model, expected):
    assert supports_temperature(model) is expected


async def test_temperature_is_dropped_for_reasoner_models():
    harness = make_client(sse(chunk("ok")))

    _ = [part async for part in harness.client.stream(MESSAGES, model="deepseek-reasoner")]

    assert "temperature" not in json.loads(harness.provider.requests[0].content)


async def test_complete_returns_text_and_usage():
    harness = make_client(
        json_response(
            completion_json(
                "Hello", usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            )
        )
    )

    result = await harness.client.complete(MESSAGES)

    assert result.text == "Hello"
    assert result.tokens_in == 7
    assert result.tokens_out == 3
    assert result.model == "deepseek-flash"


async def test_complete_tolerates_missing_usage():
    harness = make_client(json_response(completion_json("hi")))

    result = await harness.client.complete(MESSAGES)

    assert result.text == "hi"
    assert result.tokens_in is None
    assert result.tokens_out is None


async def test_complete_sends_stream_false():
    harness = make_client(json_response(completion_json()))

    await harness.client.complete(MESSAGES)

    assert json.loads(harness.provider.requests[0].content)["stream"] is False


async def test_rejected_key_is_an_auth_error_and_is_not_retried():
    harness = make_client(json_response({"error": {"message": "bad key"}}, status=401))

    with pytest.raises(AuthError, match="LLM_API_KEY"):
        await harness.client.complete(MESSAGES)

    assert len(harness.provider.requests) == 1
    assert harness.sleeper.delays == []


async def test_rate_limit_is_retried_with_exponential_backoff():
    harness = make_client(rate_limited(), rate_limited(), json_response(completion_json("finally")))

    result = await harness.client.complete(MESSAGES)

    assert result.text == "finally"
    assert len(harness.provider.requests) == 3
    assert harness.sleeper.delays == [0.5, 1.0]


async def test_persistent_rate_limit_surfaces_as_rate_limited():
    harness = make_client(rate_limited())

    with pytest.raises(RateLimited):
        await harness.client.complete(MESSAGES)

    assert len(harness.provider.requests) == 3


async def test_server_errors_are_retried_then_reported():
    harness = make_client(json_response({"error": {"message": "unavailable"}}, status=503))

    with pytest.raises(LLMError, match="503"):
        await harness.client.complete(MESSAGES)

    assert len(harness.provider.requests) == 3
    assert harness.sleeper.delays == [0.5, 1.0]


async def test_bad_requests_are_not_retried():
    harness = make_client(json_response({"error": {"message": "bad param"}}, status=400))

    with pytest.raises(LLMError, match="bad param"):
        await harness.client.complete(MESSAGES)

    assert len(harness.provider.requests) == 1
    assert harness.sleeper.delays == []


async def test_timeouts_are_retried_and_reported_as_llm_errors():
    harness = make_client(httpx2.ReadTimeout("too slow"))

    with pytest.raises(LLMError, match="timed out"):
        await harness.client.complete(MESSAGES)

    assert len(harness.provider.requests) == 3
    assert harness.sleeper.delays == [0.5, 1.0]


async def test_stream_retries_like_complete():
    harness = make_client(rate_limited(), sse(chunk("ok")))

    text = "".join([part async for part in harness.client.stream(MESSAGES)])

    assert text == "ok"
    assert len(harness.provider.requests) == 2
    assert harness.sleeper.delays == [0.5]


async def test_reasoning_effort_is_forwarded_to_the_provider():
    harness = make_client(sse(chunk("ok")))

    _ = [part async for part in harness.client.stream(MESSAGES)]

    assert harness.provider.body()["reasoning_effort"] == "none"


async def test_reasoning_effort_is_omitted_when_unset():
    harness = make_client(sse(chunk("ok")), reasoning_effort=None)

    _ = [part async for part in harness.client.stream(MESSAGES)]

    assert "reasoning_effort" not in harness.provider.body()
