"""Provider-agnostic chat client.

DeepSeek and OpenRouter are both OpenAI-compatible, so one code path serves
both and switching providers is an env change. This module owns the parts that
must not be scattered through the bot: request shaping, retry policy, and
turning SDK exceptions into errors a Slack user can actually understand.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx2
import openai
from openai import AsyncOpenAI

Messages = Sequence[Mapping[str, str]]
Sleep = Callable[[float], Awaitable[None]]


class LLMError(RuntimeError):
    """A model call failed in a way that should be reported to the user."""


class AuthError(LLMError):
    """The provider rejected the API key."""


class RateLimited(LLMError):
    """The provider kept returning 429 after every retry."""


class _Unset:
    def __repr__(self) -> str:
        return "<unset>"


UNSET = _Unset()
Temperature = float | None | _Unset


def supports_temperature(model: str) -> bool:
    """Reasoning models reject (or silently misuse) sampling parameters."""

    lowered = model.lower()
    if "reasoner" in lowered or lowered.endswith("-thinking") or "-thinking-" in lowered:
        return False
    return re.match(r"^o\d", lowered) is None


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    tokens_in: int | None
    tokens_out: int | None


class LLMClient(Protocol):
    def stream(
        self,
        messages: Messages,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: Temperature = UNSET,
    ) -> AsyncIterator[str]: ...

    async def complete(
        self,
        messages: Messages,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: Temperature = UNSET,
    ) -> Completion: ...


class OpenAICompatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float | None = 0.7,
        timeout: float = 120.0,
        max_attempts: int = 3,
        backoff_base: float = 0.5,
        sleep: Sleep = asyncio.sleep,
        reasoning_effort: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # this module owns retries, so counts stay honest
            http_client=http_client,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_attempts = max_attempts
        self._reasoning_effort = reasoning_effort
        self._backoff_base = backoff_base
        self._sleep = sleep

    # -- public API --------------------------------------------------------

    async def complete(
        self,
        messages: Messages,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: Temperature = UNSET,
    ) -> Completion:
        params = self._params(
            messages, model=model, max_tokens=max_tokens, temperature=temperature, stream=False
        )
        response = await self._call(lambda: self._client.chat.completions.create(**params))
        message = response.choices[0].message if response.choices else None
        usage = response.usage
        return Completion(
            text=(message.content or "") if message is not None else "",
            model=response.model or params["model"],
            tokens_in=usage.prompt_tokens if usage else None,
            tokens_out=usage.completion_tokens if usage else None,
        )

    async def stream(
        self,
        messages: Messages,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: Temperature = UNSET,
    ) -> AsyncIterator[str]:
        params = self._params(
            messages, model=model, max_tokens=max_tokens, temperature=temperature, stream=True
        )
        # Retries only cover failures before the first delta: once text has been
        # shown to a user, replaying the request would duplicate output.
        stream = await self._call(lambda: self._client.chat.completions.create(**params))
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
        except openai.APIError as exc:
            raise LLMError(f"the model stream was interrupted: {exc}") from exc
        finally:
            # Closing must never mask the real error.
            with contextlib.suppress(Exception):
                await stream.close()

    # -- internals ---------------------------------------------------------

    def _params(
        self,
        messages: Messages,
        *,
        model: str | None,
        max_tokens: int | None,
        temperature: Temperature,
        stream: bool,
    ) -> dict[str, object]:
        chosen = model or self._model
        params: dict[str, object] = {
            "model": chosen,
            "messages": list(messages),
            "stream": stream,
        }
        limit = self._max_tokens if max_tokens is None else max_tokens
        if limit:
            params["max_tokens"] = limit
        if self._reasoning_effort is not None:
            params["reasoning_effort"] = self._reasoning_effort
        sampling = self._temperature if temperature is UNSET else temperature
        if sampling is not None and supports_temperature(chosen):
            params["temperature"] = sampling
        return params

    async def _call(self, operation: Callable[[], Awaitable[object]]):
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await operation()
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                raise AuthError(
                    "LLM_API_KEY was rejected by the provider (401). Check the key and provider."
                ) from exc
            except openai.BadRequestError as exc:
                raise LLMError(f"the provider rejected the request: {exc.message}") from exc
            except (
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as exc:
                if attempt == self._max_attempts:
                    raise self._exhausted(exc) from exc
                await self._sleep(self._backoff_base * (2 ** (attempt - 1)))
        raise AssertionError("unreachable")

    def _exhausted(self, exc: openai.APIError) -> LLMError:
        attempts = self._max_attempts
        if isinstance(exc, openai.RateLimitError):
            return RateLimited(
                f"the model provider rate limited us (429) after {attempts} attempts"
            )
        if isinstance(exc, openai.APITimeoutError):
            return LLMError(f"the model timed out after {attempts} attempts")
        if isinstance(exc, openai.APIStatusError):
            return LLMError(
                f"the model provider returned {exc.status_code} after {attempts} attempts"
            )
        return LLMError(f"could not reach the model provider after {attempts} attempts")
