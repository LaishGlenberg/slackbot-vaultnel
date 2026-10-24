"""Stream one answer to stdout.

A debugging tool, not a bot entrypoint:

    uv run python -m slackbot_vaultnel.llm.cli "why is my sqlite locked?"
"""

from __future__ import annotations

import asyncio
import sys

from slackbot_vaultnel.config import load_settings
from slackbot_vaultnel.llm.client import OpenAICompatClient


async def run(prompt: str) -> None:
    settings = load_settings()
    client = OpenAICompatClient(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.resolved_base_url,
        model=settings.resolved_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
    )
    messages = [
        {"role": "system", "content": settings.resolved_system_prompt},
        {"role": "user", "content": prompt},
    ]
    async for delta in client.stream(messages):
        sys.stdout.write(delta)
        sys.stdout.flush()
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    prompt = " ".join(args) or "Say hello in three words."
    asyncio.run(run(prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
