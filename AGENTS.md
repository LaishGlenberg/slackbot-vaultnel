# Agent notes

## Project

Slack AI assistant (Vaultnel): Socket Mode bot, streaming replies, tiered
conversation memory in SQLite. Python 3.12, uv, asyncio.

- Design spec: `docs/spec.md` — read the relevant section before implementing a
  slice; update it when a decision changes.
- Slack app definition: `slack/manifest.yaml`.
- Env template: `.env.example` (never commit `.env`).

## Commands

```bash
uv run pytest -q                      # full suite, offline
uv run pytest -q -k memory            # one slice
uv run pytest -q -m slow              # live LLM smoke test (needs LLM_API_KEY)
uv run ruff check . && uv run ruff format --check .
uv run slackbot-vaultnel              # run the bot
```

Coverage: `uv run pytest -q --cov --cov-report=term-missing`.

## Workflow

TDD per slice: failing test first (red for the right reason), smallest
implementation, then refactor. See `docs/spec.md` §10 for the slice order and
§9.1 for the stateful test machines.

Tests never touch Slack, the network, or the real `.env`. Dependencies
(`Store`, `LLMClient`, `clock`) are injected; fakes live in `tests/fakes.py`.
