# Agent notes

## Project

Slack AI assistant (Vaultnel): Socket Mode bot, streaming replies, tiered
conversation memory in SQLite. Python 3.12, uv, asyncio.

- Design spec: `docs/spec.md` — read the relevant section before implementing a
  slice; update it when a decision changes.
- Decision log: `docs/decisions.md` — add an entry for choices with tradeoffs.
- Slack app definition: `slack/manifest.yaml`.
- Env template: `.env.example` (never commit `.env`).

## Status

Done: config, token budgeting, store (+stateful machine), LLM client,
thinking-level control, `memory/context.py` (tier selection + hard budget
ceiling), `slack/render.py` (streaming, throttling, chunking + fence repair),
`slack/handlers.py` (filter/dedupe, thread locks, replies, persistence),
`slack/app.py` + `__main__.py` (ack-first lazy listeners, socket lifecycle),
`memory/writer.py` (single write queue + rolling compaction, bad-summary
guardrails), `memory/facts.py` (strict JSON parse, scope keys, dedupe, caps,
prompt retrieval), `llm/prompts.py`, `/ask forget me`.
230 tests green; the bot connects and answers live.
Next slice: FTS5 snippet recall — same-channel BM25, exclude the current
thread, pass snippets into `PromptRequest`.

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

Do not pile unrelated changes on top of eachother without committing.

## Docs

If making a big change from the spec or test plan, add an entry to decisions.md