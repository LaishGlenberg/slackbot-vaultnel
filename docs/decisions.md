# Decisions

Short log of choices that are easy to forget the reasoning for. The full design
is in `docs/spec.md`; this file records *why*, and what it costs.

## 2026-10 — Transport: Socket Mode

No public URL, no inbound firewall rules, works from a laptop. Cost: only one
process may hold the socket, so horizontal scale-out requires an HTTP Events
migration later.

## 2026-10 — Memory: three tiers, automated, reset is last resort

Unbounded context is a cost and quality bug. Short-term window → rolling
thread summary → durable facts + FTS recall, each with a token ceiling. Reset
(`/ask reset`, 🗑️ reaction) exists, but compaction and TTL should make it rare.
Recall is FTS5 because `sqlite-vec` is unavailable in this environment;
embeddings fit behind the same interface later.

## 2026-10 — Testing: contract suite + stateful machines over a shadow model

One contract suite over `InMemoryStore` and `SqliteStore`, plus Hypothesis
`RuleBasedStateMachine` tests that compare the store against a plain-dict
model after every rule. Fake and real cannot drift silently. Same pattern
planned for the streaming renderer.

Property tests already killed two assumptions: BPE token counts are neither
prefix-monotonic (`"gno"` = 2, `"gnore"` = 1) nor subadditive (`"FB"+"BB"` = 3
vs 1+1), so budgets rely on byte-length bounds instead.

## 2026-10 — LLM layer: one OpenAI-compatible client

DeepSeek and OpenRouter share a code path; provider switching is env-only. Our
client owns retries (`max_retries=0` in the SDK) so attempt counts are honest,
and mid-stream failures are never retried because text already shown to a user
would be duplicated.

`openai` 3.x uses `httpx2`, which `respx` does not patch — tests inject an
`httpx2.MockTransport` instead. Now a rule: verify the HTTP layer of a new SDK
before trusting a mock.

## 2026-10 — Model: `deepseek-flash`, thinking off by default

The API offers `deepseek-flash` and `deepseek-v4-pro`; `deepseek-chat` and
`reasoner` are legacy aliases that both resolve to `deepseek-flash`. Reasoning
tokens are billed and count against `max_tokens`, so background jobs
(summaries, fact extraction) would silently return empty content if thinking
were left on. Replies default to `THINKING_LEVEL=off`, which is sent as
`reasoning_effort: "none"` — the literal `off` is rejected with a 422.

`low` / `high` / `max` are available per deployment. Verified live: all four
values return 200 and change the reported reasoning-token count.

OpenRouter defaults are unverified — no key has been tried yet.

## 2026-10 — Slice order: working bot before memory tiers

Originally strict dependency order (memory before handlers). Reordered so
renderer + handlers land first and the bot is usable in Slack earlier. Handlers
are written against `PromptRequest`, so later compaction/recall work only
changes what the caller puts into that request.

## 2026-10 — Prompt assembly owns selection, not retrieval

`build(request, store, budget)` receives facts and recalled snippets on the
request instead of querying the facts table or FTS itself. Assembly stays a
pure, fast-to-test function of its inputs, and retrieval can land after the
working bot without changing it. Cost: callers must scope and rank memories
(facts by user/channel scope, snippets by BM25) before calling `build`.

The total budget is a hard ceiling, not a target: optional tiers are dropped
lowest-priority first, then the 4-message short-term minimum is trimmed from
the oldest end, and only a mandatory persona + question that cannot fit at all
causes truncation of the question. Property tests assert this for arbitrary
histories.

## 2026-10 — Streamer: markers are decoration, throttle is per-message

Streamed visible text carries a trailing status marker (`:eyes:` on the
placeholder, `:white_check_mark:` / `:warning:` on the final update). Tests and
the stateful machine compare `strip_status(...)`, so presentation never weakens
the real invariants (prefix, never-shrink, final equals answer).

Flushes happen when the interval elapses *or* ≥120 characters are pending, per
spec §4.2. The burst rule deliberately allows updates closer together than
`STREAM_INTERVAL`; the machine asserts spacing only for interval-driven
updates. Slack update failures are logged, retried on the next flush, and the
final text is posted as a fresh message if the last update never succeeds.
