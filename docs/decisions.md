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

## 2026-10 — Handlers: ack first, then stream

Long answers exceed Slack's 3-second acknowledgement window, and an unacked
event is retried — two identical replies. Event and command listeners are
therefore Bolt *lazy listeners*: the envelope is acked immediately and the
work runs in the background. Duplicate `event_id`s are also dropped in a
bounded set, so a retry that slips through cannot answer twice.

Messages within one thread share an `asyncio.Lock`, which keeps answers in
order without serializing unrelated conversations. Until `memory/writer.py`
lands, handlers append turns directly after the stream through one small
`_persist` seam; the reply path never waits on the write.

`/ask forget me` currently confirms that no user facts are stored, because
facts do not exist yet. The extraction slice replaces that branch with a real
delete.

## 2026-10 — Memory writes: one queue, compactor behind it

All turn writes go through one `MemoryWriter` task. The handler awaits the
batch only *after* the final Slack update, so the user never waits on memory,
but the per-thread lock is still held until the turns (and any compaction)
finish. That is what makes the next message in the thread see the previous
ones instead of racing the writer.

Compaction rolls forward: it only reads turns above
`summarized_through_msg_id`, keeps the newest `KEEP_RECENT_TURNS` verbatim, and
summarises the rest together with the previous summary. Empty, oversized
(>1.5× cap) or failed summaries are rejected and retried on the next turn; the
watermark never moves backwards. A thread archived by reset is skipped, so an
in-flight reply cannot resurrect it.

## 2026-10 — Facts: model chooses content, code chooses scope key

The extractor returns only `{"content", "scope"}`. Scope keys are assigned by
code (latest speaker / channel / single workspace bucket), so a model cannot
attach a fact to the wrong user or channel. Extraction is a low-priority queue
job: if the queue is full it is dropped and retried later, never delaying a
persist.

A valid JSON array — including `[]` — advances `last_extracted_msg_id`, so an
empty result is not re-asked forever. Invalid JSON or a failed call leaves the
watermark alone and retries on a later turn. Facts are whitespace-normalized,
clipped past 240 characters (dropped past 720), deduped by hash and capped per
scope; the cap evicts oldest first, while prompt retrieval reads newest first.
`/ask forget me` flushes the writer queue first, then performs the real
user-scope delete, so a pending extraction cannot re-add a fact the user just
asked to forget.

## 2026-10 — Recall: FTS5 external content, ranked by the store

`messages_fts` is an external-content FTS5 table over `messages`, maintained by
triggers, so message text is stored once and the index can always be rebuilt
from source. `PRAGMA user_version` gates a one-time rebuild for databases
created before the index existed; no migration framework is warranted at this
size.

`memory/recall.py` shapes the query (lowercased, stopword-free, de-duplicated,
capped at 12 terms, OR-ed as quoted strings) because raw message text must never
reach FTS as syntax. `Store.search_messages` owns the ranking: same channel,
current thread excluded, archived threads and rows invisible, BM25 order. User
and assistant turns are both recallable; assistant hits are labelled "Vaultnel".
Recall failure is caught in the handler and never blocks a reply.
