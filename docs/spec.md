# Vaultnel — Slack AI Assistant: Spec

Status: agreed, pre-implementation
Runtime: Python 3.12, uv, asyncio

## 1. Goal

A Slack bot ("Vaultnel") that answers questions in DMs and when @mentioned in
channels, streams its answer into the message as it is generated, and maintains
memory across a thread without ever accumulating unbounded context.

Non-goals for v1: multi-workspace OAuth distribution, file/image understanding,
embedding-indexed RAG over a corpus, admin web UI.

## 2. Scope

| In scope (v1) | Out of scope |
| --- | --- |
| DM conversations | Block Kit interactive workflows |
| `@vaultnel` in public/private channels | Multi-tenant OAuth installs |
| Automatic short- + long-term memory | Voice / transcription |
| Streaming responses | Embedding-indexed RAG |
| Provider switch: DeepSeek \| OpenRouter | Analytics dashboard |
| MCP integrations (remote tools) | Multi-instance horizontal scale |
| `/ask` + thread reset | — | |

## 3. Architecture

```
Slack (Socket Mode websocket)
        │  app_mention / message.im
        ▼
slack-bolt AsyncApp  ──►  slack/handlers.py
                              │  1. filter + dedupe
                              │  2. normalize text, resolve speaker name
                              │  3. per-thread lock (serialize a thread)
                              ▼
                        memory/context.py   ← builds the prompt
                              │  facts + thread summary + recalled snippets
                              │  + short-term window (token-budgeted)
                              ▼
                        llm/client.py  (async, streaming)
                              │  openai SDK, base_url per provider
                              ▼
                        DeepSeek  |  OpenRouter
                              │
                              ▼
                        slack/render.py: postMessage → throttled chat.update
                              │
                              ├──► memory/writer.py (background queue)
                              │      persist turn, compact, extract facts
                              ▼
                        history/store.py (sqlite, aiosqlite)
```

Socket Mode means the process only needs **outbound** access: no public URL, no
ngrok, no inbound request verification. The tradeoff is that exactly one
process may hold the socket — two connected instances each receive every event
and the bot answers twice. Horizontal scale-out is out of scope until we move
to HTTP mode.

### Module layout

```
src/slackbot_vaultnel/
  __main__.py        # entrypoint: wire dependencies, start socket handler
  config.py          # pydantic-settings Settings
  slack/app.py       # AsyncApp construction, handler registration
  slack/handlers.py  # app_mention, message.im, /ask
  slack/render.py    # streaming placeholder, throttled update, chunking
  llm/client.py      # LLMClient protocol, OpenAICompatClient, provider table
  llm/prompts.py     # system prompt, summarizer prompt, extraction prompt
  memory/context.py  # pure function: (thread, query, now) -> message list
  memory/writer.py   # background writer: persist, compact, extract
  memory/tokens.py   # token estimator (tiktoken wrapper) + budgets
  integrations/mcp.py    # remote MCP servers: discovery, routing, isolation
  integrations/tools.py  # function schemas, clipping, untrusted wrapping
  llm/agent.py           # bounded tool loop (stream -> tools -> stream)
  history/store.py   # Store protocol + SqliteStore
  history/schema.sql
tests/
```

Handlers receive dependencies as arguments (`client`, `store`, `llm`, `clock`)
rather than importing globals. That is what makes the test suite cheap: no
network, no Slack, no sleeping.

## 4. Behaviour

### 4.1 Response triggers

- `app_mention` in a channel the bot is in → reply in the mention's thread
  (`thread_ts` = event `thread_ts` or `ts`).
- `message.im` → reply in the DM thread.
- Ignored: `bot_message`, `message_changed`, `message_deleted`, other subtypes,
  and anything with our own `bot_id`/`user` id — prevents bot-to-bot loops.
- A message that is only a mention plus whitespace → one-line usage hint, no
  LLM call.

### 4.2 Per-message flow

0. Acknowledge the Socket Mode envelope immediately and run the rest as a
   Bolt lazy listener. Slack retries an event after 3 seconds and a streamed
   answer is slower than that; duplicate `event_id`s are dropped as a second
   line of defence.
1. Add `:eyes:` reaction as an instant ack (best-effort; failure is non-fatal).
2. `async with thread_lock(channel, thread_ts)` so messages in one thread are
   answered in order rather than interleaved.
3. Build the prompt (§5.4) — this is where memory is selected, not dumped.
4. Post `_Thinking…_`, keep its `ts` as the update target.
5. Stream deltas; flush at most once per `STREAM_INTERVAL` or when the pending
   buffer passes 120 chars. `chat.update` is Tier 3 (≈1 req/s per channel), so
   throttling is required, not cosmetic.
6. Final `chat.update` with complete text; swap `:eyes:` for `:white_check_mark:`
   on success or `:warning:` on error.
7. Hand the turn to the background writer (§5.5). Replying never waits on
   memory bookkeeping.

### 4.3 Long answers

`chat.postMessage` text is capped at 4000 chars. Split at paragraph, then
sentence, then hard boundary — target ≤3800 chars per message. Chunk 1 updates
the placeholder; later chunks are new thread messages. Open code fences are
re-closed and re-opened across splits so nothing renders broken.

### 4.4 Slash commands

| Command | v1 behaviour |
| --- | --- |
| `/ask <question>` | Same pipeline as a mention, replies in the current thread |
| `/ask reset` | Archive thread memory (§5.6) and confirm ephemerally |
| `/ask forget me` | Delete user-scoped facts and confirm ephemerally |

`/ask reset` and `/ask forget me` are the last line of defence — normal threads
should never need them.

## 5. Memory system

Three tiers, all optional to include, each with a hard token ceiling. The
guiding rule: **a thread can run for a week and the prompt size stays flat.**

### 5.1 Tiers

| Tier | What it is | Storage | Selection | Budget |
| --- | --- | --- | --- | --- |
| Short-term | Recent turns verbatim | `messages` | newest-first until budget, min 4 messages | `SHORT_TERM_TOKENS` (2000) |
| Mid-term | Rolling summary of older turns in this thread | `threads.summary` | always, if non-empty | `SUMMARY_MAX_TOKENS` (400) |
| Long-term | Durable facts + recalled snippets from other threads | `facts`, `messages` via FTS5 | facts: all in scope; snippets: BM25 top-k | 400 + 600 |

Short-term is the only tier that grows; it grows until compaction folds the
oldest turns into mid-term and resets it. Long-term is bounded by eviction.

### 5.2 Compaction (mid-term)

Triggered after a turn is persisted when unsummarized tokens in the thread
exceed `COMPACT_TRIGGER_TOKENS` (2500):

1. Keep the newest `KEEP_RECENT_TURNS` turns (6) verbatim.
2. Take every older turn with `id > threads.summarized_through_msg_id`.
3. `summary' = summarize(summary, those turns)` — one LLM call, ≤400 output
   tokens, instructed to keep decisions, constraints, names, open questions and
   drop pleasantries and restated boilerplate.
4. Store `summary'` and the new high-water mark in one transaction.

Compaction is **rolling but reconstructable**: raw messages are never deleted,
so a bad summary can be rebuilt from source (`vaultnel recompact <thread>` — a
maintenance command, not a user-facing feature). Guardrails: reject a summary
that is empty or exceeds 1.5× the cap, keep the previous one, retry next turn.

### 5.3 Long-term: facts and recall

**Fact extraction** runs asynchronously every `FACT_EXTRACTION_EVERY` (4)
assistant turns in a thread, over the turns since the last extraction. One LLM
call returning strict JSON:

```json
[{"content": "Ships on Fly.io with sqlite on a volume", "scope": "channel"}]
```

- `scope` is `user` (about the person), `channel` (about the project/team), or
  `workspace`. The model never supplies scope keys: `user` is the latest speaker
  in the window, `channel` is the thread's channel, and `workspace` is one
  shared bucket (the bot is single-workspace in v1). Anything else is dropped.
- Extraction fires once at least four assistant turns have accumulated above
  `last_extracted_msg_id`. Valid JSON (including `[]`) advances the watermark;
  invalid JSON or a failed call leaves it in place and retries on a later turn.
- Facts are one-liners: content is whitespace-normalized, clipped past 240
  characters, and dropped beyond 3× that length.
- Facts are deduped by a normalized-content hash, capped at
  `FACT_MAX_PER_SCOPE` (100) per scope, oldest-evicted first.
- Unknown scope or an unusable entry → drop that entry, keep the rest, log at
  debug. Extraction failure never affects the reply.
- A fact already contradicted by a newer fact is left to the model to weigh —
  no contradiction detection in v1.

**Snippet recall** answers "what did we settle on X last week?" without loading
old threads:

1. The current message becomes up to 12 lowercased, de-duplicated, stopword-free
   terms, OR-ed together as quoted strings so nothing is read as FTS syntax.
2. FTS5 `MATCH` runs against `messages_fts`, an external-content index over
   `messages` kept current by insert/update/delete triggers. Archived threads and
   archived rows are invisible.
3. Results are scoped to the same channel and exclude the current thread (it is
   already in the short-term window).
4. Top `RECALL_TOP_K` (3) by BM25, each clipped to ~200 tokens by prompt assembly.

User and assistant turns are both recallable, so settled answers are not lost;
assistant snippets are labelled "Vaultnel". Older databases rebuild the index
once on startup (`PRAGMA user_version` migration), and raw messages stay the
source of truth.

SQLite 3.45 with FTS5 is present in this environment (verified); `sqlite-vec`
is not, so v1 is keyword retrieval only. Embeddings are a v2 upgrade behind the
same interface (`Store.search_messages`).

### 5.4 Prompt assembly

`memory/context.py` is a pure function `build(request, store, budget) -> list`
and returns messages in this order:

```
system:  persona + scope rules + speaker name
system:  known facts (user scope, then channel scope)     ≤ 400 tok
system:  thread summary                                   ≤ 400 tok
system:  recalled snippets, labeled with date + author    ≤ 600 tok
...      short-term turns verbatim                        ≤ 2000 tok
user:    current message
```

Total ceiling `PROMPT_BUDGET_TOKENS` (6000). Slots are allocated greedily in
that order and each tier trims itself; nothing is silently concatenated. If the
ceiling would still break (a short-term window padded by the 4-message
minimum), the oldest window turns are dropped. The current message is
truncated only if the mandatory persona + question cannot fit at all. Token
counts are computed at insert time and stored on the row, so assembly does not
re-tokenize history. `tiktoken` `cl100k_base` is used as an approximation —
documented as approximate for DeepSeek/OpenRouter tokenizers — with output
reserved at `LLM_MAX_TOKENS`.

Long-term inputs are **passed in**, not fetched here: `PromptRequest` carries
`facts` and `snippets`, already scoped and ranked by the caller. Assembly owns
selection and formatting only. That keeps it a pure function of
`(request, store, budget)` and lets retrieval (facts, FTS) land later without
touching prompt assembly.

### 5.5 Write path

All memory mutations go through one background `MemoryWriter` consuming an
`asyncio.Queue`. Single writer means no sqlite write contention and no locks
around SQL. Queue is bounded (`MEMORY_QUEUE_MAX=100`). Fact extraction is low
priority: when the queue is full the extraction job is dropped and retried on
a later turn. Persist jobs are never dropped — they apply backpressure to the
enqueueing handler instead.

### 5.6 Reset

| Action | Effect |
| --- | --- |
| 🗑️ reaction on any bot message, or `/ask reset` | thread archived: `messages.archived=1`, summary cleared; facts are not thread-scoped in v1, so user/channel/workspace facts survive |
| `/ask forget me` | user-scoped facts deleted for the requester |
| Thread idle > `THREAD_TTL_HOURS` (72) | summary+window skipped (treated as fresh), recall still works — automatic, no user action |
| `vaultnel recompact` | rebuild a summary from raw messages |

Archiving keeps rows for audit and re-compaction; it only removes them from
prompt assembly. That is what makes reset rare instead of routine.

A 🗑️ (`wastebasket`) reaction is resolved through `bot_messages`, which maps
every message the bot posted (including extra chunks of a long reply) to its
thread. The placeholder is linked as soon as it is posted, so reacting while an
answer is still streaming also resets. Reactions on other people's messages, on
unknown timestamps, or from the bot itself are ignored. `/ask reset` and the
reaction share `store.archive_thread`, so facts and raw rows survive either way.

### 5.7 Integrations (MCP)

External services are configuration, not code. `mcp.json` describes remote MCP
servers; another integration is another entry, not another module. It lives
next to the database — `data/mcp.json` in development, `/data/mcp.json` in the
container — so the existing volume already carries it and no new mount or path
variable is needed. A missing file means no integrations. `mcp.example.json` is
the template.

```json
{"mcpServers": {"github": {
    "url": "https://api.githubcopilot.com/mcp/",
    "token_env": "GITHUB_TOKEN",
    "tools": ["get_pull_request", "list_pull_requests", "get_check_runs"]}}}
```

- Discovery runs once at startup and connects lazily per server. A server that
  fails is logged and skipped; the rest still work. Only when every server fails
  does the bot start without tools.
- `token_env` **names** the environment variable holding a bearer token. Secrets
  are never inlined in config, never logged, and never sent to the model. The
  name must be upper-case, so a pasted token fails at startup instead of
  half-working.
- `tools` is an allowlist. Remote servers expose 30–40 tools whose schemas cost
  prompt space before the user says anything, so an empty allowlist on a 6000
  token budget is a mistake, not a default.
- Model-facing names are `<server>__<tool>`, sanitised to the provider's
  `[A-Za-z0-9_-]{1,64}` function-name shape, and de-duplicated after
  truncation.
- The loop is local (`llm/agent.py`), not a framework: stream a turn, execute
  the requested tools, append `assistant.tool_calls` + `tool` messages, repeat.
  It is bounded by `MCP_ROUNDS`, `MCP_CALLS_PER_TURN`, and a final round that
  runs **without** tools so an answer is always produced.
- Each result is clipped to `MCP_RESULT_TOKENS` and wrapped in
  `--- begin <server>/<tool> (untrusted external content…) ---` delimiters. The
  end marker is escaped inside the body: issue titles, PR bodies and file
  contents are attacker-controlled.
- Tool failures (bad JSON arguments, unknown tool, dead integration) become
  `tool` content the model can read. They never raise into the reply path.
- `stream_turn` yields text deltas as they arrive and tool calls once the stream
  ends, so `slack/render.py` still sees text only and "text never shrinks" holds.

## 6. Configuration

`pydantic-settings`, loaded from `.env` (gitignored), template committed as
`.env.example`. Required fields fail fast at startup naming the missing var.

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `SLACK_BOT_TOKEN` | yes | — | `xoxb-…` |
| `SLACK_APP_TOKEN` | yes | — | `xapp-…`, needs `connections:write` |
| `SLACK_SIGNING_SECRET` | no | `""` | unused in Socket Mode |
| `LLM_PROVIDER` | no | `deepseek` | `deepseek` \| `openrouter` |
| `LLM_API_KEY` | yes | — | |
| `LLM_MODEL` | no | `deepseek-chat` | |
| `LLM_UTILITY_MODEL` | no | = `LLM_MODEL` | summarizer / fact extraction |
| `LLM_BASE_URL` | no | per provider | |
| `LLM_TIMEOUT_SECONDS` | no | `120` | |
| `LLM_MAX_TOKENS` | no | `2048` | reply output cap |
| `LLM_TEMPERATURE` | no | `0.7` | dropped for reasoner models |
| `THINKING_LEVEL` | no | `off` | `off` \| `low` \| `high` \| `max`; `off` is sent as `reasoning_effort: "none"` |
| `SYSTEM_PROMPT` / `SYSTEM_PROMPT_FILE` | no | built-in | file wins |
| `INCLUDE_SPEAKER_NAMES` | no | `true` | resolve user ids → display names |
| `SHORT_TERM_TOKENS` | no | `2000` | |
| `SUMMARY_MAX_TOKENS` | no | `400` | |
| `COMPACT_TRIGGER_TOKENS` | no | `2500` | |
| `KEEP_RECENT_TURNS` | no | `6` | never compacted |
| `FACT_EXTRACTION_EVERY` | no | `4` | assistant turns |
| `FACT_MAX_PER_SCOPE` | no | `100` | eviction cap |
| `RECALL_TOP_K` | no | `3` | FTS snippets |
| `PROMPT_BUDGET_TOKENS` | no | `6000` | hard ceiling |
| `THREAD_TTL_HOURS` | no | `72` | soft reset |
| `MEMORY_QUEUE_MAX` | no | `100` | |
| `MCP_ROUNDS` | no | `3` | tool rounds before a tool-free final answer |
| `MCP_CALLS_PER_TURN` | no | `4` | hard cap on tool executions per reply |
| `MCP_RESULT_TOKENS` | no | `1200` | per tool result, before wrapping |
| `MCP_TIMEOUT_SECONDS` | no | `30` | per MCP request |
| `DB_PATH` | no | `data/vaultnel.db` | parent dir auto-created |
| `MAX_CONCURRENT_LLM` | no | `4` | global semaphore |
| `STREAM_INTERVAL` | no | `1.0` | seconds between updates |
| `LOG_LEVEL` | no | `INFO` | |

Integration servers are the one exception to env-var config: they live in
`mcp.json` next to the database because a JSON blob inside `.env` needs shell
escaping and hides the token-naming rule. Budgets stay in env.

Secrets are never logged; startup logs a redacted summary (first 6 chars of
each token).

### Provider table

| Provider | Base URL | Default model |
| --- | --- | --- |
| deepseek | `https://api.deepseek.com` | `deepseek-flash` |
| openrouter | `https://openrouter.ai/api/v1` | `deepseek/deepseek-chat` |

DeepSeek currently serves `deepseek-flash` (fast) and `deepseek-v4-pro`
(flagship). The legacy `deepseek-chat` / `reasoner` aliases still answer but
both resolve to `deepseek-flash`, and every reasoning model spends
`max_tokens` on hidden reasoning before emitting content. The bot therefore
pins `deepseek-flash` and controls thinking explicitly through
`THINKING_LEVEL` → `reasoning_effort` (`off` → `none`, since `off` is rejected
by the API). Default is off for fast answers; `low`/`high`/`max` are available
per deployment. Verified against the live API.

Both are OpenAI-compatible, so one `openai`-SDK code path serves both and
switching is env-only. Models that reject `temperature` (reasoner family) have
the param dropped rather than failing the call.

## 7. Data model

`aiosqlite`, WAL mode, `busy_timeout=5000`, one connection owned by the writer.
Schema applied at startup from `history/schema.sql` with idempotent DDL. A
`PRAGMA user_version` step rebuilds the FTS index once for databases created
before it existed; no migration framework until the schema stops moving.

```sql
CREATE TABLE IF NOT EXISTS threads (
    id               INTEGER PRIMARY KEY,
    channel_id       TEXT NOT NULL,
    thread_ts        TEXT NOT NULL,
    summary          TEXT NOT NULL DEFAULT '',
    summarized_through_msg_id INTEGER NOT NULL DEFAULT 0,
    last_extracted_msg_id     INTEGER NOT NULL DEFAULT 0,
    archived         INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE (channel_id, thread_ts)
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY,
    thread_id    INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
    content      TEXT NOT NULL,
    slack_user   TEXT,
    display_name TEXT,
    model        TEXT,
    tokens       INTEGER NOT NULL DEFAULT 0,
    archived     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, id);

-- posted bot messages (the placeholder and any extra chunks), so a 🗑️ reaction
-- can be mapped back to the thread it answered
CREATE TABLE IF NOT EXISTS bot_messages (
    slack_ts   TEXT PRIMARY KEY,
    thread_id  INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bot_messages_thread ON bot_messages(thread_id);

CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY,
    scope       TEXT NOT NULL CHECK (scope IN ('user','channel','workspace')),
    scope_key   TEXT NOT NULL,
    content     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_msg_id INTEGER,
    created_at  TEXT NOT NULL,
    UNIQUE (scope, scope_key, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_facts_scope ON facts(scope, scope_key);

-- Keyword recall (sqlite-vec unavailable in this environment)
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content='messages', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE OF content ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
```

Invariants:

- `messages.tokens` is always set at insert; prompt assembly sums it instead of
  re-tokenizing.
- `summarized_through_msg_id` is monotonic per thread; compaction never
  re-reads below it.
- `facts.content_hash` is `sha256(scope + normalized_ws_lower(content))`.
- each Slack message timestamp maps to at most one thread in `bot_messages`.
- Archiving is a flag, never a delete (except `/ask forget`).

Retention: no automatic pruning in v1. `vaultnel prune --older-than 90d` is a
later nicety.

## 8. Error handling

| Failure | Behaviour |
| --- | --- |
| Missing/invalid LLM key (401) | Edit placeholder to "check LLM_API_KEY", log once |
| Rate limit / 429 | Exponential backoff, 3 attempts, then friendly error |
| Timeout / connection reset | Friendly message, full traceback logged |
| Slack `chat.update` failure | Log, keep generating, single final `postMessage` fallback |
| SQLite locked | Retry 3× at 100ms, then continue without memory |
| Compaction/extraction failure | Non-fatal, dropped, retried next trigger; never touches the reply |
| Unhandled handler exception | Catch-all posts an apology; event never dies silently |

Logging: plain `logging`, one correlation id (`channel/thread_ts`) injected on
every line for a message's lifetime.

## 9. Testing strategy

Everything below runs offline; no Slack, no LLM, no sleeps.

| Layer | Tooling | Asserts |
| --- | --- | --- |
| `store` contract | `pytest.mark.parametrize` over `InMemoryStore` + `SqliteStore` | identical behaviour for both implementations |
| `store` stateful | Hypothesis `RuleBasedStateMachine` | invariants hold under arbitrary interleavings (§9.1) |
| `memory/context` | fake store, fake clock | budget ceilings respected, tier ordering, exclusions |
| compaction | scripted LLM returning canned summaries | trigger thresholds, high-water mark, bad-summary guard |
| fact extraction | scripted LLM, malformed JSON cases | dedupe, caps, scope validation, silent failure |
| `llm/client` | injected `httpx2` MockTransport | URL/model/auth/params, delta assembly, retry/backoff, reasoner param drop |
| `slack/render` | fake client + fake clock | chunk boundaries, fence repair, throttle cadence |
| `integrations/mcp` | fake session factory + real `mcp.types` | lazy connect, headers, allowlists, partial failure |
| `integrations/tools` | pure functions | namespacing, clipping, injection markers |
| `llm/agent` | `ScriptedTurnLLM` + fake caller | round/call caps, error recovery, result budget |
| `slack/handlers` | fake client, `InMemoryStore`, `ScriptedLLM` | mention stripping, thread targeting, ignored subtypes, no double reply |
| live smoke | `@pytest.mark.slow` | one real round trip, skipped without `LLM_API_KEY` |

Coverage gate: 85% on `memory/`, `history/`, `llm/`, `slack/render`,
`slack/handlers`. Wiring in `__main__.py` and `slack/app.py` is excluded.

### 9.1 Model-based and stateful testing

Example-based tests prove the happy path; stateful tests hunt for orders of
operations nobody thought of. Two machines, both
`hypothesis.stateful.RuleBasedStateMachine`:

**StoreMachine** — drives a thread through arbitrary interleavings of
`append_turn`, `compact`, `extract_facts`, `archive`, `open_other_thread`.
Rules write to the store *and* to a plain-dict shadow model; invariants compare
the two and assert design rules:

- `summarized_through_msg_id` never decreases
- no message is both folded into the summary and inside the short-term window
- archived threads contribute nothing to prompt assembly
- `facts` stays unique per `(scope, scope_key, content_hash)` and under its cap
- window contents are always a suffix of the thread (no holes)

The whole machine is parametrized over both store implementations, so any
divergence between the fast fake and SQLite fails here rather than in a handler
test.

**StreamerMachine** — rules `delta(text=…)` and `tick(seconds=…)`, invariants:

- every posted/updated text is a prefix of the full accumulated answer
- text never shrinks between updates for the same message
- updates are spaced by at least `STREAM_INTERVAL` unless the ≥120-character
  burst rule skipped the wait (the deliberate rate/feel tradeoff in §4.2)
- the final update always equals the complete answer (status markers stripped)

Streamed bodies are compared without the trailing `:eyes:`/`:white_check_mark:`
marker, because decoration is presentation and would otherwise break the
prefix property. Both properties were mutation-tested: always-flush and
shrink-by-one implementations fail the machine.

Hypothesis settings: `deadline=None` (tokenizer calls are slow), 50 examples
dev / 300 in CI, and every counterexample that turns up is frozen as an
`@example(...)` regression so the seed is never needed again.

Two rules keep this honest: assertions are on observable behaviour (messages
posted, update sequence, rows written), not on internals or call counts of
private helpers; and every bug fix starts with the failing test that proves it.

## 10. Development workflow (TDD)

Pragmatic red-green-refactor, one small slice at a time. Not dogma — spikes are
allowed, but a spike is thrown away and re-implemented test-first.

Per slice:

1. **Red** — write the test from the spec's acceptance criteria against the
   interface, run `uv run pytest -q -k <slice>`, watch it fail for the *right*
   reason (assertion, not import error).
2. **Green** — smallest implementation that passes. No speculative
   generality.
3. **Refactor** — tidy with tests still green.
4. **Commit** — test + implementation together, conventional subject
   (`feat(memory): compact thread summary past token budget`).

Order of slices, so every dependency exists before its consumer:

```
config        settings load/validate, missing-var errors          [done]
tokens        estimator + budget arithmetic                        [done]
store         contract suite + StoreMachine stateful invariants    [done]
llm/client    streaming, retries, params, reasoning effort         [done]
memory/ctx    tier selection, budget ceilings, prompt assembly     [done]
render        chunking, throttle, fence repair                      [done]
handlers      app_mention, message.im, /ask, reset                  [done]
wiring        __main__ + real socket  ->  bot usable in Slack here [done]
compaction    rolling summary + writer + guardrails                 [done]
extraction    facts: parse, dedupe, caps                            [done]
recall        FTS snippet retrieval                                 [done]
reset         archive flow, TTL, /ask commands                      [done]
deploy        Dockerfile + host                                     [done]
```

**Reordered after M2.** The renderer and handlers moved ahead of the memory
tiers so a working bot exists several slices earlier. This is safe because
handlers are written against the `PromptRequest` interface: adding compaction
and recall later only changes what the caller puts into that request, not the
handler pipeline or the message flow. Short-term window assembly is part of
`memory/ctx`, so the first working bot already has usable per-thread context.

The hard-ceiling guarantee is property-tested directly (`PromptBudget` highs
and lows): whatever the history, facts and snippets, the assembled prompt never
exceeds `total` and the window stays a contiguous suffix.

Interfaces get defined first because tests drive them: `Store` protocol, then
`InMemoryStore` (fast fake), then `SqliteStore` (real), with one contract suite
over both. Same shape for `LLMClient` (`ScriptedLLM` fake). Fakes live in
`tests/fakes.py`.

Time and randomness are injected (`clock: Callable[[], datetime]`), never read
from the module-level `datetime.now`, so TTL, throttling and compaction are
deterministic in tests.

CI: `uv run pytest -q` (slow tests deselected) plus `ruff`; no node matrices.

## 11. Deployment

Target: one always-on GCE e2-micro VM running the container. Socket Mode is
outbound only, so the VM needs no inbound ports beyond SSH.

- `Dockerfile`: uv multi-stage, non-root, `python -m slackbot_vaultnel`, with
the `tiktoken` encoding baked into the image. `DB_PATH=/data/vaultnel.db`.
- SQLite lives on the `vaultnel-data` Docker volume; image updates never touch
it. Single instance only — no horizontal scaling in v1.
- `cloudbuild.yaml`: push to `main` builds and pushes a commit-tagged image
plus `:latest` to Artifact Registry. The VM runs a systemd timer that pulls
`:latest` every five minutes and recreates the container only when the image
ID changes. Runbook: `deploy/README.md`.
- Health: startup logs the Socket Mode connection and a heartbeat every five
minutes; config errors exit non-zero so the restart policy kicks in.
- Secrets from `/opt/vaultnel/.env` on the host, never baked into the image or
committed.

## 12. Milestones

| # | Deliverable | Done when |
| --- | --- | --- |
| M0 | Slack app from `slack/manifest.yaml`, tokens in `.env` | `auth.test` returns `ok: true` |
| M1 | config + tokens + LLM client, TDD'd | `pytest` green; CLI prints one reply |
| M2 | store contract suite + stateful machine + sqlite schema | both stores pass the same tests and the same StateMachine |
| M3 | prompt assembly: `PromptRequest` + tier budgets | budget ceilings hold under property tests |
| M4 | streaming renderer | long answer splits correctly, updates throttled |
| M5 | Bolt app + wiring: mention + DM with short-term memory | bot answers in a real channel |
| M6 | compaction + fact extraction + recall | prompt size flat after 50 turns; facts deduped |
| M7 | reset + TTL + error handling + README | `pytest` green incl. slow smoke |
| M8 | Dockerfile + deploy | bot online with no local process |
| M9 | MCP integrations (GitHub first) | tools discovered at startup and a tool call changes the answer |

## 13. Decisions

| Question | Decision |
| --- | --- |
| Persona | Neutral helpful assistant, override via `SYSTEM_PROMPT_FILE` |
| Default model | `deepseek-flash`, thinking `off`; see `docs/decisions.md` |
| Speaker names in prompt | Yes — resolved display names (`INCLUDE_SPEAKER_NAMES`) |
| Memory reset | Automated compaction + TTL primary; `/ask reset` + 🗑️ last resort |
| Retrieval | FTS5 keyword only in v1; embeddings behind the same interface later |

Day-to-day decisions and their rationale live in `docs/decisions.md`.

Open: whether the bot should answer unprompted in channels it's invited to
(currently: no — mention or DM only), and whether `/ask` should support a
private/ephemeral mode.
