# Vaultnel

A Slack AI assistant with tiered conversation memory. Mention it in a thread or
DM it; it streams an answer back, remembers the conversation, and recalls
relevant decisions from other threads in the same channel.

## What it does

- **Threaded answers** — `@vaultnel …` in a channel or a DM, streamed in place.
- **Short-term memory** — the recent turns of the thread are in every prompt.
- **Rolling summary** — older turns are compacted into a running summary.
- **Durable facts** — scoped per user, channel and workspace, with dedupe.
- **Recall** — SQLite FTS5 retrieves relevant messages from other threads.
- **Integrations** — remote MCP servers (GitHub first) let it look things up
  live; results are clipped and treated as untrusted external content.
- **Reset** — 🗑️ on a bot reply or `/ask reset` archives the thread;
  `/ask forget me` deletes your facts.

Memory lives in one SQLite file, so the bot is a single always-on process. The
full design and rationale are in [`docs/spec.md`](docs/spec.md).

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), a Slack app in
Socket Mode, and an API key for DeepSeek or OpenRouter.

1. Create the Slack app from [`slack/manifest.yaml`](slack/manifest.yaml), turn
   on Socket Mode, create an app-level token with `connections:write`, and
   install the app to your workspace.
2. Configure the environment:

   ```bash
   cp .env.example .env
   # fill SLACK_BOT_TOKEN, SLACK_APP_TOKEN and LLM_API_KEY
   ```

   Optional: copy `mcp.example.json` to `data/mcp.json` to give it tools, and
   put a read-only token in `GITHUB_TOKEN` in `.env`. Secrets never go in
   `mcp.json`.

3. Run it:

   ```bash
   uv sync
   uv run slackbot-vaultnel
   ```

## Deploy

The bot runs as one always-on container on a small VM. Cloud Build builds and
pushes the image on every push to `main`; the VM polls for `:latest` and
restarts itself when it changes. Full setup and operations are in
[`deploy/README.md`](deploy/README.md).

```bash
# on the VM, once: docker + /opt/vaultnel/.env + the update timer
sudo /opt/vaultnel/update.sh
sudo docker logs -f vaultnel
```

`docker logs` shows startup, the Socket Mode connection and a heartbeat every
five minutes. The database lives in the `vaultnel-data` volume, so image
updates never touch it.

## Development

```bash
uv run pytest -q                      # offline suite
uv run pytest -q -m slow              # live LLM smoke test
uv run ruff check . && uv run ruff format --check .
```

Tests never touch Slack, the network or `.env`.
