PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS threads (
    id                        INTEGER PRIMARY KEY,
    channel_id                TEXT NOT NULL,
    thread_ts                 TEXT NOT NULL,
    summary                   TEXT NOT NULL DEFAULT '',
    summarized_through_msg_id INTEGER NOT NULL DEFAULT 0,
    last_extracted_msg_id     INTEGER NOT NULL DEFAULT 0,
    archived                  INTEGER NOT NULL DEFAULT 0,
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL,
    UNIQUE (channel_id, thread_ts)
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY,
    thread_id    INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content      TEXT NOT NULL,
    slack_user   TEXT,
    display_name TEXT,
    model        TEXT,
    tokens       INTEGER NOT NULL DEFAULT 0 CHECK (tokens >= 0),
    archived     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, id);

-- Maps a posted bot message (including every chunk of a long reply) back to the
-- thread it answers, so a 🗑️ reaction can reset that thread. Slack timestamps
-- are unique per workspace, hence the primary key.
CREATE TABLE IF NOT EXISTS bot_messages (
    slack_ts   TEXT PRIMARY KEY,
    thread_id  INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bot_messages_thread ON bot_messages(thread_id);

-- Keyword recall (spec §5.3). External content keeps the text in one place;
-- the triggers below keep the index in step with ``messages``.
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id',
    tokenize='unicode61'
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

CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY,
    scope         TEXT NOT NULL CHECK (scope IN ('user', 'channel', 'workspace')),
    scope_key     TEXT NOT NULL,
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    source_msg_id INTEGER,
    created_at    TEXT NOT NULL,
    UNIQUE (scope, scope_key, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_facts_scope ON facts(scope, scope_key);
