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
