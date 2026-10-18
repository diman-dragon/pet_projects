// src/constants/schemas/UserDbSchema.ts

import { APP_ROLES } from '../../services/auth/permissions';

export const USER_DB_SCHEMA = `
  CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    login         TEXT    UNIQUE NOT NULL,
    password_hash TEXT    NOT NULL,
    salt          TEXT    NOT NULL,
    role          TEXT    NOT NULL,
    full_name     TEXT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login    DATETIME
  );

  CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id     INTEGER,
    actor_login  TEXT,
    actor_role   TEXT,
    action       TEXT    NOT NULL,
    entity_type  TEXT    NOT NULL,
    entity_id    TEXT,
    entity_label TEXT,
    details      TEXT,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS audit_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL UNIQUE,
    entries_count INTEGER NOT NULL,
    last_audit_id INTEGER,
    hmac_hex      TEXT NOT NULL,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS master_key_status (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    kdf            TEXT NOT NULL DEFAULT 'ARGON2ID',
    salt           BLOB,
    memory_kib     INTEGER NOT NULL DEFAULT 262144,
    iterations     INTEGER NOT NULL DEFAULT 3,
    parallelism    INTEGER NOT NULL DEFAULT 1,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS app_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    level      TEXT    NOT NULL,
    tag        TEXT    NOT NULL,
    message    TEXT    NOT NULL,
    details    TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS agency_dictionary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,
    normalized_name TEXT    NOT NULL UNIQUE,
    code            TEXT    NOT NULL UNIQUE,
    source          TEXT    NOT NULL DEFAULT 'system',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS chat_sessions (
    id                    TEXT PRIMARY KEY,
    agent_id              TEXT NOT NULL,
    title                 TEXT,
    origin                TEXT NOT NULL DEFAULT 'global',
    context_scope         TEXT NOT NULL DEFAULT 'global',
    last_message_at       DATETIME,
    last_summarized_index INTEGER NOT NULL DEFAULT 0,
    created_at            DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at            DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS chat_messages (
    id             TEXT    PRIMARY KEY,
    session_id     TEXT    NOT NULL,
    agent_id       TEXT    NOT NULL,
    message_index  INTEGER NOT NULL,
    role           TEXT    NOT NULL,
    content        TEXT    NOT NULL,
    token_estimate INTEGER NOT NULL DEFAULT 0,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
    UNIQUE(session_id, message_index)
  );

  CREATE TABLE IF NOT EXISTS chat_summaries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT    NOT NULL,
    agent_id            TEXT    NOT NULL,
    origin              TEXT    NOT NULL DEFAULT 'global',
    context_scope       TEXT    NOT NULL DEFAULT 'global',
    start_message_index INTEGER NOT NULL,
    end_message_index   INTEGER NOT NULL,
    summary_text        TEXT    NOT NULL,
    token_estimate      INTEGER NOT NULL DEFAULT 0,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS chat_memory_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    agent_id        TEXT    NOT NULL,
    origin          TEXT    NOT NULL DEFAULT 'global',
    context_scope   TEXT    NOT NULL DEFAULT 'global',
    source_type     TEXT    NOT NULL,
    source_ref      TEXT    NOT NULL,
    role            TEXT,
    text            TEXT    NOT NULL,
    token_estimate  INTEGER NOT NULL DEFAULT 0,
    embedding_model TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
    UNIQUE(session_id, source_type, source_ref)
  );

  CREATE TABLE IF NOT EXISTS chat_memory_embeddings (
    record_id   INTEGER PRIMARY KEY,
    session_id  TEXT    NOT NULL,
    agent_id    TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    dim         INTEGER NOT NULL DEFAULT 768,
    vector_blob BLOB    NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(record_id) REFERENCES chat_memory_records(id) ON DELETE CASCADE
  );

  -- Индексы
  CREATE INDEX IF NOT EXISTS idx_audit_created_at           ON audit_log(created_at DESC);
  CREATE INDEX IF NOT EXISTS idx_audit_snapshots_date       ON audit_snapshots(snapshot_date DESC);
  CREATE INDEX IF NOT EXISTS idx_app_logs_created_at        ON app_logs(created_at DESC);
  CREATE INDEX IF NOT EXISTS idx_agency_dictionary_code     ON agency_dictionary(code);
  CREATE INDEX IF NOT EXISTS idx_agency_dictionary_name     ON agency_dictionary(normalized_name);
  CREATE INDEX IF NOT EXISTS idx_chat_messages_session_order ON chat_messages(session_id, message_index);
  CREATE INDEX IF NOT EXISTS idx_chat_summaries_session_range ON chat_summaries(session_id, end_message_index DESC);
  CREATE INDEX IF NOT EXISTS idx_chat_memory_context         ON chat_memory_records(agent_id, origin, context_scope, created_at DESC);
  CREATE INDEX IF NOT EXISTS idx_chat_memory_embeddings_model ON chat_memory_embeddings(model, agent_id, session_id);

  CREATE TRIGGER IF NOT EXISTS audit_log_no_update
  BEFORE UPDATE ON audit_log
  BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only');
  END;

  CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
  BEFORE DELETE ON audit_log
  BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only');
  END;
`;

export const USER_ROLES = APP_ROLES;
