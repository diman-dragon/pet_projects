export const PII_VAULT_SCHEMA = `
  CREATE TABLE IF NOT EXISTS pii_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pii_ref         TEXT UNIQUE NOT NULL,
    subject_code    TEXT NOT NULL,
    payload_cipher  BLOB NOT NULL,
    nonce           BLOB NOT NULL,
    aad             TEXT,
    algo            TEXT NOT NULL DEFAULT 'CHACHA20-POLY1305',
    key_version     INTEGER NOT NULL DEFAULT 1,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS pii_access_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pii_ref     TEXT NOT NULL,
    subject_id  INTEGER,
    actor_id    INTEGER,
    action      TEXT NOT NULL,
    reason      TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE INDEX IF NOT EXISTS idx_pii_records_ref ON pii_records(pii_ref);
  CREATE INDEX IF NOT EXISTS idx_pii_events_ref ON pii_access_events(pii_ref, created_at DESC);
`;
