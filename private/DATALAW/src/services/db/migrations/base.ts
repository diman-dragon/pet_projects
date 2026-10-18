import type { Migration } from './types';

export const BASE_MIGRATIONS: Migration[] = [
  {
    version: 1,
    key: 'clients_main',
    description: 'Schema baseline snapshot v1',
    sql: `SELECT 1;`,
  },
  {
    version: 1,
    key: 'user_system',
    description: 'Schema baseline snapshot v1',
    sql: `SELECT 1;`,
  },
  {
    version: 1,
    key: 'kb_main',
    description: 'Schema baseline snapshot v1',
    sql: `SELECT 1;`,
  },
  {
    version: 2,
    key: 'user_system',
    description: 'Set default app language to Russian',
    sql: `INSERT OR IGNORE INTO app_settings (key, value) VALUES ('ui_language', 'ru');`,
  },
  {
    version: 3,
    key: 'user_system',
    description: 'Set default language source to system',
    sql: `INSERT OR IGNORE INTO app_settings (key, value) VALUES ('ui_language_source', 'system');`,
  },
  {
    version: 4,
    key: 'user_system',
    description: 'Promote first admin to superadmin',
    sql: `UPDATE users SET role = 'SUPERADMIN' WHERE id = 1 AND role = 'ADMIN' AND NOT EXISTS (SELECT 1 FROM users WHERE role = 'SUPERADMIN');`,
  },
  {
    version: 100,
    key: 'clients_main',
    description: 'Add incoming_number to appeals for clean Rust/TS mapping parity',
    sql: `ALTER TABLE appeals ADD COLUMN incoming_number TEXT;`,
  },
  {
    version: 101,
    key: 'clients_main',
    description: 'Add composite case lookup indexes for client-year-counterparty queries',
    sql: `CREATE INDEX IF NOT EXISTS idx_cases_client_counterparty ON cases(client_id, counterparty);`,
  },
  {
    version: 102,
    key: 'clients_main',
    description: 'Add contract date index for year filters',
    sql: `CREATE INDEX IF NOT EXISTS idx_cases_contract_date ON cases(contract_date);`,
  },
  {
    version: 103,
    key: 'clients_main',
    description: 'Add line state index for hierarchy traversal',
    sql: `CREATE INDEX IF NOT EXISTS idx_lines_case_status ON lines(case_id, status);`,
  },
  {
    version: 104,
    key: 'clients_main',
    description: 'Add appeal state index for hierarchy traversal',
    sql: `CREATE INDEX IF NOT EXISTS idx_appeals_line_status ON appeals(line_id, status);`,
  },
  {
    version: 27,
    key: 'kb_main',
    description: 'Add FTS5 chunks_fts and vec_embeddings',
    sql: `
      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
      USING fts5(
        text,
        content=chunks,
        content_rowid=id,
        tokenize='unicode61 remove_diacritics 1'
      );

      INSERT INTO chunks_fts(rowid, text)
      SELECT id, text
      FROM chunks
      WHERE id NOT IN (SELECT rowid FROM chunks_fts);

      CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings
      USING vec0(
        chunk_id INTEGER PRIMARY KEY,
        library_id INTEGER,
        model TEXT,
        embedding FLOAT[768]
      );

      CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
      END;

      CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
      END;

      CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
      END;
    `,
  },
  {
    version: 28,
    key: 'kb_main',
    description: 'Add search_history table for query logging',
    sql: `
      CREATE TABLE IF NOT EXISTS search_history (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        query        TEXT    NOT NULL,
        query_vector JSON,
        model        TEXT,
        vector_hits  INTEGER DEFAULT 0,
        text_hits    INTEGER DEFAULT 0,
        created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
      );
      CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at DESC);
    `,
  },
];
