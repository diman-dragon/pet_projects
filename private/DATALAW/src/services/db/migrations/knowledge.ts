import type { Migration } from './types';

export const KNOWLEDGE_MIGRATIONS: Migration[] = [
  {
    version: 200,
    key: 'kb_main',
    description: 'Add lang column to library for multilingual support',
    sql: `ALTER TABLE library ADD COLUMN lang TEXT NULL;`,
  },
  {
    version: 201,
    key: 'kb_main',
    description: 'Add folder_path column to library for folder-based categorization',
    sql: `ALTER TABLE library ADD COLUMN folder_path TEXT NULL;`,
  },
  {
    version: 202,
    key: 'kb_main',
    description: 'Add composite index on library(category, lang, created_at)',
    sql: `CREATE INDEX IF NOT EXISTS idx_lib_category_lang ON library(category, lang, created_at DESC);`,
  },
  {
    version: 203,
    key: 'kb_main',
    description: 'Add index on library(folder_path)',
    sql: `CREATE INDEX IF NOT EXISTS idx_lib_folder ON library(folder_path);`,
  },
  {
    version: 204,
    key: 'kb_main',
    description: 'Add index on library(updated_at DESC)',
    sql: `CREATE INDEX IF NOT EXISTS idx_lib_updated ON library(updated_at DESC);`,
  },
  {
    version: 205,
    key: 'kb_main',
    description: 'Create library_fts virtual table for fast first-phase search',
    sql: `
      CREATE VIRTUAL TABLE IF NOT EXISTS library_fts
      USING fts5(
        title,
        content_raw,
        metadata,
        content=library,
        content_rowid=id,
        tokenize='unicode61 remove_diacritics 1'
      );
    `,
  },
  {
    version: 206,
    key: 'kb_main',
    description: 'Backfill library_fts from existing library rows',
    sql: `
      INSERT INTO library_fts(rowid, title, content_raw, metadata)
      SELECT id, title, COALESCE(content_raw, ''), COALESCE(metadata, '')
      FROM library
      WHERE id NOT IN (SELECT rowid FROM library_fts);
    `,
  },
  {
    version: 207,
    key: 'kb_main',
    description: 'Add library_fts sync triggers',
    sql: `
      CREATE TRIGGER IF NOT EXISTS library_ai AFTER INSERT ON library BEGIN
        INSERT INTO library_fts(rowid, title, content_raw, metadata)
          VALUES (new.id, new.title, COALESCE(new.content_raw, ''), COALESCE(new.metadata, ''));
      END;

      CREATE TRIGGER IF NOT EXISTS library_ad AFTER DELETE ON library BEGIN
        INSERT INTO library_fts(library_fts, rowid, title, content_raw, metadata)
          VALUES ('delete', old.id, old.title, COALESCE(old.content_raw, ''), COALESCE(old.metadata, ''));
      END;

      CREATE TRIGGER IF NOT EXISTS library_au AFTER UPDATE ON library BEGIN
        INSERT INTO library_fts(library_fts, rowid, title, content_raw, metadata)
          VALUES ('delete', old.id, old.title, COALESCE(old.content_raw, ''), COALESCE(old.metadata, ''));
        INSERT INTO library_fts(rowid, title, content_raw, metadata)
          VALUES (new.id, new.title, COALESCE(new.content_raw, ''), COALESCE(new.metadata, ''));
      END;
    `,
  },
  {
    version: 208,
    key: 'kb_main',
    description: 'Add search_selections table for ranking prior',
    sql: `
      CREATE TABLE IF NOT EXISTS search_selections (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        query_hash  TEXT    NOT NULL,
        library_id  INTEGER,
        result_id   TEXT    NOT NULL,
        selected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(library_id) REFERENCES library(id) ON DELETE CASCADE
      );
      CREATE INDEX IF NOT EXISTS idx_selections_query ON search_selections(query_hash, selected_at DESC);
    `,
  },
  {
    version: 209,
    key: 'kb_main',
    description: 'Add category_rules table for folder-based auto-categorization',
    sql: `
      CREATE TABLE IF NOT EXISTS category_rules (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        rule_type    TEXT    NOT NULL,
        pattern      TEXT    NOT NULL,
        category     TEXT    NOT NULL,
        priority     INTEGER NOT NULL DEFAULT 0,
        is_system    INTEGER NOT NULL DEFAULT 0,
        created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
      );
      CREATE INDEX IF NOT EXISTS idx_category_rules_type ON category_rules(rule_type, priority DESC);
    `,
  },
  {
    version: 210,
    key: 'kb_main',
    description: 'Seed system category_rules',
    sql: `
      INSERT OR IGNORE INTO category_rules (rule_type, pattern, category, priority, is_system) VALUES
        ('extension', 'pdf',   'Documents', 10, 1),
        ('extension', 'docx',  'Documents', 10, 1),
        ('extension', 'doc',   'Documents', 10, 1),
        ('extension', 'txt',   'Documents',  5, 1),
        ('extension', 'md',    'Notes',      5, 1),
        ('extension', 'tsv',   'Data',       5, 1),
        ('extension', 'csv',   'Data',       5, 1),
        ('path_pattern', '%/contracts/%',    'Contracts',   20, 1),
        ('path_pattern', '%/договор%',       'Contracts',   20, 1),
        ('path_pattern', '%/суд%',           'Litigation',  20, 1),
        ('path_pattern', '%/court%',         'Litigation',  20, 1),
        ('path_pattern', '%/templates/%',    'Templates',   15, 1),
        ('path_pattern', '%/шаблон%',        'Templates',   15, 1),
        ('path_pattern', '%/appeals/%',      'Appeals',     15, 1),
        ('path_pattern', '%/обращен%',       'Appeals',     15, 1),
        ('default',      '*',               'General',      0, 1);
    `,
  },
  {
    version: 211,
    key: 'kb_main',
    description: 'Add document_languages table for cross-lingual retrieval',
    sql: `
      CREATE TABLE IF NOT EXISTS document_languages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        library_id  INTEGER NOT NULL,
        chunk_id    INTEGER,
        lang        TEXT    NOT NULL,
        confidence  REAL    NOT NULL DEFAULT 1.0,
        detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(library_id) REFERENCES library(id) ON DELETE CASCADE,
        FOREIGN KEY(chunk_id)   REFERENCES chunks(id)  ON DELETE CASCADE
      );
      CREATE INDEX IF NOT EXISTS idx_doc_lang_library ON document_languages(library_id);
    `,
  },
  {
    version: 212,
    key: 'kb_main',
    description: 'Add query_hash, query_lang, query_type, latency_ms to search_history',
    sql: `
      ALTER TABLE search_history ADD COLUMN query_hash  TEXT;
      ALTER TABLE search_history ADD COLUMN query_lang  TEXT;
      ALTER TABLE search_history ADD COLUMN query_type  TEXT;
      ALTER TABLE search_history ADD COLUMN latency_ms  INTEGER;
    `,
  },
  {
    version: 213,
    key: 'kb_main',
    description: 'Add index on search_history(query_hash)',
    sql: `CREATE INDEX IF NOT EXISTS idx_search_history_hash ON search_history(query_hash);`,
  },
  {
    version: 400,
    key: 'kb_main',
    description: 'Add index on search_selections(result_id) for fast prior lookup',
    sql: `CREATE INDEX IF NOT EXISTS idx_selections_result ON search_selections(result_id, selected_at DESC);`,
  },
  {
    version: 401,
    key: 'kb_main',
    description: 'Add index on search_history(query) for prefix suggestions',
    sql: `CREATE INDEX IF NOT EXISTS idx_search_history_query ON search_history(query, created_at DESC);`,
  },
  {
    version: 402,
    key: 'kb_main',
    description: 'Add partial index for non-virtual library records for GhostRecordService',
    sql: `CREATE INDEX IF NOT EXISTS idx_library_real_files ON library(file_path) WHERE file_path NOT LIKE 'virtual://%';`,
  },
  {
    version: 403,
    key: 'kb_main',
    description: 'Add index on chunks(library_id) for fast library filtering in search',
    sql: `CREATE INDEX IF NOT EXISTS idx_chunks_library_id ON chunks(library_id);`,
  },
  {
    version: 404,
    key: 'kb_main',
    description: 'Add category column to chunks table',
    sql: `ALTER TABLE chunks ADD COLUMN category TEXT DEFAULT 'General';`,
  },
  {
    version: 405,
    key: 'kb_main',
    description: 'Add index on chunks(category) for category filtering',
    sql: `CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks(category);`,
  },
  {
    version: 406,
    key: 'kb_main',
    description: 'Add composite index on chunks(library_id, category) for fast dual filtering',
    sql: `CREATE INDEX IF NOT EXISTS idx_chunks_library_category ON chunks(library_id, category);`,
  },
  {
    version: 407,
    key: 'kb_main',
    description: 'Add index on embeddings(model) for model-specific searches',
    sql: `CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);`,
  },
  {
    version: 408,
    key: 'kb_main',
    description: 'Add index on embeddings(library_id) for library-scoped embedding lookups',
    sql: `CREATE INDEX IF NOT EXISTS idx_embeddings_library_id ON embeddings(library_id);`,
  },
  {
    version: 409,
    key: 'kb_main',
    description: 'Add composite index on embeddings(model, library_id) for model+library queries',
    sql: `CREATE INDEX IF NOT EXISTS idx_embeddings_model_library ON embeddings(model, library_id);`,
  },
  {
    version: 410,
    key: 'kb_main',
    description: 'Add index on search_history(query_lang, created_at DESC) for language-aware searches',
    sql: `CREATE INDEX IF NOT EXISTS idx_search_history_lang ON search_history(query_lang, created_at DESC);`,
  },
  {
    version: 411,
    key: 'kb_main',
    description: 'Add composite index on search_history(query_hash, query_lang) for fast cache lookups',
    sql: `CREATE INDEX IF NOT EXISTS idx_search_history_hash_lang ON search_history(query_hash, query_lang);`,
  },
  {
    version: 412,
    key: 'kb_main',
    description: 'Add index on document_languages(library_id, lang) for cross-lingual retrieval',
    sql: `CREATE INDEX IF NOT EXISTS idx_doc_lang_library_lang ON document_languages(library_id, lang);`,
  },
  {
    version: 413,
    key: 'kb_main',
    description: 'Optimize SQLite query planner with ANALYZE',
    sql: `ANALYZE;`,
  },
  {
    version: 500,
    key: 'kb_main',
    description: 'Drop old vec_embeddings table with 768 dim',
    sql: `DROP TABLE IF EXISTS vec_embeddings;`,
  },
  {
    version: 501,
    key: 'kb_main',
    description: 'Create vec_embeddings with correct 1024 dimension for bge-m3',
    sql: `
      CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings
      USING vec0(
        chunk_id INTEGER PRIMARY KEY,
        library_id INTEGER,
        model TEXT,
        embedding FLOAT[1024]
      );
    `,
  },
  {
    version: 502,
    key: 'kb_main',
    description: 'Add chunker_version column to library table',
    sql: `ALTER TABLE library ADD COLUMN chunker_version INTEGER DEFAULT 0;`,
  },
  {
    version: 503,
    key: 'kb_main',
    description: 'Add article_ref column to chunks for legal citation extraction',
    sql: `ALTER TABLE chunks ADD COLUMN article_ref TEXT NULL;`,
  },
  {
    version: 504,
    key: 'kb_main',
    description: 'Add parent_id to chunks for parent-child retrieval strategy',
    sql: `ALTER TABLE chunks ADD COLUMN parent_id INTEGER;`,
  },
  {
    version: 505,
    key: 'kb_main',
    description: 'Add index on chunks(parent_id) for fast parent retrieval',
    sql: `CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id);`,
  },
];
