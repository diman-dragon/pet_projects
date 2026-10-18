export const LEGAL_CORPUS_SCHEMA = `
  CREATE TABLE IF NOT EXISTS jurisdictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    code          TEXT UNIQUE NOT NULL,
    name          TEXT NOT NULL,
    legal_system  TEXT,
    parent_id     INTEGER,
    FOREIGN KEY(parent_id) REFERENCES jurisdictions(id)
  );

  CREATE TABLE IF NOT EXISTS courts_bodies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    jurisdiction_id INTEGER NOT NULL,
    name            TEXT NOT NULL,
    level           TEXT,
    code            TEXT,
    FOREIGN KEY(jurisdiction_id) REFERENCES jurisdictions(id)
  );

  CREATE TABLE IF NOT EXISTS acts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    jurisdiction_id INTEGER NOT NULL,
    court_body_id   INTEGER,
    act_type        TEXT NOT NULL,
    title           TEXT NOT NULL,
    act_number      TEXT,
    adopted_date    TEXT,
    effective_from  TEXT,
    effective_to    TEXT,
    status          TEXT DEFAULT 'IN_FORCE',
    lang            TEXT,
    source_url      TEXT,
    content_raw     TEXT,
    hash            TEXT,
    ingested_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(jurisdiction_id) REFERENCES jurisdictions(id),
    FOREIGN KEY(court_body_id) REFERENCES courts_bodies(id)
  );

  CREATE TABLE IF NOT EXISTS act_versions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    act_id        INTEGER NOT NULL,
    version_label TEXT NOT NULL,
    valid_from    TEXT NOT NULL,
    valid_to      TEXT,
    content_raw   TEXT NOT NULL,
    FOREIGN KEY(act_id) REFERENCES acts(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS act_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    act_id      INTEGER NOT NULL,
    version_id  INTEGER,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    article_ref TEXT,
    FOREIGN KEY(act_id) REFERENCES acts(id) ON DELETE CASCADE,
    FOREIGN KEY(version_id) REFERENCES act_versions(id) ON DELETE SET NULL,
    UNIQUE(act_id, version_id, chunk_index)
  );

  CREATE VIRTUAL TABLE IF NOT EXISTS act_chunks_fts
  USING fts5(
    text,
    content=act_chunks,
    content_rowid=id,
    tokenize='unicode61 remove_diacritics 1'
  );

  CREATE TABLE IF NOT EXISTS act_citations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    citing_act_id     INTEGER NOT NULL,
    cited_act_id      INTEGER NOT NULL,
    cited_article_ref TEXT,
    citation_type     TEXT,
    FOREIGN KEY(citing_act_id) REFERENCES acts(id) ON DELETE CASCADE,
    FOREIGN KEY(cited_act_id) REFERENCES acts(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS legal_vector_manifest (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id        INTEGER NOT NULL UNIQUE,
    jurisdiction_id INTEGER NOT NULL,
    model           TEXT NOT NULL,
    dim             INTEGER NOT NULL,
    lance_table     TEXT NOT NULL,
    indexed_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(chunk_id) REFERENCES act_chunks(id) ON DELETE CASCADE,
    FOREIGN KEY(jurisdiction_id) REFERENCES jurisdictions(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS document_templates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    jurisdiction_id INTEGER,
    act_type        TEXT,
    case_type       TEXT,
    title           TEXT NOT NULL,
    body            TEXT NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(jurisdiction_id) REFERENCES jurisdictions(id)
  );

  CREATE INDEX IF NOT EXISTS idx_acts_jurisdiction ON acts(jurisdiction_id, act_type, status);
  CREATE INDEX IF NOT EXISTS idx_acts_hash ON acts(hash);
  CREATE INDEX IF NOT EXISTS idx_act_chunks_act ON act_chunks(act_id);
  CREATE INDEX IF NOT EXISTS idx_act_chunks_article ON act_chunks(article_ref);
  CREATE INDEX IF NOT EXISTS idx_citations_citing ON act_citations(citing_act_id);
  CREATE INDEX IF NOT EXISTS idx_citations_cited ON act_citations(cited_act_id);
  CREATE INDEX IF NOT EXISTS idx_legal_vector_manifest_jurisdiction ON legal_vector_manifest(jurisdiction_id, model);

  CREATE TRIGGER IF NOT EXISTS act_chunks_fts_ai AFTER INSERT ON act_chunks BEGIN
    INSERT INTO act_chunks_fts(rowid, text) VALUES (new.id, new.text);
  END;

  CREATE TRIGGER IF NOT EXISTS act_chunks_fts_ad AFTER DELETE ON act_chunks BEGIN
    INSERT INTO act_chunks_fts(act_chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
  END;

  CREATE TRIGGER IF NOT EXISTS act_chunks_fts_au AFTER UPDATE ON act_chunks BEGIN
    INSERT INTO act_chunks_fts(act_chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO act_chunks_fts(rowid, text) VALUES (new.id, new.text);
  END;
`;
