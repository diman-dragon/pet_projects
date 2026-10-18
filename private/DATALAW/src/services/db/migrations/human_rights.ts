import type { Migration } from './types';

export const HUMAN_RIGHTS_MIGRATIONS: Migration[] = [
  {
    version: 400,
    key: 'clients_main',
    description: 'Add human-rights workflow fields to cases',
    sql: `
      ALTER TABLE cases ADD COLUMN mandate_check TEXT DEFAULT 'PENDING';
      ALTER TABLE cases ADD COLUMN mandate_notes TEXT;
      ALTER TABLE cases ADD COLUMN violation_type TEXT;
      ALTER TABLE cases ADD COLUMN risk_level TEXT DEFAULT 'UNKNOWN';
      ALTER TABLE cases ADD COLUMN authority_involved TEXT;
      ALTER TABLE cases ADD COLUMN jurisdiction_id INTEGER;
      ALTER TABLE cases ADD COLUMN erasure_candidate_at DATETIME;
      ALTER TABLE cases ADD COLUMN erasure_approved_by INTEGER;
      ALTER TABLE cases ADD COLUMN erasure_requested_by INTEGER;
    `,
  },
  {
    version: 401,
    key: 'clients_main',
    description: 'Create subjects and case access tables',
    sql: `
      CREATE TABLE IF NOT EXISTS subjects (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        display_code TEXT UNIQUE NOT NULL,
        pii_ref      TEXT NOT NULL,
        created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS case_subjects (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id     INTEGER NOT NULL,
        subject_id  INTEGER NOT NULL,
        role        TEXT NOT NULL,
        risk_level  TEXT DEFAULT 'UNKNOWN',
        consent_ref TEXT,
        added_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE,
        FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
        UNIQUE(case_id, subject_id, role)
      );

      CREATE TABLE IF NOT EXISTS case_access (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id        INTEGER NOT NULL,
        user_id        INTEGER NOT NULL,
        reason         TEXT NOT NULL,
        requested_by   INTEGER,
        approved_by    INTEGER NOT NULL,
        granted_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
        expires_at     DATETIME,
        revoked_at     DATETIME,
        UNIQUE(case_id, user_id),
        FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS pii_key_grants (
        id                          INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id                  INTEGER NOT NULL,
        user_id                     INTEGER NOT NULL,
        wrapped_dek                 BLOB NOT NULL,
        granted_via_case_access_id  INTEGER NOT NULL,
        created_at                  DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
        FOREIGN KEY(granted_via_case_access_id) REFERENCES case_access(id) ON DELETE CASCADE,
        UNIQUE(subject_id, user_id, granted_via_case_access_id)
      );
    `,
  },
  {
    version: 402,
    key: 'clients_main',
    description: 'Create case documents and drafting tables',
    sql: `
      CREATE TABLE IF NOT EXISTS case_documents (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id      INTEGER NOT NULL,
        subject_id   INTEGER,
        category     TEXT NOT NULL,
        file_name    TEXT NOT NULL,
        file_path    TEXT NOT NULL,
        file_hash    TEXT NOT NULL,
        hash_algo    TEXT DEFAULT 'SHA-256',
        collected_at DATETIME,
        summary      TEXT,
        category_suggested TEXT,
        risk_level_suggested TEXT,
        deleted_at   DATETIME DEFAULT NULL,
        created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE,
        FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE SET NULL
      );

      CREATE TABLE IF NOT EXISTS defense_strategy_steps (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id            INTEGER NOT NULL,
        claim              TEXT NOT NULL,
        legal_basis_json   TEXT NOT NULL,
        supporting_facts_json TEXT NOT NULL,
        confidence         TEXT NOT NULL DEFAULT 'LOW',
        status             TEXT NOT NULL DEFAULT 'SUGGESTED',
        created_by_agent   TEXT,
        approved_by        INTEGER,
        approved_at        DATETIME,
        created_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS document_drafts (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id           INTEGER NOT NULL,
        strategy_step_id  INTEGER,
        title             TEXT NOT NULL,
        body              TEXT NOT NULL,
        status            TEXT NOT NULL DEFAULT 'DRAFT',
        grounding_status  TEXT NOT NULL DEFAULT 'PENDING',
        verifier_report   TEXT,
        approved_by       INTEGER,
        approved_at       DATETIME,
        created_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE,
        FOREIGN KEY(strategy_step_id) REFERENCES defense_strategy_steps(id) ON DELETE SET NULL
      );
    `,
  },
  {
    version: 403,
    key: 'clients_main',
    description: 'Create case document FTS and human-rights indexes',
    sql: `
      CREATE VIRTUAL TABLE IF NOT EXISTS case_documents_fts
      USING fts5(
        file_name,
        summary,
        category,
        content=case_documents,
        content_rowid=id,
        tokenize='unicode61 remove_diacritics 1'
      );

      CREATE INDEX IF NOT EXISTS idx_subjects_display_code ON subjects(display_code);
      CREATE INDEX IF NOT EXISTS idx_case_subjects_case ON case_subjects(case_id, role);
      CREATE INDEX IF NOT EXISTS idx_case_access_case_user ON case_access(case_id, user_id, revoked_at, expires_at);
      CREATE INDEX IF NOT EXISTS idx_pii_key_grants_subject_user ON pii_key_grants(subject_id, user_id);
      CREATE INDEX IF NOT EXISTS idx_case_documents_case ON case_documents(case_id, category, deleted_at);
      CREATE INDEX IF NOT EXISTS idx_case_documents_hash ON case_documents(file_hash);
      CREATE INDEX IF NOT EXISTS idx_cases_jurisdiction ON cases(jurisdiction_id, status);
      CREATE INDEX IF NOT EXISTS idx_cases_mandate ON cases(mandate_check, risk_level);
      CREATE INDEX IF NOT EXISTS idx_strategy_case ON defense_strategy_steps(case_id, status);
      CREATE INDEX IF NOT EXISTS idx_drafts_case ON document_drafts(case_id, status, grounding_status);
    `,
  },
  {
    version: 404,
    key: 'clients_main',
    description: 'Add case document FTS sync triggers',
    sql: `
      CREATE TRIGGER IF NOT EXISTS case_documents_fts_ai AFTER INSERT ON case_documents BEGIN
        INSERT INTO case_documents_fts(rowid, file_name, summary, category)
          VALUES (new.id, new.file_name, COALESCE(new.summary, ''), new.category);
      END;

      CREATE TRIGGER IF NOT EXISTS case_documents_fts_ad AFTER DELETE ON case_documents BEGIN
        INSERT INTO case_documents_fts(case_documents_fts, rowid, file_name, summary, category)
          VALUES ('delete', old.id, old.file_name, COALESCE(old.summary, ''), old.category);
      END;

      CREATE TRIGGER IF NOT EXISTS case_documents_fts_au AFTER UPDATE ON case_documents BEGIN
        INSERT INTO case_documents_fts(case_documents_fts, rowid, file_name, summary, category)
          VALUES ('delete', old.id, old.file_name, COALESCE(old.summary, ''), old.category);
        INSERT INTO case_documents_fts(rowid, file_name, summary, category)
          VALUES (new.id, new.file_name, COALESCE(new.summary, ''), new.category);
      END;
    `,
  },
  {
    version: 405,
    key: 'clients_main',
    description: 'Add subject documents, tracks, case templates and document versions',
    sql: `
      CREATE TABLE IF NOT EXISTS subject_documents (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id   INTEGER NOT NULL,
        category     TEXT NOT NULL,
        file_name    TEXT NOT NULL,
        file_path    TEXT NOT NULL,
        file_hash    TEXT NOT NULL,
        collected_at DATETIME,
        summary      TEXT,
        deleted_at   DATETIME,
        created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS tracks (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id       INTEGER NOT NULL,
        track_type    TEXT NOT NULL,
        title         TEXT NOT NULL,
        status        TEXT DEFAULT 'ACTIVE',
        owner_user_id INTEGER,
        opened_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
        closed_at     DATETIME,
        FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS track_legal_details (
        track_id         INTEGER PRIMARY KEY,
        court_body_id    INTEGER,
        case_number      TEXT,
        proceeding_stage TEXT,
        lawyer_contact   TEXT,
        FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS track_medical_details (
        track_id          INTEGER PRIMARY KEY,
        diagnosis         TEXT,
        treating_facility TEXT,
        doctor_contact    TEXT,
        treatment_plan    TEXT,
        FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS track_media_advocacy_details (
        track_id            INTEGER PRIMARY KEY,
        campaign_name       TEXT,
        journalist_contacts TEXT,
        publications        TEXT,
        embargo_until       DATE,
        FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS track_international_mechanism_details (
        track_id             INTEGER PRIMARY KEY,
        mechanism            TEXT,
        communication_number TEXT,
        submission_deadline  DATE,
        status_at_mechanism  TEXT,
        FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS track_security_details (
        track_id              INTEGER PRIMARY KEY,
        threat_level          TEXT,
        relocation_status     TEXT,
        safe_contact_protocol TEXT,
        FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS track_family_support_details (
        track_id         INTEGER PRIMARY KEY,
        dependents_count INTEGER,
        support_type     TEXT,
        FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS case_templates (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        code                    TEXT UNIQUE NOT NULL,
        name                    TEXT NOT NULL,
        default_tracks          TEXT NOT NULL,
        relevant_doc_categories TEXT NOT NULL,
        milestone_checklist     TEXT,
        jurisdiction_hint       INTEGER
      );

      CREATE TABLE IF NOT EXISTS case_document_versions (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        case_document_id  INTEGER NOT NULL,
        version_no        INTEGER NOT NULL,
        file_path         TEXT NOT NULL,
        file_hash         TEXT NOT NULL,
        hash_algo         TEXT DEFAULT 'SHA-256',
        created_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(case_document_id) REFERENCES case_documents(id) ON DELETE CASCADE,
        UNIQUE(case_document_id, version_no)
      );

      CREATE INDEX IF NOT EXISTS idx_subject_documents_subject ON subject_documents(subject_id, deleted_at);
      CREATE INDEX IF NOT EXISTS idx_tracks_case_type ON tracks(case_id, track_type, status);
      CREATE INDEX IF NOT EXISTS idx_case_document_versions_doc ON case_document_versions(case_document_id, version_no);
    `,
  },
  {
    version: 50,
    key: 'user_system',
    description: 'Make audit log append-only and add audit snapshots',
    sql: `
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

      CREATE INDEX IF NOT EXISTS idx_audit_snapshots_date ON audit_snapshots(snapshot_date DESC);

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
    `,
  },
];
