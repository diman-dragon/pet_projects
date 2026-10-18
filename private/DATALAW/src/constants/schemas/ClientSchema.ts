// src/constants/schemas/ClientSchema.ts

export const CLIENT_SCHEMA = `
-- ── 1. КЛИЕНТЫ ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS clients (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name   TEXT    NOT NULL,
    tax_id      TEXT    UNIQUE,
    type        TEXT    DEFAULT 'LEGAL',
    status      TEXT    DEFAULT 'LEAD',
    folder_name TEXT    NOT NULL,
    client_data TEXT,
    email       TEXT,
    phone       TEXT,
    summary     TEXT,
    metadata    TEXT,
    deleted_at  DATETIME DEFAULT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── 2. ДЕЛА ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cases (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id      INTEGER NOT NULL,
    case_number    TEXT,
    title          TEXT    NOT NULL,
    case_type      TEXT,
    litigation_subtype TEXT,
    counterparty   TEXT,
    status         TEXT    DEFAULT 'active',
    folder_name    TEXT,
    metadata       TEXT,
    summary        TEXT,
    contract_date  TEXT,
    contract_type  TEXT,
    contract_amount TEXT,
    metadata_description TEXT,
    mandate_check TEXT DEFAULT 'PENDING',
    mandate_notes TEXT,
    violation_type TEXT,
    risk_level TEXT DEFAULT 'UNKNOWN',
    authority_involved TEXT,
    jurisdiction_id INTEGER,
    erasure_candidate_at DATETIME,
    erasure_approved_by INTEGER,
    erasure_requested_by INTEGER,
    deleted_at     DATETIME DEFAULT NULL,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(client_id) REFERENCES clients(id) ON DELETE CASCADE
);

-- ── 3. ЛИНИИ ЗАЩИТЫ ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lines (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id        INTEGER NOT NULL,
    title          TEXT    NOT NULL,
    status         TEXT,
    folder_name    TEXT,
    description    TEXT,
    summary        TEXT,
    consultation_text TEXT,
    consultation_topic TEXT,
    contract_analysis TEXT,
    contract_risks TEXT,
    deleted_at     DATETIME DEFAULT NULL,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
);

-- ── 4. ОБРАЩЕНИЯ ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS appeals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id          INTEGER NOT NULL,
    title            TEXT    NOT NULL,
    status           TEXT    DEFAULT 'DRAFT',
    description      TEXT,
    file_path        TEXT,
    folder_name      TEXT    NOT NULL,
    agency_code      TEXT,
    appeal_date      TEXT,
    recipient        TEXT,
    incoming_number  TEXT,
    outgoing_number  TEXT,
    summary          TEXT,
    deleted_at       DATETIME DEFAULT NULL,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(line_id) REFERENCES lines(id) ON DELETE CASCADE
);

-- ── 5. ДОКУМЕНТЫ ОБРАЩЕНИЙ ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS appeal_documents (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    appeal_id           INTEGER NOT NULL,
    file_name           TEXT    NOT NULL,
    file_path           TEXT    NOT NULL,
    file_hash           TEXT    NOT NULL,
    summary             TEXT,
    key_points          TEXT,
    duplicate_of_path   TEXT,
    duplicate_appeal_id INTEGER,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(appeal_id) REFERENCES appeals(id) ON DELETE CASCADE
);

-- ── 6. СУММАРИ ОБРАЩЕНИЙ ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS appeal_summaries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    appeal_id    INTEGER NOT NULL UNIQUE,
    text         TEXT    NOT NULL,
    key_points   TEXT,
    generated_by TEXT,
    updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(appeal_id) REFERENCES appeals(id) ON DELETE CASCADE
);

-- ── 7. КОРЗИНА ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trash_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type  TEXT    NOT NULL,
    entity_id    INTEGER NOT NULL,
    entity_label TEXT,
    actor_id     INTEGER,
    actor_login  TEXT,
    action       TEXT    NOT NULL DEFAULT 'delete',
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS client_dossier_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id     INTEGER NOT NULL,
    tax_id        TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'planned',
    providers_json TEXT   NOT NULL,
    summary_text  TEXT,
    folder_path   TEXT    NOT NULL,
    manifest_path TEXT    NOT NULL,
    dossier_json  TEXT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(client_id) REFERENCES clients(id) ON DELETE CASCADE
);

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
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_document_id INTEGER NOT NULL,
    version_no   INTEGER NOT NULL,
    file_path    TEXT NOT NULL,
    file_hash    TEXT NOT NULL,
    hash_algo    TEXT DEFAULT 'SHA-256',
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(case_document_id) REFERENCES case_documents(id) ON DELETE CASCADE,
    UNIQUE(case_document_id, version_no)
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

-- ── 8. FTS5-таблицы для федеративного поиска ─────────────────────────────────
-- Используются GlobalSearchService вместо LIKE-запросов.

CREATE VIRTUAL TABLE IF NOT EXISTS clients_fts
USING fts5(
  full_name,
  email,
  tax_id,
  metadata,
  content=clients,
  content_rowid=id,
  tokenize='unicode61 remove_diacritics 1'
);

CREATE VIRTUAL TABLE IF NOT EXISTS cases_fts
USING fts5(
  title,
  case_number,
  counterparty,
  case_type,
  metadata,
  content=cases,
  content_rowid=id,
  tokenize='unicode61 remove_diacritics 1'
);

CREATE VIRTUAL TABLE IF NOT EXISTS appeals_fts
USING fts5(
  title,
  agency_code,
  incoming_number,
  outgoing_number,
  metadata,
  content=appeals,
  content_rowid=id,
  tokenize='unicode61 remove_diacritics 1'
);

CREATE VIRTUAL TABLE IF NOT EXISTS case_documents_fts
USING fts5(
  file_name,
  summary,
  category,
  content=case_documents,
  content_rowid=id,
  tokenize='unicode61 remove_diacritics 1'
);

-- ── 9. ИНДЕКСЫ ───────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_clients_status     ON clients(status);
CREATE INDEX IF NOT EXISTS idx_clients_updated    ON clients(updated_at);
CREATE INDEX IF NOT EXISTS idx_clients_deleted    ON clients(deleted_at)  WHERE deleted_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cases_client_id    ON cases(client_id);
CREATE INDEX IF NOT EXISTS idx_cases_client_counterparty ON cases(client_id, counterparty);
CREATE INDEX IF NOT EXISTS idx_cases_contract_date ON cases(contract_date);
CREATE INDEX IF NOT EXISTS idx_cases_deleted      ON cases(deleted_at)    WHERE deleted_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_lines_case_id      ON lines(case_id);
CREATE INDEX IF NOT EXISTS idx_lines_case_status  ON lines(case_id, status);
CREATE INDEX IF NOT EXISTS idx_lines_deleted      ON lines(deleted_at)    WHERE deleted_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_appeals_line_id    ON appeals(line_id);
CREATE INDEX IF NOT EXISTS idx_appeals_line_status ON appeals(line_id, status);
CREATE INDEX IF NOT EXISTS idx_appeals_date       ON appeals(appeal_date);
CREATE INDEX IF NOT EXISTS idx_appeals_deleted    ON appeals(deleted_at)  WHERE deleted_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_appeals_folder     ON appeals(folder_name, appeal_date DESC);
CREATE INDEX IF NOT EXISTS idx_docs_hash          ON appeal_documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_trash_entity       ON trash_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_client_dossier_client ON client_dossier_snapshots(client_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_subjects_display_code ON subjects(display_code);
CREATE INDEX IF NOT EXISTS idx_case_subjects_case ON case_subjects(case_id, role);
CREATE INDEX IF NOT EXISTS idx_case_access_case_user ON case_access(case_id, user_id, revoked_at, expires_at);
CREATE INDEX IF NOT EXISTS idx_pii_key_grants_subject_user ON pii_key_grants(subject_id, user_id);
CREATE INDEX IF NOT EXISTS idx_case_documents_case ON case_documents(case_id, category, deleted_at);
CREATE INDEX IF NOT EXISTS idx_case_documents_hash ON case_documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_subject_documents_subject ON subject_documents(subject_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_tracks_case_type ON tracks(case_id, track_type, status);
CREATE INDEX IF NOT EXISTS idx_case_document_versions_doc ON case_document_versions(case_document_id, version_no);
CREATE INDEX IF NOT EXISTS idx_cases_jurisdiction ON cases(jurisdiction_id, status);
CREATE INDEX IF NOT EXISTS idx_cases_mandate ON cases(mandate_check, risk_level);
CREATE INDEX IF NOT EXISTS idx_strategy_case ON defense_strategy_steps(case_id, status);
CREATE INDEX IF NOT EXISTS idx_drafts_case ON document_drafts(case_id, status, grounding_status);

-- ── 10. Триггеры синхронизации clients_fts ────────────────────────────────────
CREATE TRIGGER IF NOT EXISTS clients_fts_ai AFTER INSERT ON clients BEGIN
  INSERT INTO clients_fts(rowid, full_name, email, tax_id, metadata)
    VALUES (new.id, new.full_name, COALESCE(new.email, ''), COALESCE(new.tax_id, ''), COALESCE(new.metadata, ''));
END;

CREATE TRIGGER IF NOT EXISTS clients_fts_ad AFTER DELETE ON clients BEGIN
  INSERT INTO clients_fts(clients_fts, rowid, full_name, email, tax_id, metadata)
    VALUES ('delete', old.id, old.full_name, COALESCE(old.email, ''), COALESCE(old.tax_id, ''), COALESCE(old.metadata, ''));
END;

CREATE TRIGGER IF NOT EXISTS clients_fts_au AFTER UPDATE ON clients BEGIN
  INSERT INTO clients_fts(clients_fts, rowid, full_name, email, tax_id, metadata)
    VALUES ('delete', old.id, old.full_name, COALESCE(old.email, ''), COALESCE(old.tax_id, ''), COALESCE(old.metadata, ''));
  INSERT INTO clients_fts(rowid, full_name, email, tax_id, metadata)
    VALUES (new.id, new.full_name, COALESCE(new.email, ''), COALESCE(new.tax_id, ''), COALESCE(new.metadata, ''));
END;

-- ── 11. Триггеры синхронизации cases_fts ──────────────────────────────────────
CREATE TRIGGER IF NOT EXISTS cases_fts_ai AFTER INSERT ON cases BEGIN
  INSERT INTO cases_fts(rowid, title, case_number, counterparty, case_type, metadata)
    VALUES (new.id, new.title, COALESCE(new.case_number, ''), COALESCE(new.counterparty, ''), COALESCE(new.case_type, ''), COALESCE(new.metadata, ''));
END;

CREATE TRIGGER IF NOT EXISTS cases_fts_ad AFTER DELETE ON cases BEGIN
  INSERT INTO cases_fts(cases_fts, rowid, title, case_number, counterparty, case_type, metadata)
    VALUES ('delete', old.id, old.title, COALESCE(old.case_number, ''), COALESCE(old.counterparty, ''), COALESCE(old.case_type, ''), COALESCE(old.metadata, ''));
END;

CREATE TRIGGER IF NOT EXISTS cases_fts_au AFTER UPDATE ON cases BEGIN
  INSERT INTO cases_fts(cases_fts, rowid, title, case_number, counterparty, case_type, metadata)
    VALUES ('delete', old.id, old.title, COALESCE(old.case_number, ''), COALESCE(old.counterparty, ''), COALESCE(old.case_type, ''), COALESCE(old.metadata, ''));
  INSERT INTO cases_fts(rowid, title, case_number, counterparty, case_type, metadata)
    VALUES (new.id, new.title, COALESCE(new.case_number, ''), COALESCE(new.counterparty, ''), COALESCE(new.case_type, ''), COALESCE(new.metadata, ''));
END;

-- ── 12. Триггеры синхронизации appeals_fts ────────────────────────────────────
CREATE TRIGGER IF NOT EXISTS appeals_fts_ai AFTER INSERT ON appeals BEGIN
  INSERT INTO appeals_fts(rowid, title, agency_code, incoming_number, outgoing_number, metadata)
    VALUES (new.id, new.title, COALESCE(new.agency_code, ''), COALESCE(new.incoming_number, ''), COALESCE(new.outgoing_number, ''), '');
END;

CREATE TRIGGER IF NOT EXISTS appeals_fts_ad AFTER DELETE ON appeals BEGIN
  INSERT INTO appeals_fts(appeals_fts, rowid, title, agency_code, incoming_number, outgoing_number, metadata)
    VALUES ('delete', old.id, old.title, COALESCE(old.agency_code, ''), COALESCE(old.incoming_number, ''), COALESCE(old.outgoing_number, ''), '');
END;

CREATE TRIGGER IF NOT EXISTS appeals_fts_au AFTER UPDATE ON appeals BEGIN
  INSERT INTO appeals_fts(appeals_fts, rowid, title, agency_code, incoming_number, outgoing_number, metadata)
    VALUES ('delete', old.id, old.title, COALESCE(old.agency_code, ''), COALESCE(old.incoming_number, ''), COALESCE(old.outgoing_number, ''), '');
  INSERT INTO appeals_fts(rowid, title, agency_code, incoming_number, outgoing_number, metadata)
    VALUES (new.id, new.title, COALESCE(new.agency_code, ''), COALESCE(new.incoming_number, ''), COALESCE(new.outgoing_number, ''), '');
END;

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
`;
