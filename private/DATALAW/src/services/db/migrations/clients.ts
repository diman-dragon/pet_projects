import type { Migration } from './types';

export const CLIENT_MIGRATIONS: Migration[] = [
  {
    version: 300,
    key: 'clients_main',
    description: 'Create clients_fts virtual table',
    sql: `
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
    `,
  },
  {
    version: 301,
    key: 'clients_main',
    description: 'Backfill clients_fts from existing clients',
    sql: `
      INSERT INTO clients_fts(rowid, full_name, email, tax_id, metadata)
      SELECT id, full_name, COALESCE(email, ''), COALESCE(tax_id, ''), COALESCE(metadata, '')
      FROM clients
      WHERE id NOT IN (SELECT rowid FROM clients_fts);
    `,
  },
  {
    version: 302,
    key: 'clients_main',
    description: 'Add clients_fts sync triggers',
    sql: `
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
    `,
  },
  {
    version: 303,
    key: 'clients_main',
    description: 'Create cases_fts virtual table',
    sql: `
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
    `,
  },
  {
    version: 304,
    key: 'clients_main',
    description: 'Backfill cases_fts from existing cases',
    sql: `
      INSERT INTO cases_fts(rowid, title, case_number, counterparty, case_type, metadata)
      SELECT id, title, COALESCE(case_number, ''), COALESCE(counterparty, ''), COALESCE(case_type, ''), COALESCE(metadata, '')
      FROM cases
      WHERE id NOT IN (SELECT rowid FROM cases_fts);
    `,
  },
  {
    version: 305,
    key: 'clients_main',
    description: 'Add cases_fts sync triggers',
    sql: `
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
    `,
  },
  {
    version: 306,
    key: 'clients_main',
    description: 'Create appeals_fts virtual table',
    sql: `
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
    `,
  },
  {
    version: 307,
    key: 'clients_main',
    description: 'Backfill appeals_fts from existing appeals',
    sql: `
      INSERT INTO appeals_fts(rowid, title, agency_code, incoming_number, outgoing_number, metadata)
      SELECT id, title, COALESCE(agency_code, ''), COALESCE(incoming_number, ''), COALESCE(outgoing_number, ''), ''
      FROM appeals
      WHERE id NOT IN (SELECT rowid FROM appeals_fts);
    `,
  },
  {
    version: 308,
    key: 'clients_main',
    description: 'Add appeals_fts sync triggers',
    sql: `
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
    `,
  },
  {
    version: 309,
    key: 'clients_main',
    description: 'Add folder index on appeals(folder_name, appeal_date)',
    sql: `CREATE INDEX IF NOT EXISTS idx_appeals_folder ON appeals(folder_name, appeal_date DESC);`,
  },
];
