// src/services/db/migrations/knowledge_graph.ts
//
// Миграции для Knowledge Graph (Граф знаний).
// Версии 600–620 зарезервированы за KG.
//
// Структура:
//   kg_entities   — узлы графа (персоны, организации, законы, суды и т.д.)
//   kg_relations  — рёбра между узлами (направленные, с типом и весом)
//   kg_entity_sources — привязка сущности к исходному документу/чанку
//   kg_extraction_jobs — очередь/журнал заданий извлечения
//
// Все таблицы используют ключ 'kb_main', так как Knowledge Graph
// логически является частью базы знаний.

import type { Migration } from './types';

export const KNOWLEDGE_GRAPH_MIGRATIONS: Migration[] = [
  // ─── 600: Основная таблица сущностей ─────────────────────────────────────
  {
    version: 600,
    key: 'kb_main',
    description: 'Create kg_entities table for Knowledge Graph nodes',
    sql: `
      CREATE TABLE IF NOT EXISTS kg_entities (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT    NOT NULL,
        entity_type   TEXT    NOT NULL,
        -- Типы: PERSON | ORGANIZATION | LAW | COURT | CASE_REF |
        --       CONCEPT | DATE | AMOUNT | LOCATION | OTHER
        canonical_name TEXT   NULL,
        -- Нормализованная форма для дедупликации (lowercase, trimmed)
        description   TEXT    NULL,
        metadata      TEXT    NULL,
        -- JSON-поле: дополнительные атрибуты (ИНН, номер дела и т.д.)
        confidence    REAL    NOT NULL DEFAULT 1.0,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE INDEX IF NOT EXISTS idx_kg_entities_type
        ON kg_entities(entity_type);

      CREATE INDEX IF NOT EXISTS idx_kg_entities_canonical
        ON kg_entities(canonical_name);

      CREATE INDEX IF NOT EXISTS idx_kg_entities_name
        ON kg_entities(name);
    `,
  },

  // ─── 601: FTS5 по сущностям ───────────────────────────────────────────────
  {
    version: 601,
    key: 'kb_main',
    description: 'Create kg_entities_fts virtual table for fast entity search',
    sql: `
      CREATE VIRTUAL TABLE IF NOT EXISTS kg_entities_fts
      USING fts5(
        name,
        canonical_name,
        description,
        content=kg_entities,
        content_rowid=id,
        tokenize='unicode61 remove_diacritics 1'
      );
    `,
  },

  // ─── 602: Триггеры синхронизации FTS ────────────────────────────────────
  {
    version: 602,
    key: 'kb_main',
    description: 'Add FTS sync triggers for kg_entities',
    sql: `
      CREATE TRIGGER IF NOT EXISTS kg_entities_ai AFTER INSERT ON kg_entities BEGIN
        INSERT INTO kg_entities_fts(rowid, name, canonical_name, description)
          VALUES (new.id, new.name,
                  COALESCE(new.canonical_name, ''),
                  COALESCE(new.description, ''));
      END;

      CREATE TRIGGER IF NOT EXISTS kg_entities_ad AFTER DELETE ON kg_entities BEGIN
        INSERT INTO kg_entities_fts(kg_entities_fts, rowid, name, canonical_name, description)
          VALUES ('delete', old.id, old.name,
                  COALESCE(old.canonical_name, ''),
                  COALESCE(old.description, ''));
      END;

      CREATE TRIGGER IF NOT EXISTS kg_entities_au AFTER UPDATE ON kg_entities BEGIN
        INSERT INTO kg_entities_fts(kg_entities_fts, rowid, name, canonical_name, description)
          VALUES ('delete', old.id, old.name,
                  COALESCE(old.canonical_name, ''),
                  COALESCE(old.description, ''));
        INSERT INTO kg_entities_fts(rowid, name, canonical_name, description)
          VALUES (new.id, new.name,
                  COALESCE(new.canonical_name, ''),
                  COALESCE(new.description, ''));
      END;

      CREATE TRIGGER IF NOT EXISTS kg_entities_updated_at
      AFTER UPDATE ON kg_entities BEGIN
        UPDATE kg_entities SET updated_at = CURRENT_TIMESTAMP WHERE id = new.id;
      END;
    `,
  },

  // ─── 603: Таблица рёбер (отношения между сущностями) ──────────────────────
  {
    version: 603,
    key: 'kb_main',
    description: 'Create kg_relations table for Knowledge Graph edges',
    sql: `
      CREATE TABLE IF NOT EXISTS kg_relations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id       INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
        target_id       INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
        relation_type   TEXT    NOT NULL,
        -- Типы: PARTY_TO | REPRESENTS | FILED_BY | DECIDED_BY | REFERENCES_LAW |
        --       COUNTERPARTY | EMPLOYED_BY | RELATED_CASE | INVOLVED_IN | OTHER
        label           TEXT    NULL,
        -- Человекочитаемое описание связи (опционально)
        weight          REAL    NOT NULL DEFAULT 1.0,
        -- Вес/уверенность (0.0–1.0)
        direction       TEXT    NOT NULL DEFAULT 'directed',
        -- 'directed' | 'undirected'
        metadata        TEXT    NULL,
        -- JSON: контекст, цитата из документа, дата установления связи
        confidence      REAL    NOT NULL DEFAULT 1.0,
        created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE INDEX IF NOT EXISTS idx_kg_relations_source
        ON kg_relations(source_id, relation_type);

      CREATE INDEX IF NOT EXISTS idx_kg_relations_target
        ON kg_relations(target_id, relation_type);

      CREATE INDEX IF NOT EXISTS idx_kg_relations_type
        ON kg_relations(relation_type);

      -- Предотвращение дублирования одинаковых рёбер
      CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_relations_unique
        ON kg_relations(source_id, target_id, relation_type);
    `,
  },

  // ─── 604: Привязка сущностей к источникам ────────────────────────────────
  {
    version: 604,
    key: 'kb_main',
    description: 'Create kg_entity_sources table linking entities to document chunks',
    sql: `
      CREATE TABLE IF NOT EXISTS kg_entity_sources (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_id   INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
        library_id  INTEGER REFERENCES library(id) ON DELETE CASCADE,
        chunk_id    INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
        -- Ссылка на конкретный чанк документа (опционально)
        client_id   INTEGER NULL,
        -- Ссылка на клиента (из clients DB, без FK — разные базы)
        case_id     INTEGER NULL,
        -- Ссылка на дело
        span_start  INTEGER NULL,
        -- Позиция в тексте (для выделения)
        span_end    INTEGER NULL,
        context_snippet TEXT NULL,
        -- Короткая цитата из документа где найдена сущность
        extracted_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE INDEX IF NOT EXISTS idx_kg_sources_entity
        ON kg_entity_sources(entity_id);

      CREATE INDEX IF NOT EXISTS idx_kg_sources_library
        ON kg_entity_sources(library_id);

      CREATE INDEX IF NOT EXISTS idx_kg_sources_chunk
        ON kg_entity_sources(chunk_id);

      CREATE INDEX IF NOT EXISTS idx_kg_sources_client
        ON kg_entity_sources(client_id);

      CREATE INDEX IF NOT EXISTS idx_kg_sources_case
        ON kg_entity_sources(case_id);
    `,
  },

  // ─── 605: Журнал заданий извлечения ──────────────────────────────────────
  {
    version: 605,
    key: 'kb_main',
    description: 'Create kg_extraction_jobs table for tracking extraction tasks',
    sql: `
      CREATE TABLE IF NOT EXISTS kg_extraction_jobs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        library_id    INTEGER REFERENCES library(id) ON DELETE CASCADE,
        chunk_id      INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
        status        TEXT    NOT NULL DEFAULT 'pending',
        -- 'pending' | 'running' | 'done' | 'failed' | 'skipped'
        entities_found INTEGER NOT NULL DEFAULT 0,
        relations_found INTEGER NOT NULL DEFAULT 0,
        error_message TEXT    NULL,
        model_used    TEXT    NULL,
        duration_ms   INTEGER NULL,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        finished_at   DATETIME NULL
      );

      CREATE INDEX IF NOT EXISTS idx_kg_jobs_status
        ON kg_extraction_jobs(status, created_at DESC);

      CREATE INDEX IF NOT EXISTS idx_kg_jobs_library
        ON kg_extraction_jobs(library_id);
    `,
  },

  // ─── 606: ANALYZE после создания всех таблиц ─────────────────────────────
  {
    version: 606,
    key: 'kb_main',
    description: 'ANALYZE after Knowledge Graph table creation',
    sql: `ANALYZE;`,
  },
];
