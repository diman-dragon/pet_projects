// src/constants/schemas/KbSchema.ts
//
// RAG-SCHEMA v3.1
//
// Изменения относительно v2:
//   library      — добавлены lang TEXT NULL, folder_path TEXT NULL, chunker_version INTEGER DEFAULT 0
//   library_fts  — новая FTS5-таблица по title + content_raw + metadata
//   search_selections — выбранные пользователем результаты (prior для ранжирования)
//   category_rules    — правила авто-категоризации по пути/расширению
//   document_languages — детализация языков на уровне документа/чанка
//
// АРХИТЕКТУРНЫЙ КОНТРАКТ:
//   Три зоны поиска: kb_main | clients_main | user_system.
//   GlobalSearchService и FederatedOrchestrator работают поверх них
//   как над независимыми источниками.

export const KB_SCHEMA = `
  -- ── 1. Мастер-запись документа ─────────────────────────────────────────────
  CREATE TABLE IF NOT EXISTS library (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT    NOT NULL,
    file_path       TEXT    UNIQUE NOT NULL,
    file_type       TEXT    NOT NULL DEFAULT 'txt',   -- txt | pdf | docx | md
    category        TEXT    NOT NULL DEFAULT 'General',
    lang            TEXT    NULL,                      -- ISO-639-1: ru | en | ... (NULL = не определён)
    folder_path     TEXT    NULL,                      -- нормализованный путь папки для категоризации
    chunker_version INTEGER DEFAULT 0,               -- версия чанкера для force reindex
    content_raw     TEXT,                              -- полный текст (для AI-контекста)
    metadata        JSON,                              -- {author?, court?, year?, tags?, category_overridden?, orphaned?}
    hash            TEXT,                              -- SHA-256 контента (для детекции изменений)
    indexed_at      DATETIME,                          -- когда последний раз проиндексирован
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  -- ── 2. Чанки документа ──────────────────────────────────────────────────────
  CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    library_id  INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    char_start  INTEGER,
    char_end    INTEGER,
    parent_id   INTEGER, -- NULL для родительских или обычных чанков
    article_ref TEXT,    -- извлеченная ссылка на статью (ст. 450 ГК)
    FOREIGN KEY(library_id) REFERENCES library(id) ON DELETE CASCADE,
    FOREIGN KEY(parent_id)  REFERENCES chunks(id)  ON DELETE SET NULL,
    UNIQUE(library_id, chunk_index)
  );

  CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
  USING fts5(
    text,
    content=chunks,
    content_rowid=id,
    tokenize='unicode61 remove_diacritics 1'
  );

  -- ── 3. FTS5 по мастер-записям библиотеки ────────────────────────────────────
  -- Позволяет искать по заголовку/тексту/метаданным без чанков.
  -- Первая фаза GlobalSearchService (быстрый ответ до эмбеддингов).
  CREATE VIRTUAL TABLE IF NOT EXISTS library_fts
  USING fts5(
    title,
    content_raw,
    metadata,
    content=library,
    content_rowid=id,
    tokenize='unicode61 remove_diacritics 1'
  );

  -- ── 4. Векторные эмбеддинги ──────────────────────────────────────────────────
  CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY,
    chunk_id    INTEGER NOT NULL UNIQUE,
    library_id  INTEGER NOT NULL,
    vector_blob BLOB    NOT NULL,
    model       TEXT    NOT NULL,
    dim         INTEGER NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(chunk_id)   REFERENCES chunks(id)   ON DELETE CASCADE,
    FOREIGN KEY(library_id) REFERENCES library(id)  ON DELETE CASCADE
  );

  CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings
  USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    library_id INTEGER,
    model TEXT,
    embedding FLOAT[1024]
  );

  -- ── 5. Журнал задач индексации ──────────────────────────────────────────────
  CREATE TABLE IF NOT EXISTS index_jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    library_id   INTEGER NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending',
    total_chunks INTEGER DEFAULT 0,
    done_chunks  INTEGER DEFAULT 0,
    error_msg    TEXT,
    started_at   DATETIME,
    finished_at  DATETIME,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(library_id) REFERENCES library(id) ON DELETE CASCADE
  );

  -- ── 6. Выборки пользователя (prior для ранжирования) ───────────────────────
  CREATE TABLE IF NOT EXISTS search_selections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash  TEXT    NOT NULL,           -- SHA-1 нормализованного запроса
    library_id  INTEGER,                    -- NULL для виртуальных документов
    result_id   TEXT    NOT NULL,           -- глобальный id результата (e.g. "lib-42", "client-7")
    selected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(library_id) REFERENCES library(id) ON DELETE CASCADE
  );

  CREATE INDEX IF NOT EXISTS idx_selections_query ON search_selections(query_hash, selected_at DESC);

  -- ── 7. Правила категоризации ────────────────────────────────────────────────
  -- Применяются FolderCategorizerService при индексации и при перемещении файлов.
  -- Порядок приоритетов: path_pattern (точный) > extension > default.
  CREATE TABLE IF NOT EXISTS category_rules (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type    TEXT    NOT NULL,   -- 'path_pattern' | 'extension' | 'default'
    pattern      TEXT    NOT NULL,  -- glob/regex для path_pattern; расширение для extension
    category     TEXT    NOT NULL,
    priority     INTEGER NOT NULL DEFAULT 0,
    is_system    INTEGER NOT NULL DEFAULT 0,  -- 1 = не трогать пользователю
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE INDEX IF NOT EXISTS idx_category_rules_type ON category_rules(rule_type, priority DESC);

  -- ── 8. Языки документов ─────────────────────────────────────────────────────
  -- Детализация на уровне документа и чанка для кросс-языкового retrieval.
  CREATE TABLE IF NOT EXISTS document_languages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    library_id  INTEGER NOT NULL,
    chunk_id    INTEGER,            -- NULL = язык всего документа
    lang        TEXT    NOT NULL,   -- ISO-639-1
    confidence  REAL    NOT NULL DEFAULT 1.0,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(library_id) REFERENCES library(id) ON DELETE CASCADE,
    FOREIGN KEY(chunk_id)   REFERENCES chunks(id)  ON DELETE CASCADE
  );

  CREATE INDEX IF NOT EXISTS idx_doc_lang_library ON document_languages(library_id);

  -- ── 9. История поисковых запросов ───────────────────────────────────────────
  CREATE TABLE IF NOT EXISTS search_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query        TEXT    NOT NULL,
    query_hash   TEXT,
    query_lang   TEXT,
    query_type   TEXT,
    query_vector JSON,
    model        TEXT,
    vector_hits  INTEGER DEFAULT 0,
    text_hits    INTEGER DEFAULT 0,
    latency_ms   INTEGER,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at DESC);
  CREATE INDEX IF NOT EXISTS idx_search_history_hash    ON search_history(query_hash);

  -- ── Индексы ──────────────────────────────────────────────────────────────────
  CREATE INDEX IF NOT EXISTS idx_lib_file_path           ON library(file_path);
  CREATE INDEX IF NOT EXISTS idx_lib_category            ON library(category);
  CREATE INDEX IF NOT EXISTS idx_lib_hash                ON library(hash);
  CREATE INDEX IF NOT EXISTS idx_lib_category_lang       ON library(category, lang, created_at DESC);
  CREATE INDEX IF NOT EXISTS idx_lib_folder              ON library(folder_path);
  CREATE INDEX IF NOT EXISTS idx_lib_updated             ON library(updated_at DESC);
  CREATE INDEX IF NOT EXISTS idx_chunks_lib              ON chunks(library_id);
  CREATE INDEX IF NOT EXISTS idx_chunks_parent           ON chunks(parent_id);
  CREATE INDEX IF NOT EXISTS idx_emb_lib                 ON embeddings(library_id);
  CREATE INDEX IF NOT EXISTS idx_emb_chunk               ON embeddings(chunk_id);
  CREATE INDEX IF NOT EXISTS idx_jobs_lib                ON index_jobs(library_id);
  CREATE INDEX IF NOT EXISTS idx_jobs_status             ON index_jobs(status);

  -- ── Триггеры синхронизации chunks_fts ────────────────────────────────────────
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

  -- ── Триггеры синхронизации library_fts ───────────────────────────────────────
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
`;
