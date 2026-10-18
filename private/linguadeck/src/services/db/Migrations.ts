// Schema versioning for migrations
// Each migration has a version number and SQL statements to execute

export interface Migration {
  version: number;
  name: string;
  sql: string[];
}

export const CURRENT_SCHEMA_VERSION = 4;

export const MIGRATIONS: Migration[] = [
  // Version 1: Initial schema
  {
    version: 1,
    name: "Initial schema",
    sql: [
      `PRAGMA foreign_keys = ON;`,
      `CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT DEFAULT CURRENT_TIMESTAMP
      );`,
      `INSERT OR IGNORE INTO schema_version (version) VALUES (1);`,
      `CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        avatar_path TEXT,
        last_login TEXT DEFAULT CURRENT_TIMESTAMP,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      );`,
      `CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL DEFAULT 1,
        name TEXT NOT NULL,
        emoji TEXT DEFAULT '🌍',
        progress INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
      );`,
      `CREATE TABLE IF NOT EXISTS decks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER NOT NULL DEFAULT 1,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        icon TEXT NOT NULL DEFAULT 'Layers3',
        color TEXT NOT NULL DEFAULT '#2563eb',
        cards_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
      );`,
      `CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        deck_id INTEGER NOT NULL,
        front TEXT NOT NULL,
        back TEXT NOT NULL,
        context TEXT,
        transcription TEXT,
        etymology TEXT,
        mnemonic TEXT,
        notes TEXT,
        example_1 TEXT,
        example_2 TEXT,
        example_3 TEXT,
        audio_path TEXT,
        image_path TEXT,
        video_path TEXT,
        source TEXT,
        tags TEXT,
        status TEXT NOT NULL DEFAULT 'new',
        due_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_reviewed_at TEXT,
        interval_days REAL NOT NULL DEFAULT 0,
        ease_factor REAL NOT NULL DEFAULT 2.5,
        repetitions INTEGER NOT NULL DEFAULT 0,
        lapses INTEGER NOT NULL DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (deck_id) REFERENCES decks(id) ON DELETE CASCADE
      );`,
      `CREATE TABLE IF NOT EXISTS review_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        deck_id INTEGER NOT NULL,
        rating INTEGER NOT NULL,
        interval_before REAL NOT NULL,
        interval_after REAL NOT NULL,
        ease_before REAL NOT NULL,
        ease_after REAL NOT NULL,
        reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(id) ON DELETE CASCADE,
        FOREIGN KEY (deck_id) REFERENCES decks(id) ON DELETE CASCADE
      );`,
      `CREATE TABLE IF NOT EXISTS view_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        card_id INTEGER NOT NULL,
        deck_id INTEGER NOT NULL,
        view_type TEXT NOT NULL DEFAULT 'view',
        viewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (card_id) REFERENCES cards(id) ON DELETE CASCADE,
        FOREIGN KEY (deck_id) REFERENCES decks(id) ON DELETE CASCADE
      );`,
      `CREATE TABLE IF NOT EXISTS app_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );`,
      `CREATE TABLE IF NOT EXISTS app_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level TEXT NOT NULL DEFAULT 'info',
        scope TEXT NOT NULL,
        message TEXT NOT NULL,
        details TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      );`,
      `CREATE INDEX IF NOT EXISTS idx_cards_deck_id ON cards(deck_id);`,
      `CREATE INDEX IF NOT EXISTS idx_cards_due_at ON cards(due_at);`,
      `CREATE INDEX IF NOT EXISTS idx_cards_deck_due ON cards(deck_id, due_at);`,
      `CREATE INDEX IF NOT EXISTS idx_reviews_reviewed_at ON review_logs(reviewed_at);`,
      `CREATE INDEX IF NOT EXISTS idx_logs_created_at ON app_logs(created_at);`,
      `INSERT OR IGNORE INTO users (id, name) VALUES (1, 'Learner');`,
      `INSERT OR IGNORE INTO courses (id, user_id, name, emoji) VALUES (1, 1, 'Main Library', '🧠');`
    ]
  },
  // Version 2: Add FTS5 search and reviewed_at column
  {
    version: 2,
    name: "FTS5 search and reviewed_at",
    sql: [
      `ALTER TABLE review_logs ADD COLUMN reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP;`,
      `CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(
        front,
        back,
        context,
        notes,
        tags,
        content='cards',
        content_rowid='id'
      );`,
      `CREATE TRIGGER IF NOT EXISTS cards_fts_insert AFTER INSERT ON cards BEGIN
        INSERT INTO cards_fts(rowid, front, back, context, notes, tags)
        VALUES (new.id, new.front, new.back, new.context, new.notes, new.tags);
      END;`,
      `CREATE TRIGGER IF NOT EXISTS cards_fts_delete AFTER DELETE ON cards BEGIN
        INSERT INTO cards_fts(cards_fts, rowid, front, back, context, notes, tags)
        VALUES ('delete', old.id, old.front, old.back, old.context, old.notes, old.tags);
      END;`,
      `CREATE TRIGGER IF NOT EXISTS cards_fts_update AFTER UPDATE ON cards BEGIN
        INSERT INTO cards_fts(cards_fts, rowid, front, back, context, notes, tags)
        VALUES ('delete', old.id, old.front, old.back, old.context, old.notes, old.tags);
        INSERT INTO cards_fts(rowid, front, back, context, notes, tags)
        VALUES (new.id, new.front, new.back, new.context, new.notes, new.tags);
      END;`,
      `INSERT OR REPLACE INTO cards_fts(rowid, front, back, context, notes, tags)
      SELECT id, front, back, context, notes, tags FROM cards;`,
      `INSERT OR REPLACE INTO schema_version (version) VALUES (2);`
    ]
  },
  // Version 3: Add time_spent_ms to view_logs for time-based analytics
  {
    version: 3,
    name: "view_logs time tracking",
    sql: [
      `ALTER TABLE view_logs ADD COLUMN time_spent_ms INTEGER NOT NULL DEFAULT 0;`,
      `CREATE INDEX IF NOT EXISTS idx_view_logs_card_id ON view_logs(card_id);`,
      `CREATE INDEX IF NOT EXISTS idx_view_logs_deck_id ON view_logs(deck_id);`,
      `CREATE INDEX IF NOT EXISTS idx_review_logs_card_id ON review_logs(card_id);`,
      `INSERT OR REPLACE INTO schema_version (version) VALUES (3);`
    ]
  },
  {
    version: 4,
    name: "behavioral study metrics and grammar module",
    sql: [
      `ALTER TABLE review_logs ADD COLUMN rating_source TEXT NOT NULL DEFAULT 'ignored';`,
      `ALTER TABLE review_logs ADD COLUMN front_time_ms INTEGER NOT NULL DEFAULT 0;`,
      `ALTER TABLE review_logs ADD COLUMN total_time_ms INTEGER NOT NULL DEFAULT 0;`,
      `ALTER TABLE review_logs ADD COLUMN flips_count INTEGER NOT NULL DEFAULT 0;`,
      `ALTER TABLE review_logs ADD COLUMN backtracks_count INTEGER NOT NULL DEFAULT 0;`,
      `CREATE TABLE IF NOT EXISTS grammar_units (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        language TEXT NOT NULL,
        level TEXT NOT NULL DEFAULT '',
        section TEXT NOT NULL DEFAULT '',
        tags TEXT,
        theory TEXT NOT NULL DEFAULT '',
        examples TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'not_started',
        mastery_streak INTEGER NOT NULL DEFAULT 0,
        last_score REAL,
        last_practiced_at TEXT,
        next_review_at TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
      );`,
      `CREATE TABLE IF NOT EXISTS grammar_exercises (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        unit_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        prompt TEXT NOT NULL,
        payload TEXT NOT NULL DEFAULT '{}',
        answer TEXT NOT NULL,
        explanation TEXT NOT NULL DEFAULT '',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (unit_id) REFERENCES grammar_units(id) ON DELETE CASCADE
      );`,
      `CREATE TABLE IF NOT EXISTS grammar_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        unit_id INTEGER NOT NULL,
        score REAL NOT NULL,
        correct_count INTEGER NOT NULL,
        total_count INTEGER NOT NULL,
        practiced_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (unit_id) REFERENCES grammar_units(id) ON DELETE CASCADE
      );`,
      `CREATE INDEX IF NOT EXISTS idx_grammar_units_status ON grammar_units(status);`,
      `CREATE INDEX IF NOT EXISTS idx_grammar_units_next_review ON grammar_units(next_review_at);`,
      `CREATE INDEX IF NOT EXISTS idx_grammar_exercises_unit ON grammar_exercises(unit_id);`,
      `CREATE INDEX IF NOT EXISTS idx_grammar_sessions_unit ON grammar_sessions(unit_id);`,
      `INSERT OR IGNORE INTO schema_version (version) VALUES (4);`
    ]
  }

];
