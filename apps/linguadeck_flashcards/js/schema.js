// schema.js
// SQL-структура: направление → колода → карточка.
window.App = window.App || {};
App.schema = (function () {
  var CREATE_TABLES_SQL =
    "CREATE TABLE IF NOT EXISTS directions (" +
    "id TEXT PRIMARY KEY, name TEXT NOT NULL, position INTEGER NOT NULL DEFAULT 0, " +
    "updated_at INTEGER NOT NULL DEFAULT 0, updated_by TEXT NOT NULL DEFAULT ''" +
    ");" +
    "CREATE TABLE IF NOT EXISTS decks (" +
    "id TEXT PRIMARY KEY, direction_id TEXT NOT NULL REFERENCES directions(id), name TEXT NOT NULL, position INTEGER NOT NULL DEFAULT 0, " +
    "updated_at INTEGER NOT NULL DEFAULT 0, updated_by TEXT NOT NULL DEFAULT ''" +
    ");" +
    "CREATE TABLE IF NOT EXISTS cards (" +
    "id TEXT PRIMARY KEY, deck_id TEXT NOT NULL REFERENCES decks(id) ON DELETE CASCADE, position INTEGER NOT NULL DEFAULT 0, " +
    "front TEXT NOT NULL, back TEXT NOT NULL, transcription TEXT, image TEXT, audio TEXT, video TEXT, examples TEXT, " +
    "due INTEGER NOT NULL DEFAULT 0, interval_days INTEGER NOT NULL DEFAULT 0, ease REAL NOT NULL DEFAULT 2.5, reps INTEGER NOT NULL DEFAULT 0, lapses INTEGER NOT NULL DEFAULT 0, show_count INTEGER NOT NULL DEFAULT 0, learned INTEGER NOT NULL DEFAULT 0, " +
    "updated_at INTEGER NOT NULL DEFAULT 0, updated_by TEXT NOT NULL DEFAULT ''" +
    ");" +
    "CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(deck_id, position);";
  return { CREATE_TABLES_SQL: CREATE_TABLES_SQL };
})();
