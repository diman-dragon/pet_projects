// db.js
// Отвечает только за сам механизм хранения: поднять sql.js, прочитать/записать
// файл базы в OPFS, дать generic run/query поверх sql.js. Про колоды/карточки
// (структура и бизнес-правила) здесь ничего не знает — это deckManager/cardManager.

window.App = window.App || {};

App.db = (function () {
  var DB_FILENAME = "flashcards.db";
  // Локальный вендор, не CDN: 1) не работает офлайн без него, сервис-воркер
  // его не закэширует автоматически 2) браузеры с Tracking Prevention (Edge)
  // блокируют сторонний storage-доступ для cdnjs.cloudflare.com и шумят в консоль.
  var SQLJS_BASE = "vendor/sql.js/";

  var sqlJsLib = null;
  var dbInstance = null;
  var opfsFileHandle = null;
  var readyPromise = null;
  var DEVICE_KEY = "flashcards.deviceId";
  var deviceId = localStorage.getItem(DEVICE_KEY) || ((crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2)) + "-" + Date.now());
  localStorage.setItem(DEVICE_KEY, deviceId);

  function loadSqlJsScript() {
    return new Promise(function (resolve, reject) {
      if (window.initSqlJs) return resolve(window.initSqlJs);
      var script = document.createElement("script");
      script.src = SQLJS_BASE + "sql-wasm.js";
      script.onload = function () { resolve(window.initSqlJs); };
      script.onerror = function () { reject(new Error("Не удалось загрузить sql.js")); };
      document.head.appendChild(script);
    });
  }

  function assertOpfsSupported() {
    if (!("storage" in navigator) || !navigator.storage.getDirectory) {
      throw new Error(
        "Браузер не поддерживает OPFS (navigator.storage.getDirectory). " +
        "Нужен современный Chromium/Edge/Safari."
      );
    }
  }

  // Читает существующий файл базы из OPFS. Возвращает null, если файла ещё нет
  // (первый запуск) — тогда создаётся пустая база.
  async function loadBytesFromOpfs() {
    var root = await navigator.storage.getDirectory();
    opfsFileHandle = await root.getFileHandle(DB_FILENAME, { create: true });
    var file = await opfsFileHandle.getFile();
    if (file.size === 0) return null;
    var buffer = await file.arrayBuffer();
    return new Uint8Array(buffer);
  }

  // Сохраняет базу с межвкладочным lock. Сначала пишем временный файл и,
  // если браузер поддерживает FileSystemFileHandle.move(), заменяем основной
  // файл одним файловым действием. На старых реализациях OPFS остаётся безопасный
  // fallback под тем же lock.
  async function persistUnlocked() {
    if (!dbInstance || !opfsFileHandle) return;
    var bytes = dbInstance.export();
    var root = await navigator.storage.getDirectory();
    var tempName = DB_FILENAME + ".tmp";
    var tempHandle = await root.getFileHandle(tempName, { create: true });
    var writable = await tempHandle.createWritable();
    try {
      await writable.write(bytes);
      await writable.close();
    } catch (err) {
      try { await writable.abort(); } catch (e) {}
      throw err;
    }
    if (typeof tempHandle.move === "function") {
      await tempHandle.move(DB_FILENAME);
      opfsFileHandle = await root.getFileHandle(DB_FILENAME, { create: false });
    } else {
      var direct = await opfsFileHandle.createWritable();
      try { await direct.write(bytes); await direct.close(); }
      catch (err2) { try { await direct.abort(); } catch (e2) {} throw err2; }
      try { await root.removeEntry(tempName); } catch (e3) {}
    }
  }

  async function persist() {
    if (!dbInstance || !opfsFileHandle) return;
    if (navigator.locks && navigator.locks.request) {
      return navigator.locks.request("flashcards-db-write", { mode: "exclusive" }, function () {
        return persistUnlocked();
      });
    }
    return persistUnlocked();
  }

  function decksTableIsEmpty() {
    var res = dbInstance.exec("SELECT COUNT(*) AS n FROM decks;");
    return !res.length || res[0].values[0][0] === 0;
  }

  // Есть ли уже колонка в таблице — читаем PRAGMA table_info, а не ловим исключение
  // от ALTER TABLE (в sql.js текст ошибки не такой стабильный, чтобы на него полагаться).
  function hasColumn(table, col) {
    var res = dbInstance.exec("PRAGMA table_info(" + table + ");");
    return !!(res.length && res[0].values.some(function (row) { return row[1] === col; }));
  }

  // Добавляет колонки SRS в базы, созданные ДО этой фичи (CREATE TABLE IF NOT EXISTS
  // их не тронет — таблица уже существует). Идемпотентно: на новых базах, где схема
  // и так уже содержит эти колонки, hasColumn просто вернёт true и ничего не делает.
  function migrateHierarchy() {
    // Старые базы не знают о направлениях. Создаём нейтральное направление
    // и переносим в него существующие колоды.
    dbInstance.run(
      "CREATE TABLE IF NOT EXISTS directions (" +
      "id TEXT PRIMARY KEY, name TEXT NOT NULL, position INTEGER NOT NULL DEFAULT 0" +
      ");"
    );
    var directionId = "direction-general";
    var dirs = dbInstance.exec("SELECT id FROM directions LIMIT 1;");
    if (!dirs.length || !dirs[0].values.length) {
      dbInstance.run(
        "INSERT INTO directions (id, name, position) VALUES (?, ?, 0);",
        [directionId, "Общее"]
      );
    } else {
      directionId = dirs[0].values[0][0];
    }

    if (!hasColumn("decks", "direction_id")) {
      dbInstance.run("ALTER TABLE decks ADD COLUMN direction_id TEXT;");
    }
    dbInstance.run(
      "UPDATE decks SET direction_id = ? WHERE direction_id IS NULL OR direction_id = '';",
      [directionId]
    );
    dbInstance.run("CREATE INDEX IF NOT EXISTS idx_decks_direction ON decks(direction_id, position);");
  }

  function migrateSyncColumns() {
    [["directions","updated_at","updated_at INTEGER NOT NULL DEFAULT 0"],["directions","updated_by","updated_by TEXT NOT NULL DEFAULT ''"],
     ["decks","updated_at","updated_at INTEGER NOT NULL DEFAULT 0"],["decks","updated_by","updated_by TEXT NOT NULL DEFAULT ''"],
     ["cards","updated_at","updated_at INTEGER NOT NULL DEFAULT 0"],["cards","updated_by","updated_by TEXT NOT NULL DEFAULT ''"]].forEach(function (c) {
      if (!hasColumn(c[0], c[1])) dbInstance.run("ALTER TABLE " + c[0] + " ADD COLUMN " + c[2] + ";");
    });
    dbInstance.run("CREATE TABLE IF NOT EXISTS sync_tombstones (entity_type TEXT NOT NULL, entity_id TEXT NOT NULL, updated_at INTEGER NOT NULL, updated_by TEXT NOT NULL, PRIMARY KEY(entity_type, entity_id));");
    dbInstance.run("CREATE INDEX IF NOT EXISTS idx_sync_tombstones_time ON sync_tombstones(updated_at);");
    var now = Date.now();
    ["directions","decks","cards"].forEach(function (t) {
      dbInstance.run("UPDATE " + t + " SET updated_at = ?, updated_by = ? WHERE updated_at = 0;", [now, deviceId]);
    });
  }

  function migrateSrsColumns() {
    [
      ["due", "due INTEGER NOT NULL DEFAULT 0"],
      ["interval_days", "interval_days INTEGER NOT NULL DEFAULT 0"],
      ["ease", "ease REAL NOT NULL DEFAULT 2.5"],
      ["reps", "reps INTEGER NOT NULL DEFAULT 0"],
      ["lapses", "lapses INTEGER NOT NULL DEFAULT 0"],
      ["show_count", "show_count INTEGER NOT NULL DEFAULT 0"],
      ["learned", "learned INTEGER NOT NULL DEFAULT 0"]
    ].forEach(function (col) {
      if (!hasColumn("cards", col[0])) dbInstance.run("ALTER TABLE cards ADD COLUMN " + col[1] + ";");
    });
  }

  function migrateStats() {
    dbInstance.run(
      "CREATE TABLE IF NOT EXISTS study_events (" +
      "id INTEGER PRIMARY KEY AUTOINCREMENT," +
      "event_uid TEXT UNIQUE," +
      "card_id TEXT, deck_id TEXT, event_type TEXT NOT NULL, rating TEXT, created_at INTEGER NOT NULL" +
      ");"
    );
    if (!hasColumn("study_events", "event_uid")) dbInstance.run("ALTER TABLE study_events ADD COLUMN event_uid TEXT;");
    var ev = dbInstance.exec("SELECT id FROM study_events WHERE event_uid IS NULL OR event_uid = '';");
    if (ev.length) ev[0].values.forEach(function (r) { dbInstance.run("UPDATE study_events SET event_uid = ? WHERE id = ?;", [deviceId + "-legacy-" + r[0], r[0]]); });
    dbInstance.run("CREATE UNIQUE INDEX IF NOT EXISTS idx_study_events_uid ON study_events(event_uid);");
    dbInstance.run("CREATE INDEX IF NOT EXISTS idx_study_events_time ON study_events(created_at);");
    dbInstance.run("CREATE INDEX IF NOT EXISTS idx_study_events_card ON study_events(card_id, created_at);");
  }

  function getUserVersion() {
    var res = dbInstance.exec("PRAGMA user_version;");
    return res.length ? res[0].values[0][0] : 0;
  }

  // Инициализация — идемпотентна, можно звать из нескольких мест,
  // реальная работа выполнится один раз (see readyPromise).
  function init() {
    if (readyPromise) return readyPromise;

    readyPromise = (async function () {
      assertOpfsSupported();

      var initSqlJsFn = await loadSqlJsScript();
      sqlJsLib = await initSqlJsFn({
        locateFile: function (file) { return SQLJS_BASE + file; }
      });

      var existingBytes = await loadBytesFromOpfs();
      dbInstance = existingBytes
        ? new sqlJsLib.Database(existingBytes)
        : new sqlJsLib.Database();

      dbInstance.run(App.schema.CREATE_TABLES_SQL);
      migrateHierarchy();
      migrateSyncColumns();
      migrateSrsColumns();
      migrateStats();
      await persist();

      // Мок-данные заливаем ТОЛЬКО при самом первом запуске. Признак — PRAGMA user_version
      // (лежит в файле базы). Проверка «нет колод» здесь не годится: пользователь теперь
      // может удалить все колоды, и они не должны воскресать после перезагрузки.
      if (getUserVersion() === 0) {
        if (decksTableIsEmpty()) {
          App.seed.insertInto(dbInstance);
          console.log("[db] Первый запуск — залил мок-данные из seed.js");
        }
        dbInstance.run("PRAGMA user_version = 1;");
        await persist();
      }

      console.log("[db] Готово. Файл базы: OPFS/" + DB_FILENAME);
      return dbInstance;
    })();

    return readyPromise;
  }

  // SELECT с параметрами → массив обычных объектов ({colName: value, ...}).
  function query(sql, params) {
    var stmt = dbInstance.prepare(sql);
    if (params) stmt.bind(params);
    var rows = [];
    while (stmt.step()) rows.push(stmt.getAsObject());
    stmt.free();
    return rows;
  }

  // INSERT/UPDATE/DELETE. Сама сохраняет базу в OPFS после мутации —
  // вызывающему коду не нужно помнить об этом отдельно.
  async function run(sql, params) {
    dbInstance.run(sql, params || []);
    await persist();
  }

  // Один и тот же INSERT/UPDATE для пачки строк в одной транзакции и с ОДНИМ
  // сохранением в OPFS в конце (run() в цикле экспортировал бы всю базу на каждую строку).
  // Ошибка на любой строке → ROLLBACK, в базе не остаётся ничего.
  async function runMany(sql, paramsList) {
    if (!paramsList.length) return;
    dbInstance.run("BEGIN;");
    try {
      var stmt = dbInstance.prepare(sql);
      try {
        paramsList.forEach(function (params) { stmt.run(params); });
      } finally {
        stmt.free();
      }
      dbInstance.run("COMMIT;");
    } catch (err) {
      try { dbInstance.run("ROLLBACK;"); } catch (e) { /* транзакция уже закрыта */ }
      throw err;
    }
    await persist();
  }

  // Несколько РАЗНЫХ запросов в одной транзакции: statements = [[sql, params], ...].
  // Всё или ничего; сохранение в OPFS — один раз в конце.
  async function transaction(statements) {
    dbInstance.run("BEGIN;");
    try {
      statements.forEach(function (st) { dbInstance.run(st[0], st[1] || []); });
      dbInstance.run("COMMIT;");
    } catch (err) {
      try { dbInstance.run("ROLLBACK;"); } catch (e) { /* транзакция уже закрыта */ }
      throw err;
    }
    await persist();
  }

  function getInstance() {
    return dbInstance;
  }

  return {
    init: init,
    query: query,
    run: run,
    runMany: runMany,
    transaction: transaction,
    persist: persist,
    getInstance: getInstance,
    getDeviceId: function () { return deviceId; }
  };
})();
