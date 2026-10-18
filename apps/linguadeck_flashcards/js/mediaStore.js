// mediaStore.js
// Хранилище пользовательских медиафайлов (картинки/аудио карточек, добавленных руками).
//
// Зачем: на GitHub Pages нет бэкенда, в статическую папку media/ из браузера писать
// нельзя. Поэтому такие файлы лежат в OPFS по ТОМУ ЖЕ пути, что и в базе
// ("media/<deckId>/<file>"), а resolve() подменяет путь на blob:-URL, если файл
// найден в OPFS. Пути мок-данных из репозитория в OPFS не найдутся и вернутся как есть —
// render.js/app.js разницы не видят.

window.App = window.App || {};

App.mediaStore = (function () {
  var ROOT_DIR = "media";

  // "media/<deckId>/<file>" → blob:URL. Заполняется в init() и в put().
  var urlCache = {};

  var MIME_BY_EXT = {
    jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", webp: "image/webp",
    gif: "image/gif", avif: "image/avif", svg: "image/svg+xml", bmp: "image/bmp",
    mp3: "audio/mpeg", ogg: "audio/ogg", oga: "audio/ogg", opus: "audio/ogg",
    wav: "audio/wav", m4a: "audio/mp4", aac: "audio/aac", flac: "audio/flac",
    weba: "audio/webm", webm: "audio/webm"
  };

  function isSupported() {
    return !!(navigator.storage && navigator.storage.getDirectory);
  }

  function extOf(name) {
    var m = /\.([a-z0-9]+)$/i.exec(name || "");
    return m ? m[1].toLowerCase() : "";
  }

  // OPFS у файлов часто отдаёт пустой type, а Safari без корректного типа
  // не проигрывает blob-аудио — поэтому тип выставляем сами по расширению.
  function toUrl(file, path) {
    var type = MIME_BY_EXT[extOf(path)] || file.type || "";
    return URL.createObjectURL(new Blob([file], { type: type }));
  }

  // "media/deck/x.jpg" → ["media","deck","x.jpg"], с защитой от "..", "" и обратных слэшей
  function splitPath(path) {
    var parts = String(path || "").split("/");
    parts.forEach(function (p) {
      if (!p || p === "." || p === ".." || p.indexOf("\\") !== -1 || p.indexOf("\0") !== -1) {
        throw new Error("Недопустимый путь к медиафайлу: " + path);
      }
    });
    if (parts[0] !== ROOT_DIR || parts.length < 2) {
      throw new Error("Путь должен начинаться с " + ROOT_DIR + "/: " + path);
    }
    return parts;
  }

  async function getParentDir(parts, create) {
    var dir = await navigator.storage.getDirectory();
    for (var i = 0; i < parts.length - 1; i++) {
      dir = await dir.getDirectoryHandle(parts[i], { create: create });
    }
    return dir;
  }

  async function scan(dirHandle, prefix) {
    for await (var entry of dirHandle.values()) {
      var path = prefix + "/" + entry.name;
      if (entry.kind === "directory") {
        await scan(entry, path);
      } else {
        urlCache[path] = toUrl(await entry.getFile(), path);
      }
    }
  }

  // Вызывать после db.init(), до первой отрисовки. Никогда не бросает:
  // нет OPFS или папки media/ — значит пользовательских файлов просто нет.
  async function init() {
    if (!isSupported()) return;
    try {
      var root = await navigator.storage.getDirectory();
      var mediaDir = await root.getDirectoryHandle(ROOT_DIR); // без create — папки может не быть
      await scan(mediaDir, ROOT_DIR);
    } catch (err) {
      if (!err || err.name !== "NotFoundError") {
        console.warn("[mediaStore] Не удалось прочитать OPFS/" + ROOT_DIR + ":", err);
      }
    }
  }

  async function read(path) {
    if (!isSupported() || !path) return null;
    var parts = splitPath(path);
    try {
      var root = await navigator.storage.getDirectory();
      var dir = await getParentDir(parts, false);
      var handle = await dir.getFileHandle(parts[parts.length - 1], { create: false });
      return handle.getFile();
    } catch (err) { return null; }
  }

  // Синхронно: путь из базы → URL, который можно отдать в <img>/Audio.
  function resolve(path) {
    if (!path) return path;
    return urlCache[path] || path;
  }

  // Записывает файл в OPFS по path (перезаписывает, если уже есть).
  async function put(path, file) {
    if (!isSupported()) throw new Error("Браузер не поддерживает OPFS — файл сохранить некуда.");
    var parts = splitPath(path);
    var dir = await getParentDir(parts, true);
    var handle = await dir.getFileHandle(parts[parts.length - 1], { create: true });
    var writable = await handle.createWritable();
    try {
      await writable.write(file);
      await writable.close();
    } catch (err) {
      try { await writable.abort(); } catch (e) { /* уже закрыт */ }
      throw err;
    }
    if (urlCache[path]) URL.revokeObjectURL(urlCache[path]);
    urlCache[path] = toUrl(await handle.getFile(), path);
    return path;
  }

  // Удаляет файл. Ошибки глушит: используется и для отката после неудачной вставки.
  async function remove(path) {
    try {
      var parts = splitPath(path);
      var dir = await getParentDir(parts, false);
      await dir.removeEntry(parts[parts.length - 1]);
    } catch (err) {
      if (!err || err.name !== "NotFoundError") console.warn("[mediaStore] remove:", path, err);
    }
    if (urlCache[path]) {
      URL.revokeObjectURL(urlCache[path]);
      delete urlCache[path];
    }
  }

  // Удаляет ВСЮ папку media/<name>/ (при удалении колоды) и её blob-URL.
  async function removeDir(name) {
    var prefix = ROOT_DIR + "/" + name + "/";
    splitPath(prefix + "x"); // только валидация имени
    try {
      var root = await navigator.storage.getDirectory();
      var mediaDir = await root.getDirectoryHandle(ROOT_DIR);
      await mediaDir.removeEntry(name, { recursive: true });
    } catch (err) {
      if (!err || err.name !== "NotFoundError") console.warn("[mediaStore] removeDir:", name, err);
    }
    Object.keys(urlCache).forEach(function (p) {
      if (p.indexOf(prefix) === 0) {
        URL.revokeObjectURL(urlCache[p]);
        delete urlCache[p];
      }
    });
  }

  return {
    init: init,
    resolve: resolve,
    put: put,
    read: read,
    remove: remove,
    removeDir: removeDir,
    isSupported: isSupported,
    extOf: extOf,
    MIME_BY_EXT: MIME_BY_EXT
  };
})();
