// addCard.js
// Добавление ОДНОЙ карточки в колоду: текст + опционально файл картинки и/или аудио.
// Ничего не знает про DOM: получает готовые значения, кидает Error с русским текстом,
// который диалог показывает как есть.

window.App = window.App || {};

App.addCard = (function () {
  var MAX_FILE_BYTES = 20 * 1024 * 1024;
  var KIND_LABEL = { image: "изображение", audio: "аудио" };

  // Когда у файла нет расширения — берём из MIME
  var EXT_BY_MIME_SUBTYPE = { jpeg: "jpg", mpeg: "mp3", mp4: "m4a", "x-m4a": "m4a", "x-wav": "wav", "svg+xml": "svg" };

  function clean(value) {
    return (value == null ? "" : String(value)).trim();
  }

  function cleanList(items) {
    return (items || []).map(clean).filter(Boolean);
  }

  function matchesKind(file, kind) {
    if (file.type && file.type.indexOf(kind + "/") === 0) return true;
    // type бывает пустым (или video/webm у аудио-записи) — смотрим на расширение
    var byExt = App.mediaStore.MIME_BY_EXT[App.mediaStore.extOf(file.name)] || "";
    return byExt.indexOf(kind + "/") === 0;
  }

  function extFor(file) {
    var ext = App.mediaStore.extOf(file.name);
    if (ext && App.mediaStore.MIME_BY_EXT[ext]) return ext;
    var sub = (file.type || "").split("/")[1] || "";
    sub = EXT_BY_MIME_SUBTYPE[sub] || sub.replace(/^x-/, "").replace(/[^a-z0-9]/gi, "").toLowerCase();
    return sub || "bin";
  }

  function checkFile(file, kind) {
    if (!matchesKind(file, kind)) {
      throw new Error("Файл «" + file.name + "» не похож на " + KIND_LABEL[kind] + ".");
    }
    if (file.size > MAX_FILE_BYTES) {
      throw new Error("Файл «" + file.name + "» слишком большой (максимум " + (MAX_FILE_BYTES >> 20) + " МБ).");
    }
    if (file.size === 0) {
      throw new Error("Файл «" + file.name + "» пустой.");
    }
  }

  // Единое правило именования файлов: media/<deckId>/<cardId>-<image|audio>.<ext>
  function mediaPath(deckId, cardId, kind, file) {
    return "media/" + deckId + "/" + cardId + "-" + kind + "." + extFor(file);
  }

  // input: { front, back, transcription?, examples?: string[], imageFile?: File, audioFile?: File }
  // Возвращает { id }. При любой ошибке уже записанные в OPFS файлы удаляются.
  async function add(deckId, input) {
    if (!deckId) throw new Error("Не выбрана колода.");

    var front = clean(input.front);
    var back = clean(input.back);
    if (!front) throw new Error("Заполните лицевую сторону (термин).");
    if (!back) throw new Error("Заполните обратную сторону (перевод / определение).");

    var files = [];
    if (input.imageFile) files.push({ kind: "image", file: input.imageFile });
    if (input.audioFile) files.push({ kind: "audio", file: input.audioFile });
    files.forEach(function (f) { checkFile(f.file, f.kind); });

    var id = App.cardManager.newId(deckId);
    var card = {
      id: id,
      front: front,
      back: back,
      transcription: clean(input.transcription) || undefined,
      examples: cleanList(input.examples)
    };

    var savedPaths = [];
    try {
      for (var i = 0; i < files.length; i++) {
        var f = files[i];
        var path = mediaPath(deckId, id, f.kind, f.file);
        await App.mediaStore.put(path, f.file);
        savedPaths.push(path);
        card[f.kind] = path;
      }
      await App.cardManager.addCards(deckId, [card]);
    } catch (err) {
      await Promise.all(savedPaths.map(function (p) { return App.mediaStore.remove(p); }));
      throw err;
    }
    return { id: id };
  }

  return {
    add: add,
    // общие с editCard.js
    checkFile: checkFile,
    mediaPath: mediaPath
  };
})();
