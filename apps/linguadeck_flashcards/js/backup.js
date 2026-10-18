// backup.js
// Экспорт колод в JSON-файл (со встроенными медиа в base64 — на GitHub Pages
// нет бэкенда, поэтому zip с отдельными файлами усложнил бы дело без выгоды)
// и разбор + запись обратно такого файла. Формат один и тот же для экспорта
// одной колоды и полного бэкапа — разница только в длине decks[].

window.App = window.App || {};

App.backup = (function () {
  var SCHEMA_VERSION = 3;
  var MAX_IMPORT_BYTES = 80 * 1024 * 1024; // файл текстовый (base64), но может быть большим

  function clean(v) { return (v == null ? "" : String(v)).trim(); }
  function basename(path) { return String(path || "").split("/").pop(); }

  // ---------- экспорт ----------

  async function urlToBase64(url) {
    var res = await fetch(url);
    if (!res.ok) throw new Error("не удалось прочитать файл (HTTP " + res.status + ")");
    var buf = await (await res.blob()).arrayBuffer();
    var bytes = new Uint8Array(buf);
    var binary = "";
    var CHUNK = 0x8000; // apply() с огромным массивом аргументов падает — режем на куски
    for (var i = 0; i < bytes.length; i += CHUNK) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    return btoa(binary);
  }

  // rawPath — как хранится в базе ("media/<deckId>/<file>" или null).
  async function mediaField(rawPath) {
    if (!rawPath) return null;
    var name = basename(rawPath);
    var type = App.mediaStore.MIME_BY_EXT[App.mediaStore.extOf(name)] || "application/octet-stream";
    var data = await urlToBase64(App.mediaStore.resolve(rawPath));
    return { name: name, type: type, data: data };
  }

  async function deckToJson(deck) {
    var raws = App.cardManager.getRawForDeck(deck.id);
    var cards = [];
    for (var i = 0; i < raws.length; i++) {
      var r = raws[i];
      cards.push({
        front: r.front,
        back: r.back,
        transcription: r.transcription || undefined,
        examples: (function(){ try { return r.examples ? JSON.parse(r.examples) : undefined; } catch(e) { return undefined; } })(),
        due: r.due || 0,
        interval: r.interval_days || 0,
        ease: r.ease == null ? 2.5 : r.ease,
        reps: r.reps || 0,
        lapses: r.lapses || 0,
        showCount: r.show_count || 0,
        learned: !!r.learned,
        image: await mediaField(r.image),
        audio: await mediaField(r.audio)
      });
    }
    var direction = App.deckManager.getDirections().find(function (d) { return d.id === deck.directionId; });
    return { direction: direction ? direction.name : "Общее", name: deck.name, cards: cards };
  }

  function safeFilename(name) {
    return name.replace(/[\\/:*?"<>|]+/g, "_").slice(0, 80);
  }

  function download(payload, filenameBase) {
    var blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = safeFilename(filenameBase) + ".json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
  }

  async function exportDeck(deck) {
    download({
      app: "flashcards-app",
      schemaVersion: SCHEMA_VERSION,
      exportedAt: new Date().toISOString(),
      decks: [await deckToJson(deck)]
    }, deck.name);
  }

  async function exportAll(decks) {
    var jsonDecks = [];
    for (var i = 0; i < decks.length; i++) jsonDecks.push(await deckToJson(decks[i]));
    download({
      app: "flashcards-app",
      schemaVersion: SCHEMA_VERSION,
      exportedAt: new Date().toISOString(),
      decks: jsonDecks,
      studyEvents: App.db.query("SELECT card_id, deck_id, event_type, rating, created_at FROM study_events ORDER BY id;")
    }, "flashcards-backup-" + new Date().toISOString().slice(0, 10));
  }

  // ---------- импорт: разбор файла (без обращений к базе) ----------

  function isValidMediaSpec(m) {
    return !!m && typeof m === "object" &&
      typeof m.data === "string" && m.data.length > 0 &&
      typeof m.type === "string" && /^(image|audio)\//.test(m.type);
  }

  // Читает и проверяет файл. Ничего не пишет в базу — только разбор с валидацией,
  // как batchAdd.parse(): плохие карточки/колоды пропускаются с причиной, а не рушат весь импорт.
  // → { decks: [{ name, cards: [{front,back,transcription,examples,image,audio}], cardErrors[] }],
  //     fileErrors: [string] }
  async function parseFile(file) {
    if (file.size > MAX_IMPORT_BYTES) {
      throw new Error("Файл слишком большой (максимум " + (MAX_IMPORT_BYTES >> 20) + " МБ).");
    }
    var text;
    try {
      text = await file.text();
    } catch (err) {
      throw new Error("Не удалось прочитать файл.");
    }
    var data;
    try {
      data = JSON.parse(text);
    } catch (err) {
      throw new Error("Это не похоже на файл экспорта флешкарт (некорректный JSON).");
    }
    if (!data || typeof data !== "object" || !Array.isArray(data.decks)) {
      throw new Error("Это не похоже на файл экспорта флешкарт (нет поля «decks»).");
    }

    var fileErrors = [];
    var decks = [];

    data.decks.forEach(function (d, di) {
      var name = clean(d && d.name).slice(0, App.deckManager.NAME_MAX);
      if (!name) { fileErrors.push("Колода #" + (di + 1) + ": нет названия, пропущена."); return; }
      if (!d || !Array.isArray(d.cards)) { fileErrors.push("Колода «" + name + "»: нет списка карточек, пропущена."); return; }

      var cards = [];
      var cardErrors = [];
      d.cards.forEach(function (c, ci) {
        var front = clean(c && c.front);
        var back = clean(c && c.back);
        if (!front || !back) {
          cardErrors.push({ line: ci + 1, reason: !front ? "нет лицевой стороны" : "нет обратной стороны" });
          return;
        }
        var examples = Array.isArray(c.examples) ? c.examples.map(clean).filter(Boolean) : [];
        var image = isValidMediaSpec(c.image) ? c.image : null;
        var showCount = Number.isFinite(Number(c.showCount)) ? Math.max(0, Number(c.showCount)) : 0;
        var learned = !!c.learned;
        var due = Number.isFinite(Number(c.due)) ? Number(c.due) : 0;
        var interval = Number.isFinite(Number(c.interval)) ? Math.max(0, Number(c.interval)) : 0;
        var ease = Number.isFinite(Number(c.ease)) ? Math.max(1.3, Number(c.ease)) : 2.5;
        var reps = Number.isFinite(Number(c.reps)) ? Math.max(0, Number(c.reps)) : 0;
        var lapses = Number.isFinite(Number(c.lapses)) ? Math.max(0, Number(c.lapses)) : 0;
        var audio = isValidMediaSpec(c.audio) ? c.audio : null;
        if (c.image && !image) {
          cardErrors.push({ line: ci + 1, reason: "картинка «" + App.util.short(front, 20) + "» повреждена — карточка добавится без неё" });
        }
        if (c.audio && !audio) {
          cardErrors.push({ line: ci + 1, reason: "аудио «" + App.util.short(front, 20) + "» повреждено — карточка добавится без него" });
        }
        cards.push({ front: front, back: back, transcription: clean(c.transcription) || undefined, examples: examples, image: image, audio: audio, due: due, interval: interval, ease: ease, reps: reps, lapses: lapses, showCount: showCount, learned: learned });
      });

      decks.push({ direction: clean(d && d.direction) || "Импорт", name: name, cards: cards, cardErrors: cardErrors });
    });

    return { decks: decks, fileErrors: fileErrors };
  }

  // ---------- импорт: запись в базу ----------

  function base64ToFile(spec) {
    if (!isValidMediaSpec(spec)) throw new Error("Повреждённый медиафайл.");
    var estimatedBytes = Math.floor(spec.data.length * 0.75);
    if (estimatedBytes > 20 * 1024 * 1024) throw new Error("Медиафайл слишком большой (максимум 20 МБ).");
    var binary = atob(spec.data);
    var bytes = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new File([bytes], spec.name || "file", { type: spec.type });
  }

  function uniqueDirectionName(base) {
    base = clean(base) || "Импорт";
    var taken = App.deckManager.getDirections().map(function (d) { return d.name.toLowerCase(); });
    if (taken.indexOf(base.toLowerCase()) === -1) return base;
    var n = 2, name;
    do {
      var suffix = " (" + n + ")";
      name = base.slice(0, App.deckManager.NAME_MAX - suffix.length) + suffix;
      n++;
    } while (taken.indexOf(name.toLowerCase()) !== -1);
    return name;
  }

  function uniqueDeckName(base, directionId) {
    var taken = App.deckManager.getDecks()
      .filter(function (d) { return d.directionId === directionId; })
      .map(function (d) { return d.name.toLowerCase(); });
    if (taken.indexOf(base.toLowerCase()) === -1) return base;
    var n = 2, name;
    do {
      var suffix = " (" + n + ")";
      name = base.slice(0, App.deckManager.NAME_MAX - suffix.length) + suffix;
      n++;
    } while (taken.indexOf(name.toLowerCase()) !== -1);
    return name;
  }

  async function importParsed(parsed) {
    var importedDecks = 0, importedCards = 0, failedDecks = [], firstDeckId = null;
    var directionMap = {};

    for (var i = 0; i < parsed.decks.length; i++) {
      var d = parsed.decks[i];
      if (!d.cards.length) { failedDecks.push({ name: d.name, error: "нет ни одной пригодной карточки" }); continue; }

      var sourceDirection = clean(d.direction) || "Импорт";
      var directionKey = sourceDirection.toLowerCase();
      var directionId = directionMap[directionKey];

      try {
        if (!directionId) {
          var directionName = uniqueDirectionName(sourceDirection);
          directionId = await App.deckManager.createDirection(directionName);
          directionMap[directionKey] = directionId;
        }

        var deckId = await App.deckManager.createDeck(uniqueDeckName(d.name, directionId), directionId);
        var written = [];
        try {
          var records = [];
          for (var j = 0; j < d.cards.length; j++) {
            var c = d.cards[j];
            var id = App.cardManager.newId(deckId);
            var rec = { id: id, front: c.front, back: c.back, transcription: c.transcription, examples: c.examples, due: c.due || 0, interval: c.interval || 0, ease: c.ease == null ? 2.5 : c.ease, reps: c.reps || 0, lapses: c.lapses || 0, showCount: c.showCount || 0, learned: !!c.learned };
            if (c.image) {
              var imgFile = base64ToFile(c.image);
              var imgPath = App.addCard.mediaPath(deckId, id, "image", imgFile);
              await App.mediaStore.put(imgPath, imgFile);
              written.push(imgPath);
              rec.image = imgPath;
            }
            if (c.audio) {
              var audFile = base64ToFile(c.audio);
              var audPath = App.addCard.mediaPath(deckId, id, "audio", audFile);
              await App.mediaStore.put(audPath, audFile);
              written.push(audPath);
              rec.audio = audPath;
            }
            records.push(rec);
          }
          await App.cardManager.addCards(deckId, records);
          importedDecks++;
          importedCards += records.length;
          if (!firstDeckId) firstDeckId = deckId;
        } catch (err) {
          await Promise.all(written.map(function (p) { return App.mediaStore.remove(p); }));
          try { await App.deckManager.deleteDeck(deckId); } catch (e) {}
          failedDecks.push({ name: d.name, error: err && err.message ? err.message : String(err) });
        }
      } catch (err) {
        failedDecks.push({ name: d.name, error: err && err.message ? err.message : String(err) });
      }
    }

    return { importedDecks: importedDecks, importedCards: importedCards, failedDecks: failedDecks, firstDeckId: firstDeckId };
  }


  // ---------- Google Drive sync: lossless snapshot ----------
  async function rawMediaField(path) {
    if (!path) return null;
    var url = App.mediaStore.resolve(path);
    try {
      var response = await fetch(url);
      if (!response.ok) return null;
      var blob = await response.blob();
      var buffer = await blob.arrayBuffer();
      var bytes = new Uint8Array(buffer);
      var binary = "";
      var CHUNK = 0x8000;
      for (var i = 0; i < bytes.length; i += CHUNK) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
      }
      return { path: path, type: blob.type || "application/octet-stream", data: btoa(binary) };
    } catch (err) { return null; }
  }

  async function createSyncSnapshot() {
    var directions = App.db.query("SELECT id, name, position FROM directions ORDER BY position;");
    var decks = App.db.query("SELECT id, direction_id, name, position FROM decks ORDER BY position;");
    var cards = App.db.query("SELECT * FROM cards ORDER BY deck_id, position;");
    var studyEvents = App.db.query("SELECT card_id, deck_id, event_type, rating, created_at FROM study_events ORDER BY id;");
    var out = { app: "flashcards-app", syncVersion: 2, exportedAt: new Date().toISOString(), directions: directions, decks: [], cards: [], studyEvents: studyEvents, media: [] };
    var seenMedia = {};
    decks.forEach(function (d) { out.decks.push({ id:d.id, direction_id:d.direction_id, name:d.name, position:d.position }); });
    for (var i = 0; i < cards.length; i++) {
      var c = cards[i];
      out.cards.push({ id:c.id, deck_id:c.deck_id, position:c.position, front:c.front, back:c.back, transcription:c.transcription, image:c.image, audio:c.audio, video:c.video, examples:c.examples, due:c.due, interval_days:c.interval_days, ease:c.ease, reps:c.reps, lapses:c.lapses, show_count:c.show_count, learned:c.learned });
      var paths = [c.image, c.audio, c.video];
      for (var j = 0; j < paths.length; j++) {
        var path = paths[j];
        if (!path || seenMedia[path]) continue;
        seenMedia[path] = true;
        var spec = await rawMediaField(path);
        if (spec) out.media.push(spec);
      }
    }
    return out;
  }

  function validateSyncSnapshot(data) {
    if (!data || data.app !== "flashcards-app" || (data.syncVersion !== 1 && data.syncVersion !== 2) || !Array.isArray(data.directions) || !Array.isArray(data.decks) || !Array.isArray(data.cards)) {
      throw new Error("Файл Google Drive не является корректным снимком базы флешкарт.");
    }
    var directionIds = {};
    data.directions.forEach(function (d) { directionIds[d.id] = true; });
    var deckIds = {};
    data.decks.forEach(function (d) { if (!directionIds[d.direction_id]) throw new Error("Снимок содержит колоду без направления."); deckIds[d.id] = true; });
    data.cards.forEach(function (c) { if (!deckIds[c.deck_id]) throw new Error("Снимок содержит карточку без колоды."); });
  }

  async function restoreSyncSnapshot(data) {
    validateSyncSnapshot(data);
    var media = Array.isArray(data.media) ? data.media : [];
    for (var i = 0; i < media.length; i++) {
      if (!media[i].path || typeof media[i].data !== "string") continue;
      var binary = atob(media[i].data);
      var bytes = new Uint8Array(binary.length);
      for (var k = 0; k < binary.length; k++) bytes[k] = binary.charCodeAt(k);
      await App.mediaStore.put(media[i].path, new File([bytes], basename(media[i].path), { type: media[i].type || "application/octet-stream" }));
    }
    var db = App.db.getInstance();
    db.run("BEGIN;");
    try {
      db.run("DELETE FROM cards;");
      db.run("DELETE FROM decks;");
      db.run("DELETE FROM directions;");
      db.run("DELETE FROM study_events;");
      var ds = db.prepare("INSERT INTO directions (id,name,position) VALUES (?,?,?);");
      data.directions.forEach(function (d) { ds.run([d.id,d.name,d.position || 0]); }); ds.free();
      var ks = db.prepare("INSERT INTO decks (id,direction_id,name,position) VALUES (?,?,?,?);");
      data.decks.forEach(function (d) { ks.run([d.id,d.direction_id,d.name,d.position || 0]); }); ks.free();
      var cs = db.prepare("INSERT INTO cards (id,deck_id,position,front,back,transcription,image,audio,video,examples,due,interval_days,ease,reps,lapses,show_count,learned) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);");
      data.cards.forEach(function (c) { cs.run([c.id,c.deck_id,c.position || 0,c.front,c.back,c.transcription || null,c.image || null,c.audio || null,c.video || null,c.examples || null,c.due || 0,c.interval_days || 0,c.ease == null ? 2.5 : c.ease,c.reps || 0,c.lapses || 0,c.show_count || 0,c.learned ? 1 : 0]); }); cs.free();
      var es = db.prepare("INSERT INTO study_events (card_id,deck_id,event_type,rating,created_at) VALUES (?,?,?,?,?);");
      (Array.isArray(data.studyEvents) ? data.studyEvents : []).forEach(function (e) {
        if (e.card_id && e.deck_id && e.event_type && Number.isFinite(Number(e.created_at))) es.run([e.card_id,e.deck_id,e.event_type,e.rating || null,Number(e.created_at)]);
      });
      es.free();
      db.run("PRAGMA user_version = 1;");
      db.run("COMMIT;");
    } catch (err) { try { db.run("ROLLBACK;"); } catch (e) {} throw err; }
    await App.db.persist();
    await App.mediaStore.init();
  }

  return {
    exportDeck: exportDeck,
    exportAll: exportAll,
    parseFile: parseFile,
    importParsed: importParsed,
    createSyncSnapshot: createSyncSnapshot,
    restoreSyncSnapshot: restoreSyncSnapshot
  };
})();
