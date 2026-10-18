// editCard.js
// Правка и удаление ОДНОЙ существующей карточки, включая её медиафайлы.
// Как и addCard.js, не знает про DOM и кидает Error с русским текстом.

window.App = window.App || {};

App.editCard = (function () {
  var KINDS = ["image", "audio"];

  function clean(v) { return (v == null ? "" : String(v)).trim(); }

  // input: { front, back, transcription, examples: string[],
  //          image: { file?: File, remove?: bool }, audio: { file?, remove? } }
  //   file    — заменить файл; remove — убрать файл (если file задан, он важнее);
  //   ничего  — оставить как есть.
  async function update(cardId, input) {
    var raw = App.cardManager.getRaw(cardId);
    if (!raw) throw new Error("Карточка не найдена (возможно, уже удалена).");

    var front = clean(input.front);
    var back = clean(input.back);
    if (!front) throw new Error("Заполните лицевую сторону (термин).");
    if (!back) throw new Error("Заполните обратную сторону (перевод / определение).");

    KINDS.forEach(function (k) {
      var spec = input[k];
      if (spec && spec.file) App.addCard.checkFile(spec.file, k);
    });

    var next = { image: raw.image || null, audio: raw.audio || null };
    var written = [];
    try {
      for (var i = 0; i < KINDS.length; i++) {
        var k = KINDS[i];
        var spec = input[k] || {};
        if (spec.file) {
          var path = App.addCard.mediaPath(raw.deck_id, cardId, k, spec.file);
          await App.mediaStore.put(path, spec.file);
          if (path !== raw[k]) written.push(path); // перезапись того же пути откатить нельзя, но и удалять нечего
          next[k] = path;
        } else if (spec.remove) {
          next[k] = null;
        }
      }
      await App.cardManager.updateCard(cardId, {
        front: front,
        back: back,
        transcription: clean(input.transcription) || undefined,
        image: next.image,
        audio: next.audio,
        examples: (input.examples || []).map(clean).filter(Boolean)
      });
    } catch (err) {
      await Promise.all(written.map(function (p) { return App.mediaStore.remove(p); }));
      throw err;
    }

    // Старые файлы, на которые карточка больше не ссылается, удаляем уже после успешной записи
    await Promise.all(KINDS.map(function (k) {
      return (raw[k] && raw[k] !== next[k]) ? App.mediaStore.remove(raw[k]) : null;
    }));
  }

  // Удаляет карточку и её файлы из OPFS (файлы мок-данных из репозитория в OPFS
  // отсутствуют — для них remove ничего не делает).
  async function remove(cardId) {
    var raw = App.cardManager.getRaw(cardId);
    if (!raw) return;
    await App.cardManager.deleteCard(cardId);
    await Promise.all(["image", "audio", "video"].map(function (k) {
      return raw[k] ? App.mediaStore.remove(raw[k]) : null;
    }));
  }

  return { update: update, remove: remove };
})();
