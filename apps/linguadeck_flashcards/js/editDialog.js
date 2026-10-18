// editDialog.js
// UI диалога «Редактировать карточку». Значения из формы отдаёт в editCard.update();
// сам в базу и в OPFS не пишет. Пользовательский текст — только через textContent/value.

window.App = window.App || {};

App.editDialog = (function () {
  var opts = { onChanged: null };
  var els = {};
  var cardId = null;
  var busy = false;

  var IDS = ["dialog", "close", "front", "back", "transcription", "examples",
    "image", "image-cur", "image-remove-wrap", "image-remove",
    "audio", "audio-cur", "audio-remove-wrap", "audio-remove",
    "status", "save", "cancel"];

  function setStatus(text) {
    els.status.textContent = text || "";
    els.status.hidden = !text;
    els.status.className = "status is-error";
  }

  // Что сейчас лежит в карточке: миниатюра / плеер + галочка «Удалить файл»
  function renderCurrent(kind, path) {
    var box = els[kind + "-cur"];
    var wrap = els[kind + "-remove-wrap"];
    box.textContent = "";
    els[kind + "-remove"].checked = false;
    els[kind + "-remove"].disabled = false;
    els[kind].value = "";
    if (!path) {
      box.textContent = "Сейчас: нет файла";
      wrap.hidden = true;
      return;
    }
    var url = App.mediaStore.resolve(path);
    var node;
    if (kind === "image") {
      node = document.createElement("img");
      node.className = "thumb";
      node.alt = "";
    } else {
      node = document.createElement("audio");
      node.controls = true;
      node.preload = "none";
    }
    node.src = url;
    box.appendChild(node);
    wrap.hidden = false;
  }

  function open(id) {
    var raw = App.cardManager.getRaw(id);
    if (!raw) return;
    cardId = id;
    els.front.value = raw.front || "";
    els.back.value = raw.back || "";
    els.transcription.value = raw.transcription || "";
    els.examples.value = raw.examples ? JSON.parse(raw.examples).join("\n") : "";
    renderCurrent("image", raw.image);
    renderCurrent("audio", raw.audio);
    setStatus("");
    els.dialog.showModal();
    els.front.focus();
  }

  async function save() {
    if (busy) return;
    busy = true;
    els.save.disabled = true;
    setStatus("");
    try {
      await App.editCard.update(cardId, {
        front: els.front.value,
        back: els.back.value,
        transcription: els.transcription.value,
        examples: els.examples.value.split(/\r\n|\r|\n/),
        image: { file: els.image.files[0] || null, remove: els["image-remove"].checked },
        audio: { file: els.audio.files[0] || null, remove: els["audio-remove"].checked }
      });
      els.dialog.close();
      try {
        if (opts.onChanged) opts.onChanged(cardId);
      } catch (err) {
        console.error("[editDialog] onChanged:", err); // сохранено — юзеру не ошибка
      }
    } catch (err) {
      setStatus(err && err.message ? err.message : String(err));
    } finally {
      busy = false;
      els.save.disabled = false;
    }
  }

  function init(options) {
    opts = options;
    IDS.forEach(function (id) { els[id] = document.getElementById("edit-" + id); });

    els.close.addEventListener("click", function () { els.dialog.close(); });
    els.cancel.addEventListener("click", function () { els.dialog.close(); });
    els.dialog.addEventListener("click", function (e) {
      if (e.target === els.dialog) els.dialog.close();
    });
    // при закрытии гасим плеер и чистим превью
    els.dialog.addEventListener("close", function () {
      els["audio-cur"].textContent = "";
      els["image-cur"].textContent = "";
    });

    // выбранный файл заменяет старый — «удалить» тогда не имеет смысла
    ["image", "audio"].forEach(function (k) {
      els[k].addEventListener("change", function () {
        var has = !!els[k].files[0];
        els[k + "-remove"].checked = false;
        els[k + "-remove"].disabled = has;
      });
    });

    els.save.addEventListener("click", save);
    els.dialog.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        save();
      }
    });
  }

  return { init: init, open: open };
})();
