// directionDialog.js
// Диалог создания/переименования направления.

window.App = window.App || {};

App.directionDialog = (function () {
  var opts = { onChanged: null };
  var els = {};
  var mode = "create";
  var directionId = null;
  var busy = false;

  function setStatus(text) {
    els.status.textContent = text || "";
    els.status.hidden = !text;
    els.status.className = "status is-error";
  }

  function open(options) {
    mode = options.mode;
    directionId = options.direction ? options.direction.id : null;
    els.title.textContent = mode === "rename" ? "Переименовать направление" : "Новое направление";
    els.save.textContent = mode === "rename" ? "Сохранить" : "Создать";
    els.name.maxLength = App.deckManager.NAME_MAX;
    els.name.value = options.direction ? options.direction.name : "";
    setStatus("");
    els.dialog.showModal();
    els.name.focus();
    els.name.select();
  }

  async function save() {
    if (busy) return;
    busy = true;
    els.save.disabled = true;
    setStatus("");
    try {
      if (mode === "rename") await App.deckManager.renameDirection(directionId, els.name.value);
      else await App.deckManager.createDirection(els.name.value);
      els.dialog.close();
      try { if (opts.onChanged) opts.onChanged(mode); }
      catch (err) { console.error("[directionDialog] onChanged:", err); }
    } catch (err) {
      setStatus(err && err.message ? err.message : String(err));
    } finally {
      busy = false;
      els.save.disabled = false;
    }
  }

  function init(options) {
    opts = options;
    ["dialog", "title", "name", "status", "save", "cancel"].forEach(function (id) {
      els[id] = document.getElementById("direction-" + id);
    });
    els.cancel.addEventListener("click", function () { els.dialog.close(); });
    els.dialog.addEventListener("click", function (e) { if (e.target === els.dialog) els.dialog.close(); });
    els.save.addEventListener("click", save);
    els.name.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); save(); }
    });
  }

  return { init: init, open: open };
})();
