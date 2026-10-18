// deckDialog.js
// Диалог создания/переименования колоды.

window.App = window.App || {};

App.deckDialog = (function () {
  var opts = { onChanged: null };
  var els = {};
  var mode = "create";
  var deckId = null;
  var busy = false;

  function setStatus(text) {
    els.status.textContent = text || "";
    els.status.hidden = !text;
    els.status.className = "status is-error";
  }

  function open(options) {
    mode = options.mode;
    deckId = options.deck ? options.deck.id : null;
    els.title.textContent = mode === "rename" ? "Переименовать колоду" : "Новая колода";
    els.save.textContent = mode === "rename" ? "Сохранить" : "Создать";
    els.name.maxLength = App.deckManager.NAME_MAX;
    els.name.value = options.deck ? options.deck.name : "";
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
      if (mode === "rename") await App.deckManager.renameDeck(deckId, els.name.value);
      else await App.deckManager.createDeck(els.name.value);
      els.dialog.close();
      try { if (opts.onChanged) opts.onChanged(mode); }
      catch (err) { console.error("[deckDialog] onChanged:", err); }
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
      els[id] = document.getElementById("deck-" + id);
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
