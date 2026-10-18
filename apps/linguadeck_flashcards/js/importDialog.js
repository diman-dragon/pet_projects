// importDialog.js
// UI диалога импорта: выбор .json-файла → предпросмотр (сколько колод/карточек,
// что пропущено) → подтверждение. Сам разбор и запись — в backup.js.

window.App = window.App || {};

App.importDialog = (function () {
  var opts = { onImported: null };
  var els = {};
  var parsed = null;
  var busy = false;

  function setStatus(text, kind) {
    els.status.textContent = text || "";
    els.status.hidden = !text;
    els.status.className = "status" + (kind ? " is-" + kind : "");
  }

  function addLine(parent, className, text) {
    var p = document.createElement("p");
    p.className = className;
    p.textContent = text;
    parent.appendChild(p);
  }

  function renderPreview() {
    var box = els.preview;
    box.textContent = "";
    if (!parsed) { box.hidden = true; els.submit.disabled = true; els.submit.textContent = "Импортировать"; return; }
    box.hidden = false;

    var totalCards = parsed.decks.reduce(function (s, d) { return s + d.cards.length; }, 0);
    addLine(box, "preview-summary", "Колод: " + parsed.decks.length + " · карточек: " + totalCards);

    parsed.decks.forEach(function (d) {
      var n = d.cards.length;
      addLine(box, "preview-row", "«" + App.util.short(d.name, 40) + "» — " + n + " " +
        App.util.plural(n, ["карточка", "карточки", "карточек"]) +
        (d.cardErrors.length ? "  (пропущено: " + d.cardErrors.length + ")" : ""));
    });

    parsed.fileErrors.forEach(function (e) { addLine(box, "preview-error", e); });

    var cardErrors = [];
    parsed.decks.forEach(function (d) { cardErrors = cardErrors.concat(d.cardErrors); });
    cardErrors.slice(0, 5).forEach(function (e) { addLine(box, "preview-muted", "Строка " + e.line + ": " + e.reason); });
    if (cardErrors.length > 5) addLine(box, "preview-muted", "…и ещё " + (cardErrors.length - 5));

    els.submit.disabled = busy || !totalCards;
    els.submit.textContent = totalCards ? "Импортировать (" + totalCards + ")" : "Импортировать";
  }

  async function loadFile(file) {
    parsed = null;
    renderPreview();
    setStatus("Читаю файл…");
    try {
      parsed = await App.backup.parseFile(file);
      setStatus("");
    } catch (err) {
      setStatus(err && err.message ? err.message : String(err), "error");
    }
    renderPreview();
  }

  async function submit() {
    if (busy || !parsed) return;
    busy = true;
    els.submit.disabled = true;
    setStatus("Импортирую…");
    try {
      var res = await App.backup.importParsed(parsed);
      var msg = "Импортировано: " + res.importedDecks + " " +
        App.util.plural(res.importedDecks, ["колода", "колоды", "колод"]) +
        " · " + res.importedCards + " " + App.util.plural(res.importedCards, ["карточка", "карточки", "карточек"]);
      if (res.failedDecks.length) {
        msg += " · не удалось: " + res.failedDecks.map(function (f) { return f.name; }).join(", ");
      }
      setStatus(msg, res.failedDecks.length ? "error" : "ok");

      if (res.importedDecks && opts.onImported) {
        try { opts.onImported(res.firstDeckId); } catch (e) { console.error("[importDialog] onImported:", e); }
      }
      parsed = null;
      els.file.value = "";
      renderPreview();
      if (!res.failedDecks.length) setTimeout(function () { els.dialog.close(); }, 1400);
    } catch (err) {
      setStatus(err && err.message ? err.message : String(err), "error");
    } finally {
      busy = false;
      renderPreview();
    }
  }

  function open() {
    parsed = null;
    els.file.value = "";
    setStatus("");
    renderPreview();
    els.dialog.showModal();
  }

  function init(options) {
    opts = options;
    ["dialog", "close", "file", "status", "preview", "submit", "cancel"].forEach(function (id) {
      els[id] = document.getElementById("import-" + id);
    });
    els.close.addEventListener("click", function () { els.dialog.close(); });
    els.cancel.addEventListener("click", function () { els.dialog.close(); });
    els.dialog.addEventListener("click", function (e) { if (e.target === els.dialog) els.dialog.close(); });
    els.file.addEventListener("change", function () {
      var f = els.file.files[0];
      if (f) loadFile(f);
    });
    els.submit.addEventListener("click", submit);
  }

  return { init: init, open: open };
})();
