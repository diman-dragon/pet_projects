// addDialog.js
// UI диалога «+»: две вкладки (одна карточка / пакетно). Собирает значения из формы
// и отдаёт их в addCard.js / batchAdd.js; сам в базу и в OPFS не пишет.
// Весь пользовательский текст выводится через textContent (без innerHTML).

window.App = window.App || {};

App.addDialog = (function () {
  var opts = { getDeck: null, onAdded: null };
  var els = {};
  var busy = false;
  var previewTimer = null;
  var lastParse = null;
  var currentTab = "single";

  var PREVIEW_ROWS = 3;
  var PREVIEW_ERRORS = 6;

  function $(id) { return document.getElementById(id); }

  var plural = App.util.plural;
  var short = App.util.short;

  function delimName(d) {
    return d === "\t" ? "Tab" : "«" + d + "»";
  }

  function setStatus(text, kind) {
    els.status.textContent = text || "";
    els.status.hidden = !text;
    els.status.className = "status" + (kind ? " is-" + kind : "");
  }

  function setBusy(value) {
    busy = value;
    els["single-submit"].disabled = value;
    els["batch-submit"].disabled = value || !(lastParse && lastParse.rows.length);
  }

  function notifyAdded(firstId) {
    try {
      if (opts.onAdded) opts.onAdded(firstId);
    } catch (err) {
      console.error("[addDialog] onAdded:", err); // карточки уже добавлены — юзеру не ошибка
    }
  }

  // ---------- вкладки ----------

  function selectTab(name) {
    currentTab = name;
    ["single", "batch"].forEach(function (t) {
      var active = t === name;
      els["tab-" + t].classList.toggle("is-active", active);
      els["tab-" + t].setAttribute("aria-selected", active ? "true" : "false");
      els["tab-" + t].tabIndex = active ? 0 : -1;
      els["panel-" + t].hidden = !active;
    });
    setStatus("");
    if (name === "batch") updatePreview();
  }

  // ---------- одна карточка ----------

  function clearSingleForm() {
    ["front", "back", "transcription", "examples"].forEach(function (k) { els[k].value = ""; });
    els.image.value = "";
    els.audio.value = "";
  }

  async function submitSingle() {
    if (busy) return;
    var deck = opts.getDeck();
    setBusy(true);
    setStatus("");
    try {
      var res = await App.addCard.add(deck && deck.id, {
        front: els.front.value,
        back: els.back.value,
        transcription: els.transcription.value,
        examples: els.examples.value.split(/\r\n|\r|\n/),
        imageFile: els.image.files[0] || null,
        audioFile: els.audio.files[0] || null
      });
      clearSingleForm();
      setStatus("Карточка добавлена ✓", "ok");
      notifyAdded(res.id);
      els.front.focus();
    } catch (err) {
      setStatus(err && err.message ? err.message : String(err), "error");
    } finally {
      setBusy(false);
    }
  }

  // ---------- пакетно ----------

  function batchOptions() {
    var deck = opts.getDeck();
    return {
      delimiter: els.delim.value,
      skipHeader: els.header.checked,
      skipDuplicates: els.dups.checked,
      existing: deck ? deck.cards : []
    };
  }

  function addLine(parent, className, text) {
    var p = document.createElement("p");
    p.className = className;
    p.textContent = text;
    parent.appendChild(p);
    return p;
  }

  function renderPreview() {
    var box = els.preview;
    box.textContent = "";
    var res = lastParse;
    var hasText = !!els.text.value.trim();
    box.hidden = !hasText;
    if (!hasText || !res) return;

    var auto = els.delim.value === "auto";
    var head = "Найдено: " + res.rows.length + " · пропущено: " + res.errors.length +
      " · разделитель: " + delimName(res.delimiter) + (auto ? " (авто)" : "");
    addLine(box, "preview-summary", head);
    if (res.headerLine) addLine(box, "preview-muted", "Строка " + res.headerLine + " — заголовок, пропущена");

    res.rows.slice(0, PREVIEW_ROWS).forEach(function (r) {
      var extra = [];
      if (r.transcription) extra.push(r.transcription);
      if (r.examples.length) extra.push(r.examples.length + " " + plural(r.examples.length, ["пример", "примера", "примеров"]));
      addLine(box, "preview-row", short(r.front, 30) + "  →  " + short(r.back, 30) +
        (extra.length ? "   (" + extra.join(" · ") + ")" : ""));
    });
    if (res.rows.length > PREVIEW_ROWS) {
      addLine(box, "preview-muted", "…и ещё " + (res.rows.length - PREVIEW_ROWS));
    }

    res.errors.slice(0, PREVIEW_ERRORS).forEach(function (e) {
      addLine(box, "preview-error", "Строка " + e.line + ": " + e.reason + " — «" + short(e.raw.trim(), 40) + "»");
    });
    if (res.errors.length > PREVIEW_ERRORS) {
      addLine(box, "preview-muted", "…и ещё " + (res.errors.length - PREVIEW_ERRORS) + " пропущенных");
    }
  }

  function updatePreview() {
    lastParse = App.batchAdd.parse(els.text.value, batchOptions());
    renderPreview();
    var n = lastParse.rows.length;
    els["batch-submit"].textContent = n ? "Добавить " + n + " " + plural(n, ["карточку", "карточки", "карточек"]) : "Добавить";
    els["batch-submit"].disabled = busy || !n;
  }

  function schedulePreview() {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(updatePreview, 150);
  }

  async function loadFile() {
    var file = els.file.files[0];
    if (!file) return;
    try {
      els.text.value = await App.batchAdd.readFile(file);
      setStatus("");
      updatePreview();
    } catch (err) {
      setStatus(err && err.message ? err.message : String(err), "error");
    } finally {
      els.file.value = ""; // чтобы повторный выбор того же файла снова сработал
    }
  }

  async function submitBatch() {
    if (busy) return;
    clearTimeout(previewTimer);
    updatePreview(); // берём актуальный разбор, а не тот, что был 150 мс назад
    var deck = opts.getDeck();
    if (!lastParse.rows.length) return;
    var skipped = lastParse.errors.length;

    setBusy(true);
    setStatus("");
    try {
      var res = await App.batchAdd.add(deck && deck.id, lastParse.rows);
      els.text.value = "";
      updatePreview();
      setStatus(
        "Добавлено: " + res.count + (skipped ? " · пропущено: " + skipped : "") + " ✓",
        "ok"
      );
      notifyAdded(res.ids[0]);
    } catch (err) {
      setStatus(err && err.message ? err.message : String(err), "error");
    } finally {
      setBusy(false);
    }
  }

  // ---------- открытие / привязка событий ----------

  function open() {
    var deck = opts.getDeck();
    els["deck-name"].textContent = deck ? "Колода: " + deck.name : "Нет доступной колоды";
    var direction = App.deckManager.getCurrentDirection();
    if (els.hierarchy) els.hierarchy.textContent = direction && deck ? "Направление: " + direction.name + " → Колода: " + deck.name : "";
    setStatus("");
    els.dialog.showModal();
    updatePreview(); // колода могла смениться — пересчитываем дубликаты
    if (currentTab === "single") els.front.focus();
    else els.text.focus();
  }

  function close() {
    els.dialog.close();
  }

  function init(options) {
    opts = options;
    ["dialog", "close", "deck-name", "status", "tabs",
     "tab-single", "tab-batch", "panel-single", "panel-batch",
     "front", "back", "transcription", "examples", "image", "audio", "single-submit",
     "text", "file", "delim", "header", "dups", "preview", "batch-submit"
    ].forEach(function (id) {
      els[id] = $("add-" + id); // id в разметке — "add-<имя>"
    });
    els.hierarchy = $("add-hierarchy");

    els.close.addEventListener("click", close);
    // клик по подложке (сам <dialog> без внутренних отступов) закрывает диалог
    els.dialog.addEventListener("click", function (e) {
      if (e.target === els.dialog) close();
    });

    ["single", "batch"].forEach(function (t) {
      els["tab-" + t].addEventListener("click", function () { selectTab(t); });
    });
    // стрелки ← → между вкладками (доступность)
    els.tabs.addEventListener("keydown", function (e) {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      var next = currentTab === "single" ? "batch" : "single";
      selectTab(next);
      els["tab-" + next].focus();
    });

    els["single-submit"].addEventListener("click", submitSingle);
    els["batch-submit"].addEventListener("click", submitBatch);
    els.text.addEventListener("input", schedulePreview);
    [els.delim, els.header, els.dups].forEach(function (el) {
      el.addEventListener("change", updatePreview);
    });
    els.file.addEventListener("change", loadFile);

    // Ctrl/Cmd+Enter — отправить текущую вкладку
    els.dialog.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        if (currentTab === "single") submitSingle(); else submitBatch();
      }
    });
  }

  return {
    init: init,
    open: open,
    close: close
  };
})();
