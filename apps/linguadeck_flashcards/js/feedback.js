// feedback.js
// Обратная связь пользователю: диалог подтверждения (App.confirm) и всплывающее
// сообщение (App.toast). Разметка — #confirm-dialog и #toast в index.html.

window.App = window.App || {};

App.confirm = (function () {
  // ask({ title, message, confirmText? }) → Promise<boolean>
  // Esc, клик по подложке и «Отмена» = false. Фокус по умолчанию на «Отмена».
  function ask(o) {
    var dlg = document.getElementById("confirm-dialog");
    var ok = document.getElementById("confirm-ok");
    var cancel = document.getElementById("confirm-cancel");
    document.getElementById("confirm-title").textContent = o.title || "Подтвердите действие";
    document.getElementById("confirm-text").textContent = o.message || "";
    ok.textContent = o.confirmText || "Удалить";

    return new Promise(function (resolve) {
      function onOk() { dlg.close("ok"); }
      function onCancel() { dlg.close("cancel"); }
      function onBackdrop(e) { if (e.target === dlg) dlg.close("cancel"); }
      function onClose() {
        ok.removeEventListener("click", onOk);
        cancel.removeEventListener("click", onCancel);
        dlg.removeEventListener("click", onBackdrop);
        resolve(dlg.returnValue === "ok");
      }
      ok.addEventListener("click", onOk);
      cancel.addEventListener("click", onCancel);
      dlg.addEventListener("click", onBackdrop);
      dlg.addEventListener("close", onClose, { once: true });
      dlg.returnValue = "";
      dlg.showModal();
      cancel.focus();
    });
  }

  return { ask: ask };
})();

App.toast = (function () {
  var timer = null;

  // show("Текст", "ok" | "error")
  function show(text, kind) {
    var el = document.getElementById("toast");
    el.textContent = text;
    el.className = "toast is-" + (kind || "ok");
    el.hidden = false;
    clearTimeout(timer);
    timer = setTimeout(function () { el.hidden = true; }, kind === "error" ? 5000 : 2500);
  }

  return { show: show };
})();
