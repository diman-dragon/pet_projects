// menu.js
// Выпадающее меню «⋯» в верхней панели. Только показ/скрытие и клавиатура;
// что делает каждый пункт — решает actions.js (пункты различаются по data-action).

window.App = window.App || {};

App.menu = (function () {
  var btn = null;
  var menu = null;
  var onAction = null;
  var backdrop = null;
  var closeBtn = null;

  function items() {
    return Array.prototype.filter.call(menu.querySelectorAll("[data-action], [data-sidebar-study]"), function (el) { return !el.disabled; });
  }

  function isOpen() { return !menu.hidden; }

  function open() {
    menu.hidden = false;
    if (backdrop) backdrop.hidden = false;
    requestAnimationFrame(function () { menu.classList.add("is-open"); });
    btn.setAttribute("aria-expanded", "true");
    var first = items()[0];
    if (first) first.focus();
  }

  function close(returnFocus) {
    menu.classList.remove("is-open");
    if (backdrop) backdrop.hidden = true;
    menu.hidden = true;
    btn.setAttribute("aria-expanded", "false");
    if (returnFocus) btn.focus();
  }

  function setEnabled(action, enabled) {
    var el = menu.querySelector('[data-action="' + action + '"]');
    if (el) el.disabled = !enabled;
  }

  function init(options) {
    onAction = options.onAction;
    btn = document.getElementById("btn-menu");
    menu = document.getElementById("menu");
    backdrop = document.getElementById("menu-backdrop");
    closeBtn = document.getElementById("sidebar-close");

    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      if (isOpen()) close(false); else open();
    });

    if (closeBtn) closeBtn.addEventListener("click", function () { close(false); });
    if (backdrop) backdrop.addEventListener("click", function () { close(false); });

    menu.addEventListener("click", function (e) {
      var study = e.target.closest("[data-sidebar-study]");
      if (study) { close(false); onAction("study"); return; }
      var el = e.target.closest("[data-action]");
      if (!el || el.disabled) return;
      close(false);
      onAction(el.dataset.action);
    });

    document.addEventListener("click", function (e) {
      if (isOpen() && e.target !== btn && !menu.contains(e.target)) close(false);
    });

    menu.addEventListener("keydown", function (e) {
      var list = items();
      var i = list.indexOf(document.activeElement);
      if (e.key === "ArrowDown") { e.preventDefault(); list[(i + 1) % list.length].focus(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); list[(i - 1 + list.length) % list.length].focus(); }
      else if (e.key === "Escape") { e.preventDefault(); close(true); }
      else if (e.key === "Tab") { close(false); }
    });
  }

  return { init: init, setEnabled: setEnabled };
})();
