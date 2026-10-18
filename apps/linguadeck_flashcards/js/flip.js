// flip.js
// Отвечает только за визуальное состояние "перевёрнута/не перевёрнута".
// Не знает ничего о колодах и навигации.

window.App = window.App || {};

App.flip = (function () {
  const FLIPPED_CLASS = "is-flipped";

  function toggle(cardEl) {
    cardEl.classList.toggle(FLIPPED_CLASS);
    var flipped = cardEl.classList.contains(FLIPPED_CLASS);
    cardEl.setAttribute("aria-pressed", flipped ? "true" : "false");
  }

  function reset(cardEl) {
    cardEl.classList.remove(FLIPPED_CLASS);
    cardEl.setAttribute("aria-pressed", "false");
  }

  function isFlipped(cardEl) {
    return cardEl.classList.contains(FLIPPED_CLASS);
  }

  return {
    toggle: toggle,
    reset: reset,
    isFlipped: isFlipped
  };
})();
