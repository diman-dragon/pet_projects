// navigation.js
// Отвечает только за текущий индекс карточки внутри колоды.
// Ничего не знает про DOM и про переворот — только про позицию.

window.App = window.App || {};

App.navigation = (function () {
  let index = 0;
  let total = 0;

  function reset(cardCount) {
    total = cardCount;
    index = 0;
  }

  function next() {
    if (total === 0) return index;
    index = (index + 1) % total; // зацикливаем — удобно для повторения
    return index;
  }

  function prev() {
    if (total === 0) return index;
    index = (index - 1 + total) % total;
    return index;
  }

  // Перейти к конкретной карточке (после добавления новых). Вне диапазона — игнорируем.
  function goTo(i) {
    if (i >= 0 && i < total) index = i;
    return index;
  }

  function getIndex() {
    return index;
  }

  function getTotal() {
    return total;
  }

  return {
    reset: reset,
    next: next,
    prev: prev,
    goTo: goTo,
    getIndex: getIndex,
    getTotal: getTotal
  };
})();
