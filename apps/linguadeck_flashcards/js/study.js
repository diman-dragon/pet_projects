// study.js
// Состояние одной учебной сессии: очередь карточек colody, чей срок наступил.
// Не знает про DOM — только про очередь и сохранение оценки через cardManager.

window.App = window.App || {};

App.study = (function () {
  var queue = [];
  var started = false;
  var busy = false;

  // cards — deck.cards (уже в порядке позиции). Берутся ТЕ ЖЕ объекты (не копии):
  // после grade() мы правим их поля прямо в них, поэтому deckManager.refresh(),
  // вызванный после сессии, просто перезатрёт их свежими данными из базы — ничего
  // не рассинхронизируется.
  function start(cards) {
    queue = App.srs.dueCards(cards);
    started = true;
  }

  function isActive() {
    return started;
  }

  function current() {
    return queue[0] || null;
  }

  function remaining() {
    return queue.length;
  }

  // Прогноз интервала по каждой из 4 оценок для текущей карточки — для подписей
  // под кнопками. Ничего не пишет и не мутирует.
  function preview(card) {
    var out = {};
    App.srs.RATINGS.forEach(function (r) { out[r] = App.srs.grade(card, r).interval; });
    return out;
  }

  // Сохраняет оценку в базу и продвигает очередь. Если запись в базу упала —
  // бросает исключение, очередь и текущая карточка НЕ трогаются (можно повторить).
  async function grade(rating) {
    if (busy) return null;
    var card = queue[0];
    if (!card) return null;
    busy = true;
    var next = App.srs.grade(card, rating);
    next.rating = rating;
    try { await App.cardManager.updateSrs(card.id, next); } finally { busy = false; }
    card.due = next.due;
    card.interval = next.interval;
    card.ease = next.ease;
    card.reps = next.reps;
    card.lapses = next.lapses;
    card.learned = !!next.learned;
    queue.shift();
    if (rating === App.srs.AGAIN) queue.push(card); // увидим ещё раз в этой же сессии
    return card;
  }

  // Пропуск не считается изучением: карточка остаётся невыученной и просто
  // убирается из текущей очереди. Она снова попадёт в следующую учебную сессию.
  async function skip() {
    if (busy) return null;
    var card = queue[0];
    if (!card) return null;
    busy = true;
    try {
      await App.cardManager.skip(card.id);
      card.learned = false;
      queue.shift();
      return card;
    } finally {
      busy = false;
    }
  }

  function stop() {
    queue = [];
    started = false;
    busy = false;
  }

  function isBusy() { return busy; }

  return {
    start: start,
    isActive: isActive,
    current: current,
    remaining: remaining,
    preview: preview,
    grade: grade,
    skip: skip,
    stop: stop,
    isBusy: isBusy
  };
})();
