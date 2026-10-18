// srs.js
// Чистая логика интервального повторения (упрощённый SM-2, 4 кнопки-оценки).
// Ничего не знает про DOM и про базу — принимает и возвращает обычные объекты.
// due/interval — в ДНЯХ от эпохи (Math.floor(Date.now()/86400000)), без времени суток:
// повторения гранулярны по дню, а не по часам/минутам.

window.App = window.App || {};

App.srs = (function () {
  var AGAIN = "again";
  var HARD = "hard";
  var GOOD = "good";
  var EASY = "easy";
  var RATINGS = [AGAIN, HARD, GOOD, EASY];

  var DEFAULT_EASE = 2.5;
  var MIN_EASE = 1.3;

  // День SRS определяется по локальному календарю пользователя, а не по UTC.
  // Так «сегодня» действительно меняется в полночь локального времени.
  function today() {
    var d = new Date();
    return Math.floor(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) / 86400000);
  }

  // Значения по умолчанию для только что созданной карточки — due в прошлом/сегодня,
  // поэтому новая карточка сразу попадает в очередь на повторение.
  function initial() {
    return { due: 0, interval: 0, ease: DEFAULT_EASE, reps: 0, lapses: 0 };
  }

  function clampEase(e) {
    return Math.max(MIN_EASE, e);
  }

  // card: { interval?, ease?, reps?, lapses? } (недостающие поля = как у новой карточки)
  // rating: one of AGAIN/HARD/GOOD/EASY
  // → { due, interval, ease, reps, lapses } — не мутирует card, чистая функция.
  function grade(card, rating) {
    var ease = typeof card.ease === "number" ? card.ease : DEFAULT_EASE;
    var interval = typeof card.interval === "number" ? card.interval : 0;
    var reps = typeof card.reps === "number" ? card.reps : 0;
    var lapses = typeof card.lapses === "number" ? card.lapses : 0;
    var t = today();

    if (rating === AGAIN) {
      // Забыли — сбрасываем прогресс интервала, но не "стаж" (ease снижаем, а не обнуляем),
      // due = сегодня: в этой же сессии карточка вернётся ещё раз (см. study.js).
      return { due: t, interval: 0, ease: clampEase(ease - 0.2), reps: 0, lapses: lapses + 1, learned: false };
    }

    reps += 1;
    var nextInterval;
    if (rating === HARD) {
      ease = clampEase(ease - 0.15);
      nextInterval = reps <= 2 ? 1 : Math.max(1, Math.round(interval * 1.2));
    } else if (rating === EASY) {
      ease = clampEase(ease + 0.15);
      nextInterval = reps === 1 ? 4 : (reps === 2 ? 7 : Math.max(1, Math.round(interval * ease * 1.3)));
    } else { // GOOD
      nextInterval = reps === 1 ? 1 : (reps === 2 ? 6 : Math.max(1, Math.round(interval * ease)));
    }

    return { due: t + nextInterval, interval: nextInterval, ease: ease, reps: reps, lapses: lapses, learned: true };
  }

  // Карточки колоды, чей срок повторения уже наступил, в порядке due → позиция.
  // sort() в современных движках стабилен, поэтому при равном due порядок карточек
  // (уже отсортированных по position в SQL-запросе) не ломается.
  function dueCards(cards) {
    var t = today();
    return cards
      .filter(function (c) { return (c.due || 0) <= t; })
      .sort(function (a, b) { return (a.due || 0) - (b.due || 0); });
  }

  function dueCount(cards) {
    return dueCards(cards).length;
  }

  function formatDays(n) {
    if (n <= 0) return "сегодня";
    if (n === 1) return "завтра";
    return "через " + n + " " + App.util.plural(n, ["день", "дня", "дней"]);
  }

  return {
    AGAIN: AGAIN, HARD: HARD, GOOD: GOOD, EASY: EASY, RATINGS: RATINGS,
    today: today,
    initial: initial,
    grade: grade,
    dueCards: dueCards,
    dueCount: dueCount,
    formatDays: formatDays
  };
})();
