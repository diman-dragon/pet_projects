// util.js
// Мелкие чистые помощники без состояния и без DOM.

window.App = window.App || {};

App.util = (function () {
  // plural(5, ["карточку", "карточки", "карточек"]) → "карточек"
  function plural(n, forms) {
    var m10 = n % 10;
    var m100 = n % 100;
    if (m10 === 1 && m100 !== 11) return forms[0];
    if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return forms[1];
    return forms[2];
  }

  function short(s, n) {
    s = String(s);
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }

  return { plural: plural, short: short };
})();
