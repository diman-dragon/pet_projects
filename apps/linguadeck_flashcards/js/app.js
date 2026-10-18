// app.js
// Точка входа: берёт готовые модули и связывает их через события DOM.
// Сюда не добавляем новую логику — только вызовы других модулей.

window.App = window.App || {};

(function () {
  function $(id) {
    return document.getElementById(id);
  }

  // Собираем ссылки на все узлы карточки один раз
  var cardEls = {
    card: null,
    term: null,
    transcription: null,
    image: null,
    description: null,
    examples: null,
    audioBtn: null,
    videoBtn: null,
    showCount: null
  };

  var currentAudio = null;
  var studyMode = false; // false = обычный обзор колоды, true = учебная сессия (study.js)

  function stopAudio() {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
    }
  }

  // Сколько карточек текущей колоды уже пора повторить — бейдж на кнопке «Учить»
  // и условие, можно ли вообще начать сессию.
  function updateStudyBadge() {
    var deck = App.deckManager.getCurrentDeck();
    var n = deck ? App.srs.dueCount(deck.cards) : 0;
    $("study-badge").textContent = n;
    var sidebarBadge = $("sidebar-study-badge");
    if (sidebarBadge) sidebarBadge.textContent = n;
    $("btn-study").disabled = studyMode || !deck || n === 0;
    var topDirection = $("top-current-direction");
    var topDeck = $("top-current-deck");
    if (topDirection) topDirection.textContent = App.deckManager.getCurrentDirection() ? App.deckManager.getCurrentDirection().name : "Выберите направление";
    if (topDeck) topDeck.textContent = deck ? deck.name : "колода";
  }

  function renderBrowse() {
    var deck = App.deckManager.getCurrentDeck();
    var cards = deck ? deck.cards : [];
    var idx = App.navigation.getIndex();
    var card = cards[idx] || null;

    if (card && lastShownStudyCardId !== "browse:" + card.id) {
      lastShownStudyCardId = "browse:" + card.id;
      card.showCount = (card.showCount || 0) + 1;
      App.cardManager.recordShow(card.id).catch(function (err) { console.warn("Не удалось сохранить количество показов:", err); });
    }
    App.render.renderCard(cardEls, card);
    App.render.renderProgress($("progress"), idx, App.navigation.getTotal());
    App.flip.reset(cardEls.card);
    stopAudio();
    $("hint").textContent = "нажмите на карточку, чтобы перевернуть";
  }

  // Заполняет подписи-прогнозы под кнопками оценки для ТЕКУЩЕЙ карточки сессии
  // и показывает/прячет саму строку кнопок в зависимости от того, перевёрнута ли карточка:
  // оценивать «как вспомнилось» можно только увидев ответ.
  function syncGradeRow() {
    var card = App.study.current();
    var flipped = App.flip.isFlipped(cardEls.card);
    var row = $("grade-row");
    row.hidden = !flipped || !card;
    if (flipped && card) {
      var preview = App.study.preview(card);
      App.srs.RATINGS.forEach(function (r) {
        $("grade-sub-" + r).textContent = App.srs.formatDays(preview[r]);
      });
    }
    $("hint").textContent = flipped ? "оцените, как вспомнилась карточка" : "нажмите на карточку, чтобы увидеть ответ";
  }

  var lastShownStudyCardId = null;

  function renderStudy() {
    var card = App.study.current();
    $("study-done").hidden = !!card;
    cardEls.card.hidden = !card;
    $("grade-row").hidden = true;

    if (!card) {
      $("progress").textContent = "Готово";
      stopAudio();
      return;
    }
    // В учебе каждый новый показ считается, включая повтор после «Снова».
    lastShownStudyCardId = card.id;
    card.showCount = (card.showCount || 0) + 1;
    App.cardManager.recordShow(card.id).catch(function (err) {
      console.warn("Не удалось сохранить количество показов:", err);
    });
    App.render.renderCard(cardEls, card);
    App.flip.reset(cardEls.card);
    stopAudio();
    $("progress").textContent = "Осталось: " + App.study.remaining();
    $("hint").textContent = "нажмите на карточку, чтобы увидеть ответ";
  }

  function refreshCard() {
    if (studyMode) renderStudy(); else renderBrowse();
    App.actions.updateState();
  }

  function selectDirection(directionId) {
    App.deckManager.setCurrentDirection(directionId);
    var deck = App.deckManager.getCurrentDeck();
    App.navigation.reset(deck ? deck.cards.length : 0);
    App.render.renderDeckOptions($("deck-select"), App.deckManager.getVisibleDecks(), deck ? deck.id : null);
    updateStudyBadge();
    refreshCard();
  }

  function selectDeck(deckId) {
    App.deckManager.setCurrentDeck(deckId);
    var deck = App.deckManager.getCurrentDeck();
    App.navigation.reset(deck ? deck.cards.length : 0);
    updateStudyBadge();
    refreshCard();
  }

  function currentCard() {
    if (studyMode) return App.study.current();
    var deck = App.deckManager.getCurrentDeck();
    return deck ? (deck.cards[App.navigation.getIndex()] || null) : null;
  }

  function startStudy() {
    var deck = App.deckManager.getCurrentDeck();
    if (!deck) return;
    App.study.start(deck.cards);
    lastShownStudyCardId = null;
    if (!App.study.remaining()) { updateStudyBadge(); return; } // кнопка и так задизейблена — подстраховка
    studyMode = true;
    setStudyChrome(true);
    refreshCard();
  }

  function exitStudy() {
    App.study.stop();
    lastShownStudyCardId = null;
    studyMode = false;
    setStudyChrome(false);
    // Оценки могли поменять due у карточек — обновляем всё разом, как после любой мутации.
    App.deckManager.refresh();
    var deck = App.deckManager.getCurrentDeck();
    App.navigation.reset(deck ? deck.cards.length : 0);
    App.render.renderDirectionOptions($("direction-select"), App.deckManager.getDirections(), App.deckManager.getCurrentDirection() ? App.deckManager.getCurrentDirection().id : null);
    App.render.renderDeckOptions($("deck-select"), App.deckManager.getVisibleDecks(), deck ? deck.id : null);
    updateStudyBadge();
    refreshCard();
  }

  // Переключает верхнюю панель и низ экрана между «обзором» и «сессией»:
  // во время сессии нельзя сменить/добавить/удалить колоду или карточку (см. actions.js).
  function setStudyChrome(active) {
    $("direction-select").disabled = active;
    $("deck-select").disabled = active;
    $("controls-browse").hidden = active;
    $("controls-study").hidden = !active;
    $("study-done").hidden = true;
    updateStudyBadge();
    setGradeBusy(false);
  }

  function setGradeBusy(busy) {
    document.querySelectorAll("#grade-row .grade-btn, #btn-study-skip").forEach(function (btn) {
      btn.disabled = !!busy;
    });
  }

  async function gradeCurrent(rating) {
    if (App.study.isBusy()) return;
    setGradeBusy(true);
    try {
      await App.study.grade(rating);
      refreshCard();
    } catch (err) {
      App.toast.show("Не удалось сохранить оценку: " + (err && err.message ? err.message : err), "error");
    } finally {
      setGradeBusy(false);
    }
  }

  async function skipCurrent() {
    if (App.study.isBusy()) return;
    setGradeBusy(true);
    try {
      await App.study.skip();
      refreshCard();
    } catch (err) {
      App.toast.show("Не удалось пропустить карточку: " + (err && err.message ? err.message : err), "error");
    } finally {
      setGradeBusy(false);
    }
  }

  // Перечитать колоды из базы и перерисовать всё после любой мутации (добавление, правка,
  // удаление, колоды). focus: { cardId } — перейти на эту карточку; { index } — остаться
  // на позиции (с поправкой, если карточек стало меньше); без focus — в начало колоды.
  function reloadUi(focus) {
    App.deckManager.refresh();
    var deck = App.deckManager.getCurrentDeck();
    var cards = deck ? deck.cards : [];
    App.navigation.reset(cards.length);
    if (focus && focus.cardId) {
      App.navigation.goTo(cards.findIndex(function (c) { return c.id === focus.cardId; }));
    } else if (focus && focus.index != null) {
      App.navigation.goTo(Math.min(focus.index, cards.length - 1));
    }
    App.render.renderDirectionOptions($("direction-select"), App.deckManager.getDirections(), App.deckManager.getCurrentDirection() ? App.deckManager.getCurrentDirection().id : null);
    App.render.renderDeckOptions($("deck-select"), App.deckManager.getVisibleDecks(), deck ? deck.id : null);
    updateStudyBadge();
    refreshCard();
  }

  // Стрелки листают карточки только когда пользователь не печатает и диалог закрыт
  function isTypingContext(target) {
    if (document.querySelector("dialog[open]")) return true;
    return !!(target && target.closest && target.closest("input, textarea, select, [contenteditable]"));
  }

  function playAudio(url) {
    if (!url) return;
    try {
      if (currentAudio) currentAudio.pause();
      currentAudio = new Audio(url);
      currentAudio.play().catch(function (err) {
        console.warn("Не удалось воспроизвести аудио:", err);
      });
    } catch (err) {
      console.warn("Ошибка аудио:", err);
    }
  }

  function openVideo(url) {
    if (!url) return;
    // Лёгкий вариант без модалки на этом этапе — открываем в новой вкладке.
    // Позже можно заменить на встроенный проигрыватель/лайтбокс.
    window.open(url, "_blank", "noopener");
  }

  function init() {
    cardEls.card = $("card");
    cardEls.term = $("card-term");
    cardEls.transcription = $("card-transcription");
    cardEls.image = $("card-image");
    cardEls.description = $("card-description");
    cardEls.examples = $("card-examples");
    cardEls.audioBtn = $("btn-audio");
    cardEls.videoBtn = $("btn-video");
    cardEls.showCount = $("card-show-count");

    bindEvents();
    App.stats.init();
    App.addDialog.init({
      getDeck: App.deckManager.getCurrentDeck,
      onAdded: function (firstCardId) { reloadUi({ cardId: firstCardId }); }
    });
    App.editDialog.init({
      onChanged: function (cardId) {
        reloadUi({ cardId: cardId });
        App.toast.show("Карточка сохранена");
      }
    });
    App.directionDialog.init({
      onChanged: function (mode) {
        reloadUi();
        App.toast.show(mode === "rename" ? "Направление переименовано" : "Направление создано");
      }
    });
    App.syncDialog.init({
      onImported: function () {
        reloadUi();
        App.toast.show("Данные загружены из Google Drive");
      }
    });
    App.deckDialog.init({
      onChanged: function (mode) {
        // переименование — остаёмся на той же карточке; новая колода открывается с начала
        reloadUi(mode === "rename" ? { index: App.navigation.getIndex() } : undefined);
        App.toast.show(mode === "rename" ? "Колода переименована" : "Колода создана");
      }
    });
    App.importDialog.init({
      onImported: function (firstDeckId) {
        if (firstDeckId) App.deckManager.setCurrentDeck(firstDeckId);
        reloadUi();
      }
    });
    App.actions.init({ reload: reloadUi, getCurrentCard: currentCard, isStudying: function () { return studyMode; } });
    App.menu.init({ onAction: App.actions.handle });

    // UI строим только после того, как база готова (первый запуск —
    // это ещё и заливка мок-данных из seed.js, см. db.js).
    App.db.init()
      .then(function () {
        // Медиа из OPFS (файлы карточек, добавленных руками) → blob-URL для резолвера
        return App.mediaStore.init();
      })
      .then(function () {
        App.deckManager.init();
        var firstDeck = App.deckManager.getCurrentDeck();
        App.navigation.reset(firstDeck ? firstDeck.cards.length : 0);
        App.render.renderDirectionOptions($("direction-select"), App.deckManager.getDirections(), App.deckManager.getCurrentDirection() ? App.deckManager.getCurrentDirection().id : null);
        App.render.renderDeckOptions($("deck-select"), App.deckManager.getVisibleDecks(), firstDeck ? firstDeck.id : null);
        updateStudyBadge();
        refreshCard(); // заодно включает кнопки «+» и «⋯» (App.actions.updateState)
      })
      .catch(function (err) {
        console.error("[app] База не поднялась:", err);
        cardEls.term.textContent = "Не удалось открыть базу";
        cardEls.description.textContent = String(err && err.message ? err.message : err);
      });

    // PWA: офлайн-кэш через service worker
    if ("serviceWorker" in navigator) {
      window.addEventListener("load", function () {
        navigator.serviceWorker.register("service-worker.js").catch(function (err) {
          console.warn("Service worker не зарегистрирован:", err);
        });
      });
    }
  }

  // Все addEventListener в одном месте — раньше это было хвостом init(),
  // теперь вынесено, чтобы init() не разрастался при добавлении шагов инициализации.
  function bindEvents() {
    $("direction-select").addEventListener("change", function (e) {
      selectDirection(e.target.value);
    });

    $("deck-select").addEventListener("change", function (e) {
      selectDeck(e.target.value);
    });

    // Переворот по клику/тапу и по Enter/Space (доступность с клавиатуры)
    cardEls.card.addEventListener("click", function () {
      App.flip.toggle(cardEls.card);
      if (studyMode) syncGradeRow();
    });
    cardEls.card.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        App.flip.toggle(cardEls.card);
        if (studyMode) syncGradeRow();
      }
    });

    $("btn-add").addEventListener("click", function () {
      App.addDialog.open();
    });

    $("btn-study").addEventListener("click", startStudy);
    $("btn-study-exit").addEventListener("click", exitStudy);
    $("btn-study-done-exit").addEventListener("click", exitStudy);
    $("grade-row").addEventListener("click", function (e) {
      var btn = e.target.closest(".grade-btn");
      if (btn) gradeCurrent(btn.dataset.rating);
    });
    $("btn-study-skip").addEventListener("click", skipCurrent);

    // Медиакнопки: свой клик, карточку не переворачивают
    cardEls.audioBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      playAudio(e.currentTarget.dataset.url);
    });
    cardEls.videoBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      openVideo(e.currentTarget.dataset.url);
    });

    // Навигация кнопками
    $("btn-prev").addEventListener("click", function () {
      App.navigation.prev();
      refreshCard();
    });
    $("btn-next").addEventListener("click", function () {
      App.navigation.next();
      refreshCard();
    });

    // Навигация стрелками клавиатуры
    document.addEventListener("keydown", function (e) {
      if (isTypingContext(e.target)) return;
      if (studyMode) {
        if (e.key === "ArrowRight") {
          e.preventDefault();
          skipCurrent();
        } else if (e.key === "1" || e.key === "2" || e.key === "3" || e.key === "4") {
          e.preventDefault();
          var rating = ({"1":App.srs.AGAIN,"2":App.srs.HARD,"3":App.srs.GOOD,"4":App.srs.EASY})[e.key];
          gradeCurrent(rating);
        }
        return;
      }
      if (e.key === "ArrowRight") {
        App.navigation.next();
        refreshCard();
      } else if (e.key === "ArrowLeft") {
        App.navigation.prev();
        refreshCard();
      }
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
