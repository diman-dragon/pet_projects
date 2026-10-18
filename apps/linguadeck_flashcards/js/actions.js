// actions.js
// Действия из меню «⋯»: что делает каждый пункт, порядок «подтвердить → выполнить → перерисовать».
// Сама запись в базу — в editCard.js / deckManager.js; окна — в *Dialog.js и feedback.js.

window.App = window.App || {};

App.actions = (function () {
  var ctx = { reload: null, getCurrentCard: null, isStudying: function () { return false; } };

  function errText(err) {
    return err && err.message ? err.message : String(err);
  }

  async function deleteCard() {
    var card = ctx.getCurrentCard();
    if (!card) return;
    var ok = await App.confirm.ask({
      title: "Удалить карточку?",
      message: "«" + App.util.short(card.front, 60) + "» будет удалена вместе с её файлами. Это нельзя отменить."
    });
    if (!ok) return;
    var index = App.navigation.getIndex();
    try {
      await App.editCard.remove(card.id);
      ctx.reload({ index: index }); // остаёмся на том же месте (или на последней, если удалили последнюю)
      App.toast.show("Карточка удалена");
    } catch (err) {
      App.toast.show("Не удалось удалить: " + errText(err), "error");
    }
  }

  async function deleteDirection() {
    var direction = App.deckManager.getCurrentDirection();
    if (!direction) return;
    var count = App.deckManager.getDecks().filter(function (d) { return d.directionId === direction.id; }).length;
    var ok = await App.confirm.ask({
      title: "Удалить направление?",
      message: "«" + App.util.short(direction.name, 60) + "» будет удалено вместе с " + count + " " +
        App.util.plural(count, ["колодой", "колодами", "колодами"]) + ", карточками и файлами. Это нельзя отменить."
    });
    if (!ok) return;
    try {
      await App.deckManager.deleteDirection(direction.id);
      ctx.reload();
      App.toast.show("Направление удалено");
    } catch (err) {
      App.toast.show("Не удалось удалить: " + errText(err), "error");
    }
  }

  async function deleteDeck() {
    var deck = App.deckManager.getCurrentDeck();
    if (!deck) return;
    var n = deck.cards.length;
    var ok = await App.confirm.ask({
      title: "Удалить колоду?",
      message: "Колода «" + App.util.short(deck.name, 60) + "» будет удалена вместе с " + n + " " +
        App.util.plural(n, ["карточкой", "карточками", "карточками"]) + " и их файлами. Это нельзя отменить."
    });
    if (!ok) return;
    try {
      await App.deckManager.deleteDeck(deck.id);
      ctx.reload();
      App.toast.show("Колода удалена");
    } catch (err) {
      App.toast.show("Не удалось удалить: " + errText(err), "error");
    }
  }

  async function exportDeck() {
    var deck = App.deckManager.getCurrentDeck();
    if (!deck) return;
    try {
      await App.backup.exportDeck(deck);
      App.toast.show("Колода экспортирована");
    } catch (err) {
      App.toast.show("Не удалось экспортировать: " + errText(err), "error");
    }
  }

  async function exportAll() {
    var decks = App.deckManager.getDecks();
    if (!decks.length) return;
    try {
      await App.backup.exportAll(decks);
      App.toast.show("Бэкап экспортирован");
    } catch (err) {
      App.toast.show("Не удалось экспортировать: " + errText(err), "error");
    }
  }

  function handle(action) {
    var deck = App.deckManager.getCurrentDeck();
    var direction = App.deckManager.getCurrentDirection();
    var card = ctx.getCurrentCard();
    if (action === "study") document.getElementById("btn-study").click();
    else if (action === "new-card") document.getElementById("btn-add").click();
    else if (action === "edit-card" && card) App.editDialog.open(card.id);
    else if (action === "delete-card") deleteCard();
    else if (action === "new-direction") App.directionDialog.open({ mode: "create" });
    else if (action === "rename-direction" && App.deckManager.getCurrentDirection()) App.directionDialog.open({ mode: "rename", direction: App.deckManager.getCurrentDirection() });
    else if (action === "delete-direction") deleteDirection();
    else if (action === "new-deck") App.deckDialog.open({ mode: "create" });
    else if (action === "rename-deck" && deck) App.deckDialog.open({ mode: "rename", deck: deck });
    else if (action === "delete-deck") deleteDeck();
    else if (action === "export-deck") exportDeck();
    else if (action === "export-all") exportAll();
    else if (action === "import") App.importDialog.open();
    else if (action === "stats") App.stats.open(deck ? deck.id : null);
    else if (action === "sync-drive") App.syncDialog.open();
  }

  // Что доступно прямо сейчас: без колоды нет «+», переименования и удаления колоды;
  // без карточки — правки и удаления карточки. Во время учебной сессии ЛЮБые правки
  // (карточек и колод) недоступны — сессия работает со своей очередью и снимком
  // карточек (study.js), reload() колоды во время неё не вызывается вовсе; сначала
  // «Завершить сессию».
  function updateState() {
    var deck = App.deckManager.getCurrentDeck();
    var direction = App.deckManager.getCurrentDirection();
    var card = ctx.getCurrentCard();
    var studying = ctx.isStudying();
    document.getElementById("btn-add").disabled = !deck || studying;
    var sidebarStudy = document.querySelector("[data-sidebar-study]");
    if (sidebarStudy) sidebarStudy.disabled = studying || !deck || App.srs.dueCount(deck.cards) === 0;
    App.menu.setEnabled("new-card", !!deck && !studying);
    // Во время сессии в меню всё равно нет ни одного доступного пункта — если оставить
    // кнопку активной, меню откроется пустым и не закроется ни по Esc, ни кликом мимо
    // (фокус при открытии некуда ставить — items() пуст). Проще не открывать его вовсе.
    document.getElementById("btn-menu").disabled = studying;
    App.menu.setEnabled("edit-card", !!card && !studying);
    App.menu.setEnabled("delete-card", !!card && !studying);
    App.menu.setEnabled("new-direction", !studying);
    App.menu.setEnabled("rename-direction", !!direction && !studying);
    App.menu.setEnabled("delete-direction", !!direction && !studying);
    App.menu.setEnabled("new-deck", !!direction && !studying);
    App.menu.setEnabled("rename-deck", !!deck && !studying);
    App.menu.setEnabled("delete-deck", !!deck && !studying);
    App.menu.setEnabled("export-deck", !!deck && !studying);
    App.menu.setEnabled("export-all", App.deckManager.getDecks().length > 0 && !studying);
    App.menu.setEnabled("import", !studying);
    App.menu.setEnabled("stats", !studying && App.deckManager.getDecks().length > 0);
    App.menu.setEnabled("sync-drive", !studying);
  }

  function init(options) {
    ctx = options;
  }

  return { init: init, handle: handle, updateState: updateState };
})();
