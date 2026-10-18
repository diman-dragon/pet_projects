// deckManager.js
// Иерархия: направление → колода → карточки.

window.App = window.App || {};

App.deckManager = (function () {
  var NAME_MAX = 60;
  var directions = [];
  var decks = [];
  var currentDirectionId = null;
  var currentDeckId = null;

  function refresh() {
    directions = App.db.query("SELECT id, name FROM directions ORDER BY position;").map(function (row) {
      return { id: row.id, name: row.name };
    });

    var rows = App.db.query("SELECT id, direction_id, name FROM decks ORDER BY position;");
    decks = rows.map(function (row) {
      return {
        id: row.id,
        directionId: row.direction_id,
        name: row.name,
        cards: App.cardManager.getCardsForDeck(row.id)
      };
    });

    if (!directions.some(function (d) { return d.id === currentDirectionId; })) {
      currentDirectionId = directions.length ? directions[0].id : null;
    }

    var visible = decks.filter(function (d) { return d.directionId === currentDirectionId; });
    if (!visible.some(function (d) { return d.id === currentDeckId; })) {
      currentDeckId = visible.length ? visible[0].id : null;
    }
  }

  function init() { refresh(); }
  function getDirections() { return directions; }
  function getDecks() { return decks; }
  function getVisibleDecks() {
    return decks.filter(function (d) { return d.directionId === currentDirectionId; });
  }
  function getCurrentDirection() {
    return directions.find(function (d) { return d.id === currentDirectionId; }) || null;
  }
  function getCurrentDeck() {
    return decks.find(function (d) { return d.id === currentDeckId; }) || null;
  }

  function setCurrentDirection(id) {
    if (!directions.some(function (d) { return d.id === id; })) return false;
    currentDirectionId = id;
    var visible = getVisibleDecks();
    currentDeckId = visible.length ? visible[0].id : null;
    return true;
  }

  function setCurrentDeck(id) {
    var deck = decks.find(function (d) { return d.id === id; });
    if (!deck || deck.directionId !== currentDirectionId) return false;
    currentDeckId = id;
    return true;
  }

  function validateName(name, exceptId, collection) {
    var n = String(name == null ? "" : name).trim().replace(/\s+/g, " ");
    if (!n) throw new Error("Введите название.");
    if (n.length > NAME_MAX) throw new Error("Слишком длинное название (максимум " + NAME_MAX + " символов).");
    var low = n.toLowerCase();
    var clash = collection.some(function (item) { return item.id !== exceptId && item.name.toLowerCase() === low; });
    if (clash) throw new Error("Название уже используется.");
    return n;
  }

  function validateDirectionName(name, exceptId) {
    return validateName(name, exceptId, directions);
  }

  function validateDeckName(name, exceptId) {
    var visible = getVisibleDecks();
    return validateName(name, exceptId, visible);
  }

  function newId(prefix) {
    var rnd = (window.crypto && crypto.randomUUID)
      ? crypto.randomUUID().slice(0, 8)
      : Math.random().toString(36).slice(2, 10).padEnd(8, "0");
    return prefix + "-" + rnd;
  }

  async function createDirection(name) {
    var clean = validateDirectionName(name);
    var id = newId("direction");
    var pos = App.db.query("SELECT COALESCE(MAX(position), -1) + 1 AS p FROM directions;")[0].p;
    await App.db.run("INSERT INTO directions (id, name, position, updated_at, updated_by) VALUES (?, ?, ?, ?, ?);", [id, clean, pos, Date.now(), App.db.getDeviceId()]);
    refresh();
    currentDirectionId = id;
    currentDeckId = null;
    return id;
  }

  async function renameDirection(id, name) {
    var clean = validateDirectionName(name, id);
    await App.db.run("UPDATE directions SET name = ?, updated_at = ?, updated_by = ? WHERE id = ?;", [clean, Date.now(), App.db.getDeviceId(), id]);
    refresh();
  }

  async function deleteDirection(id) {
    var direction = directions.find(function (d) { return d.id === id; });
    if (!direction) return;
    var ids = decks.filter(function (d) { return d.directionId === id; }).map(function (d) { return d.id; });
    var t=Date.now(), by=App.db.getDeviceId(), stmts=[["INSERT OR REPLACE INTO sync_tombstones(entity_type,entity_id,updated_at,updated_by) VALUES ('direction',?,?,?);",[id,t,by]]];
    ids.forEach(function(deckId){ stmts.push(["INSERT OR REPLACE INTO sync_tombstones(entity_type,entity_id,updated_at,updated_by) VALUES ('deck',?,?,?);",[deckId,t,by]]); });
    if(ids.length) { var cardIds=App.db.query("SELECT id FROM cards WHERE deck_id IN ("+ids.map(function(){return "?";}).join(",")+");",ids); cardIds.forEach(function(r){stmts.push(["INSERT OR REPLACE INTO sync_tombstones(entity_type,entity_id,updated_at,updated_by) VALUES ('card',?,?,?);",[r.id,t,by]]);}); }
    stmts.push(["DELETE FROM cards WHERE deck_id IN (" + (ids.length ? ids.map(function () { return "?"; }).join(",") : "SELECT NULL") + ");", ids]);
    stmts.push(["DELETE FROM decks WHERE direction_id = ?;", [id]]);
    stmts.push(["DELETE FROM directions WHERE id = ?;", [id]]);
    await App.db.transaction(stmts);
    await Promise.all(ids.map(function (deckId) { return App.mediaStore.removeDir(deckId); }));
    refresh();
  }

  async function createDeck(name, directionId) {
    var dirId = directionId || currentDirectionId;
    if (!dirId || !directions.some(function (d) { return d.id === dirId; })) {
      throw new Error("Сначала создайте направление.");
    }
    var clean = validateName(name, null, decks.filter(function (d) { return d.directionId === dirId; }));
    var id = newId("deck");
    var posRow = App.db.query("SELECT COALESCE(MAX(position), -1) + 1 AS p FROM decks WHERE direction_id = ?;", [dirId]);
    var pos = posRow.length ? posRow[0].p : 0;
    await App.db.run("INSERT INTO decks (id, direction_id, name, position, updated_at, updated_by) VALUES (?, ?, ?, ?, ?, ?);", [id, dirId, clean, pos, Date.now(), App.db.getDeviceId()]);
    refresh();
    currentDirectionId = dirId;
    currentDeckId = id;
    return id;
  }

  async function renameDeck(id, name) {
    var deck = decks.find(function (d) { return d.id === id; });
    if (!deck) throw new Error("Колода не найдена.");
    var clean = validateName(name, id, decks.filter(function (d) { return d.directionId === deck.directionId; }));
    await App.db.run("UPDATE decks SET name = ?, updated_at = ?, updated_by = ? WHERE id = ?;", [clean, Date.now(), App.db.getDeviceId(), id]);
    refresh();
  }

  async function deleteDeck(id) {
    var deck = decks.find(function (d) { return d.id === id; });
    if (!deck) return;
    var t=Date.now(), by=App.db.getDeviceId(), cardIds=App.db.query("SELECT id FROM cards WHERE deck_id = ?;",[id]);
    var stmts=[["INSERT OR REPLACE INTO sync_tombstones(entity_type,entity_id,updated_at,updated_by) VALUES ('deck',?,?,?);",[id,t,by]]];
    cardIds.forEach(function(r){stmts.push(["INSERT OR REPLACE INTO sync_tombstones(entity_type,entity_id,updated_at,updated_by) VALUES ('card',?,?,?);",[r.id,t,by]]);});
    stmts.push(["DELETE FROM cards WHERE deck_id = ?;", [id]],["DELETE FROM decks WHERE id = ?;", [id]]);
    await App.db.transaction(stmts);
    await App.mediaStore.removeDir(id);
    refresh();
  }

  return {
    init: init, refresh: refresh,
    getDirections: getDirections, getDecks: getDecks, getVisibleDecks: getVisibleDecks,
    getCurrentDirection: getCurrentDirection, getCurrentDeck: getCurrentDeck,
    setCurrentDirection: setCurrentDirection, setCurrentDeck: setCurrentDeck,
    createDirection: createDirection, renameDirection: renameDirection, deleteDirection: deleteDirection,
    createDeck: createDeck, renameDeck: renameDeck, deleteDeck: deleteDeck,
    NAME_MAX: NAME_MAX
  };
})();
