// cardManager.js — карточки + локальные изменения/SRS/события для синхронизации.
window.App = window.App || {};
App.cardManager = (function () {
  var INSERT_SQL =
    "INSERT INTO cards (id, deck_id, position, front, back, transcription, image, audio, video, examples, due, interval_days, ease, reps, lapses, show_count, learned, updated_at, updated_by) " +
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

  function now() { return Date.now(); }
  function touch(id) { return [now(), App.db.getDeviceId()]; }
  function newUid() { return App.db.getDeviceId() + "-" + now() + "-" + Math.random().toString(36).slice(2, 10); }
  function event(cardId, deckId, type, rating) {
    App.db.getInstance().run("INSERT OR IGNORE INTO study_events (event_uid,card_id,deck_id,event_type,rating,created_at) VALUES (?,?,?,?,?,?);", [newUid(), cardId, deckId, type, rating || null, now()]);
  }
  function rowToCard(row) {
    return { id: row.id, front: row.front, back: row.back, transcription: row.transcription || undefined,
      image: App.mediaStore.resolve(row.image) || undefined, audio: App.mediaStore.resolve(row.audio) || undefined, video: App.mediaStore.resolve(row.video) || undefined,
      examples: (function () { if (!row.examples) return undefined; try { return JSON.parse(row.examples); } catch (e) { return undefined; } })(),
      due: row.due, interval: row.interval_days, ease: row.ease, reps: row.reps, lapses: row.lapses, showCount: row.show_count || 0, learned: !!row.learned,
      updatedAt: row.updated_at || 0, updatedBy: row.updated_by || "" };
  }
  function getCardsForDeck(deckId) { return getRawForDeck(deckId).map(rowToCard); }
  function getRawForDeck(deckId) { return App.db.query("SELECT * FROM cards WHERE deck_id = ? ORDER BY position;", [deckId]); }
  function getRaw(id) { var rows = App.db.query("SELECT * FROM cards WHERE id = ?;", [id]); return rows.length ? rows[0] : null; }

  async function updateCard(id, f) {
    var t=touch(id), raw=getRaw(id);
    await App.db.run("UPDATE cards SET front=?,back=?,transcription=?,image=?,audio=?,examples=?,updated_at=?,updated_by=? WHERE id=?;",
      [f.front,f.back,f.transcription||null,f.image||null,f.audio||null,f.examples&&f.examples.length?JSON.stringify(f.examples):null,t[0],t[1],id]);
  }
  async function updateSrs(id, f) {
    var raw=getRaw(id), t=touch(id);
    await App.db.transaction([
      ["UPDATE cards SET due=?,interval_days=?,ease=?,reps=?,lapses=?,learned=?,updated_at=?,updated_by=? WHERE id=?;",[f.due,f.interval,f.ease,f.reps,f.lapses,f.learned?1:0,t[0],t[1],id]],
      ["INSERT OR IGNORE INTO study_events (event_uid,card_id,deck_id,event_type,rating,created_at) VALUES (?,?,?,?,?,?);",[newUid(),id,raw?raw.deck_id:null,"grade",f.rating||null,now()]]
    ]);
  }
  async function recordShow(id) {
    var raw=getRaw(id), t=touch(id);
    await App.db.transaction([
      ["UPDATE cards SET show_count=show_count+1,updated_at=?,updated_by=? WHERE id=?;",[t[0],t[1],id]],
      ["INSERT OR IGNORE INTO study_events (event_uid,card_id,deck_id,event_type,rating,created_at) VALUES (?,?,?,?,?,?);",[newUid(),id,raw?raw.deck_id:null,"show",null,now()]]
    ]);
  }
  async function skip(id) {
    var raw=getRaw(id), t=touch(id);
    await App.db.transaction([
      ["UPDATE cards SET learned=0,updated_at=?,updated_by=? WHERE id=?;",[t[0],t[1],id]],
      ["INSERT OR IGNORE INTO study_events (event_uid,card_id,deck_id,event_type,rating,created_at) VALUES (?,?,?,?,?,?);",[newUid(),id,raw?raw.deck_id:null,"skip",null,now()]]
    ]);
  }
  async function deleteCard(id) {
    var raw=getRaw(id); if(!raw) return;
    var t=touch(id);
    await App.db.transaction([
      ["INSERT OR REPLACE INTO sync_tombstones(entity_type,entity_id,updated_at,updated_by) VALUES ('card',?,?,?);",[id,t[0],t[1]]],
      ["DELETE FROM cards WHERE id=?;",[id]]
    ]);
  }
  function newId(deckId) { var rnd=(window.crypto&&crypto.randomUUID)?crypto.randomUUID().slice(0,8):Math.random().toString(36).slice(2,10).padEnd(8,"0"); return deckId+"-"+rnd; }
  function nextPosition(deckId) { var rows=App.db.query("SELECT COALESCE(MAX(position),-1)+1 AS p FROM cards WHERE deck_id=?;",[deckId]); return rows.length?rows[0].p:0; }
  async function addCards(deckId,cards) {
    var pos=nextPosition(deckId), ids=[], t=now(), by=App.db.getDeviceId();
    var params=cards.map(function(c){var id=c.id||newId(deckId);ids.push(id);return [id,deckId,pos++,c.front,c.back,c.transcription||null,c.image||null,c.audio||null,c.video||null,c.examples&&c.examples.length?JSON.stringify(c.examples):null,
      Number.isFinite(Number(c.due))?Number(c.due):0,Number.isFinite(Number(c.interval))?Math.max(0,Number(c.interval)):0,Number.isFinite(Number(c.ease))?Math.max(1.3,Number(c.ease)):2.5,
      Number.isFinite(Number(c.reps))?Math.max(0,Number(c.reps)):0,Number.isFinite(Number(c.lapses))?Math.max(0,Number(c.lapses)):0,Number.isFinite(Number(c.showCount))?Math.max(0,Number(c.showCount)):0,c.learned?1:0,
      Number.isFinite(Number(c.updatedAt))?Number(c.updatedAt):t,c.updatedBy||by];});
    try { await App.db.runMany(INSERT_SQL,params); } catch(err){ try{var marks=ids.map(function(){return "?";}).join(",");App.db.getInstance().run("DELETE FROM cards WHERE id IN ("+marks+");",ids);}catch(e){} throw err; }
    return ids;
  }
  return {getCardsForDeck,getRawForDeck,getRaw,updateCard,updateSrs,recordShow,skip,deleteCard,addCards,newId};
})();
