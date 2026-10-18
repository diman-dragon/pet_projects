// Google Drive sync: local SQLite + OPFS media, with merge by entity revision.
window.App = window.App || {};
App.googleDriveSync = (function () {
  var ROOT_NAME = "Flashcards";
  var MANIFEST_NAME = "manifest.json";
  var SCOPE = "https://www.googleapis.com/auth/drive.file";
  var clientId = localStorage.getItem("flashcards.googleClientId") || "";
  var tokenClient = null, accessToken = null, folderId = null;

  function setClientId(id){clientId=String(id||"").trim(); if(clientId)localStorage.setItem("flashcards.googleClientId",clientId);else localStorage.removeItem("flashcards.googleClientId"); tokenClient=null; accessToken=null; folderId=null;}
  function getClientId(){return clientId;}
  function waitForGoogle(){return new Promise(function(resolve,reject){var started=Date.now();(function poll(){if(window.google&&google.accounts&&google.accounts.oauth2)return resolve();if(Date.now()-started>10000)return reject(new Error("Google Identity Services не загрузился."));setTimeout(poll,100);})();});}
  async function getToken(){if(!clientId)throw new Error("Укажите Google OAuth Client ID.");await waitForGoogle();if(!tokenClient)tokenClient=google.accounts.oauth2.initTokenClient({client_id:clientId,scope:SCOPE,callback:function(){}});return new Promise(function(resolve,reject){tokenClient.callback=function(r){if(!r||r.error)return reject(new Error((r&&r.error_description)||(r&&r.error)||"Не удалось авторизоваться в Google."));accessToken=r.access_token;resolve(accessToken);};tokenClient.requestAccessToken({prompt:"consent"});});}
  async function api(url,options){var token=accessToken||await getToken(), opts=Object.assign({},options||{});opts.headers=Object.assign({},opts.headers||{},{Authorization:"Bearer "+token});var r=await fetch(url,opts);if(r.status===401){accessToken=null;token=await getToken();opts.headers.Authorization="Bearer "+token;r=await fetch(url,opts);}if(!r.ok)throw new Error("Google Drive: HTTP "+r.status+" — "+(await r.text()).slice(0,300));return r;}
  function q(s){return encodeURIComponent(s);}
  async function findChild(name, mime, parent){var query="name = '"+name.replace(/'/g,"\\'")+"' and trashed = false"+(mime?" and mimeType = '"+mime+"'":"")+(parent?" and '"+parent+"' in parents":"");var r=await api("https://www.googleapis.com/drive/v3/files?q="+q(query)+"&spaces=drive&fields=files(id,name,mimeType,modifiedTime,size,md5Checksum)&pageSize=100",{method:"GET"});var d=await r.json();return d.files&&d.files[0]||null;}
  async function ensureFolder(){if(folderId)return folderId;var folder=await findChild(ROOT_NAME,"application/vnd.google-apps.folder",null);if(!folder){var r=await api("https://www.googleapis.com/drive/v3/files",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:ROOT_NAME,mimeType:"application/vnd.google-apps.folder"})});folder=await r.json();}folderId=folder.id;return folderId;}
  async function uploadJson(name,obj,parent){var existing=await findChild(name,"application/json",parent);var boundary="fc_"+Math.random().toString(36).slice(2);var data=new Blob([JSON.stringify(obj)],{type:"application/json"});var meta={name:name,mimeType:"application/json"};if(parent)meta.parents=[parent];var body=new Blob(["--"+boundary+"\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n",JSON.stringify(meta),"\r\n--"+boundary+"\r\nContent-Type: application/json\r\n\r\n",data,"\r\n--"+boundary+"--"]);var url=existing?"https://www.googleapis.com/upload/drive/v3/files/"+encodeURIComponent(existing.id)+"?uploadType=multipart":"https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart";var r=await api(url,{method:existing?"PATCH":"POST",headers:{"Content-Type":"multipart/related; boundary="+boundary},body:body});return r.json();}
  async function uploadBlob(name,file,parent){var existing=await findChild(name,null,parent);var boundary="fc_"+Math.random().toString(36).slice(2);var meta={name:name,mimeType:file.type||"application/octet-stream"};if(parent)meta.parents=[parent];var body=new Blob(["--"+boundary+"\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n",JSON.stringify(meta),"\r\n--"+boundary+"\r\nContent-Type: "+(file.type||"application/octet-stream")+"\r\n\r\n",file,"\r\n--"+boundary+"--"]);var url=existing?"https://www.googleapis.com/upload/drive/v3/files/"+encodeURIComponent(existing.id)+"?uploadType=multipart":"https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart";var r=await api(url,{method:existing?"PATCH":"POST",headers:{"Content-Type":"multipart/related; boundary="+boundary},body:body});return r.json();}
  async function downloadFile(id){var r=await api("https://www.googleapis.com/drive/v3/files/"+encodeURIComponent(id)+"?alt=media",{method:"GET"});return r.blob();}
  async function sha256(file){var b=await file.arrayBuffer(), h=await crypto.subtle.digest("SHA-256",b), a=new Uint8Array(h), out="";a.forEach(function(x){out+=("0"+x.toString(16)).slice(-2);});return out;}
  function ext(path){var m=/\.([a-z0-9]+)$/i.exec(path||"");return m?m[1].toLowerCase():"bin";}
  function mediaName(hash,path){return hash+"."+ext(path);}
  function choose(a,b){if(!a)return b;if(!b)return a;var at=Number(a.updated_at||0),bt=Number(b.updated_at||0);if(at!==bt)return at>bt?a:b;return String(a.updated_by||"")>=String(b.updated_by||"")?a:b;}
  function mergeMaps(local,remote){var m={};local.forEach(function(x){m[x.id]=x;});remote.forEach(function(x){m[x.id]=choose(m[x.id],x);});return Object.keys(m).map(function(k){return m[k];});}
  function mergeTombstones(local,remote){var m={};local.concat(remote).forEach(function(x){var k=x.entity_type+":"+x.entity_id;m[k]=choose(m[k],x);});return Object.keys(m).map(function(k){return m[k];});}
  function applyTombstones(items,tombs,type){var tm={};tombs.filter(function(t){return t.entity_type===type;}).forEach(function(t){tm[t.entity_id]=t;});return items.filter(function(x){var t=tm[x.id];return !t||choose(x,t)===x;});}
  async function buildManifest(){
    var db=App.db, directions=db.query("SELECT id,name,position,updated_at,updated_by FROM directions ORDER BY position;"), decks=db.query("SELECT id,direction_id,name,position,updated_at,updated_by FROM decks ORDER BY position;"), cards=db.query("SELECT * FROM cards ORDER BY deck_id,position;"), tomb=db.query("SELECT entity_type,entity_id,updated_at,updated_by FROM sync_tombstones;"), events=db.query("SELECT event_uid,card_id,deck_id,event_type,rating,created_at FROM study_events ORDER BY id;");
    var media=[], seen={};
    for(var i=0;i<cards.length;i++){for(var j=0;j<3;j++){var path=[cards[i].image,cards[i].audio,cards[i].video][j];if(!path||seen[path])continue;seen[path]=true;var f=await App.mediaStore.read(path);if(f){media.push({path:path,hash:await sha256(f),name:mediaName(await sha256(f),path),type:f.type||"application/octet-stream",size:f.size});}}}
    return {app:"flashcards-app",syncVersion:3,deviceId:db.getDeviceId(),exportedAt:new Date().toISOString(),directions:directions,decks:decks,cards:cards,tombstones:tomb,studyEvents:events,media:media};
  }
  async function mergeManifests(local,remote){
    remote=remote||{directions:[],decks:[],cards:[],tombstones:[],studyEvents:[],media:[]};
    var tomb=mergeTombstones(local.tombstones||[],remote.tombstones||[]);
    var dirs=applyTombstones(mergeMaps(local.directions||[],remote.directions||[]),tomb,"direction");
    var decks=applyTombstones(mergeMaps(local.decks||[],remote.decks||[]),tomb,"deck").filter(function(d){return dirs.some(function(x){return x.id===d.direction_id;});});
    var cards=applyTombstones(mergeMaps(local.cards||[],remote.cards||[]),tomb,"card").filter(function(c){return decks.some(function(d){return d.id===c.deck_id;});});
    var em={};(local.studyEvents||[]).concat(remote.studyEvents||[]).forEach(function(e){if(e.event_uid)em[e.event_uid]=e;});
    var events=Object.keys(em).map(function(k){return em[k];});
    var showCounts={}; events.forEach(function(e){if(e.event_type==="show"&&e.card_id)showCounts[e.card_id]=(showCounts[e.card_id]||0)+1;});
    cards.forEach(function(c){if(showCounts[c.id]!=null)c.show_count=showCounts[c.id];});
    var mm={};(local.media||[]).concat(remote.media||[]).forEach(function(m){mm[m.path]=m;});
    var livePaths={};cards.forEach(function(c){[c.image,c.audio,c.video].forEach(function(p){if(p)livePaths[p]=true;});});
    var media=Object.keys(livePaths).map(function(p){return mm[p];}).filter(Boolean);
    return {app:"flashcards-app",syncVersion:3,deviceId:App.db.getDeviceId(),exportedAt:new Date().toISOString(),directions:dirs,decks:decks,cards:cards,tombstones:tomb,studyEvents:events,media:media};
  }
  async function applyManifest(m){
    var db=App.db.getInstance();db.run("BEGIN;");try{
      db.run("DELETE FROM cards;");db.run("DELETE FROM decks;");db.run("DELETE FROM directions;");db.run("DELETE FROM sync_tombstones;");db.run("DELETE FROM study_events;");
      var d=db.prepare("INSERT INTO directions(id,name,position,updated_at,updated_by) VALUES(?,?,?,?,?);");m.directions.forEach(function(x){d.run([x.id,x.name,x.position||0,x.updated_at||0,x.updated_by||""]);});d.free();
      var k=db.prepare("INSERT INTO decks(id,direction_id,name,position,updated_at,updated_by) VALUES(?,?,?,?,?,?);");m.decks.forEach(function(x){k.run([x.id,x.direction_id,x.name,x.position||0,x.updated_at||0,x.updated_by||""]);});k.free();
      var c=db.prepare("INSERT INTO cards(id,deck_id,position,front,back,transcription,image,audio,video,examples,due,interval_days,ease,reps,lapses,show_count,learned,updated_at,updated_by) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);");m.cards.forEach(function(x){c.run([x.id,x.deck_id,x.position||0,x.front,x.back,x.transcription||null,x.image||null,x.audio||null,x.video||null,x.examples||null,x.due||0,x.interval_days||0,x.ease==null?2.5:x.ease,x.reps||0,x.lapses||0,x.show_count||0,x.learned?1:0,x.updated_at||0,x.updated_by||""]);});c.free();
      var t=db.prepare("INSERT INTO sync_tombstones(entity_type,entity_id,updated_at,updated_by) VALUES(?,?,?,?);");(m.tombstones||[]).forEach(function(x){t.run([x.entity_type,x.entity_id,x.updated_at,x.updated_by]);});t.free();
      var e=db.prepare("INSERT OR IGNORE INTO study_events(event_uid,card_id,deck_id,event_type,rating,created_at) VALUES(?,?,?,?,?,?);");(m.studyEvents||[]).forEach(function(x){e.run([x.event_uid,x.card_id,x.deck_id,x.event_type,x.rating||null,x.created_at]);});e.free();db.run("COMMIT;");
    }catch(err){try{db.run("ROLLBACK;");}catch(e){}throw err;} await App.db.persist();
  }
  async function sync(mode){
    mode=mode||"sync"; await getToken(); var root=await ensureFolder(); var manifestFile=await findChild(MANIFEST_NAME,"application/json",root); var local=await buildManifest(); var remote=null;
    if(manifestFile){var r=await downloadFile(manifestFile.id);remote=JSON.parse(await r.text());}
    if(!remote && mode==="pull") throw new Error("В Google Drive ещё нет синхронизации. Сначала выполните отправку.");
    var merged=await mergeManifests(local,remote);
    // Получаем отсутствующие/обновлённые медиа из Drive. OPFS остаётся основным локальным хранилищем.
    for(var i=0;i<(merged.media||[]).length;i++){
      var m=merged.media[i], localFile=await App.mediaStore.read(m.path);
      if(localFile && await sha256(localFile)===m.hash) continue;
      var mf=await findChild(m.name,null,root);
      if(mf){var blob=await downloadFile(mf.id);await App.mediaStore.put(m.path,new File([blob],m.name,{type:m.type}));}
      else if(mode==="pull") throw new Error("В Google Drive не найдено медиа: "+m.name);
    }
    await applyManifest(merged);
    if(mode!=="pull") {
      for(var j=0;j<(merged.media||[]).length;j++){var mm=merged.media[j],lf=await App.mediaStore.read(mm.path);if(!lf)continue;var existing=await findChild(mm.name,null,root);if(!existing)await uploadBlob(mm.name,lf,root);}
      await uploadJson(MANIFEST_NAME,merged,root);
    }
    var oldPaths=(local.media||[]).map(function(x){return x.path;}),live={};(merged.media||[]).forEach(function(x){live[x.path]=true;});
    for(var z=0;z<oldPaths.length;z++)if(!live[oldPaths[z]])await App.mediaStore.remove(oldPaths[z]);
    return {directions:merged.directions.length,decks:merged.decks.length,cards:merged.cards.length,media:merged.media.length,events:merged.studyEvents.length,mode:mode};
  }

  function locked(mode){
    if(navigator.locks&&navigator.locks.request)return navigator.locks.request("flashcards-sync",{mode:"exclusive"},function(){return sync(mode);});
    return sync(mode);
  }
  return {setClientId:setClientId,getClientId:getClientId,connect:getToken,sync:function(){return locked("sync");},push:function(){return locked("push");},pull:function(){return locked("pull");},buildManifest:buildManifest};
})();
