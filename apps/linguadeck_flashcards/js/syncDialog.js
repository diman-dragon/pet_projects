window.App = window.App || {};
App.syncDialog = (function () {
  var opts={},dialog,clientId,status,connect,sync,push,pull;
  function message(text,kind){status.textContent=text||"";status.hidden=!text;status.className="status"+(kind?" is-"+kind:"");}
  function busy(v){connect.disabled=v;sync.disabled=v;push.disabled=v;pull.disabled=v;}
  async function connectGoogle(){busy(true);message("Подключение…");try{App.googleDriveSync.setClientId(clientId.value);await App.googleDriveSync.connect();message("Google Drive подключён ✓","ok");}catch(e){message(e.message||String(e),"error");}finally{busy(false);}}
  async function run(mode,label){busy(true);message(label);try{App.googleDriveSync.setClientId(clientId.value);var r=await App.googleDriveSync[mode]();message("Готово ✓ · "+r.cards+" карточек · "+r.media+" медиа","ok");if(opts.onImported)opts.onImported();}catch(e){console.error(e);message(e.message||String(e),"error");}finally{busy(false);}}
  function open(){clientId.value=App.googleDriveSync.getClientId();message("");dialog.showModal();}
  function init(options){opts=options||{};dialog=document.getElementById("sync-dialog");clientId=document.getElementById("sync-client-id");status=document.getElementById("sync-status");connect=document.getElementById("sync-connect");sync=document.getElementById("sync-sync");push=document.getElementById("sync-push");pull=document.getElementById("sync-pull");document.getElementById("sync-close").addEventListener("click",function(){dialog.close();});document.getElementById("sync-cancel").addEventListener("click",function(){dialog.close();});connect.addEventListener("click",connectGoogle);sync.addEventListener("click",function(){run("sync","Объединяю изменения и синхронизирую…")});push.addEventListener("click",function(){run("push","Объединяю локальные изменения и отправляю их…")});pull.addEventListener("click",function(){run("pull","Получаю изменения с Google Drive…")});}
  return {init:init,open:open};
})();
