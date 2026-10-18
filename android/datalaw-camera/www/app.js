(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// rAF не срабатывает, если WebView на паузе — поэтому всегда с запасным таймером, чтобы запуск не завис
const nextFrames = () => Promise.race([new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))), new Promise((r) => setTimeout(r, 150))]);
const DL = window.DL || { name: 'DataLaw Camera', version: 'dev', code: 0, privacyUrl: '#' };

/* ================= Диагностика и тайм-ауты (камера не должна «висеть») ================= */
// Журнал хранится в localStorage: переживает зависание и принудительное закрытие приложения
const DIAG_KEY = 'datalaw.diag';
const diagLog = [];
try {
  const prev = JSON.parse(localStorage.getItem(DIAG_KEY) || '[]');
  if (Array.isArray(prev) && prev.length) { diagLog.push(...prev.slice(-60), '— — — предыдущий запуск выше — — —'); }
} catch {}
function renderDiag() {
  const set = (id, n) => { const el = $(id); if (el) el.textContent = diagLog.slice(-n).join('\n'); };
  set('#diag', 120); set('#bootLog', 7); set('#errLog', 14);
}
function diag(msg) {
  const t = new Date().toISOString().slice(11, 19);
  diagLog.push(`${t} ${msg}`);
  if (diagLog.length > 120) diagLog.shift();
  try { localStorage.setItem(DIAG_KEY, JSON.stringify(diagLog.slice(-100))); } catch {}
  renderDiag();
}
async function copyDiag(btn) {
  const text = `${DL.name} ${DL.version} (${DL.code})\n${navigator.userAgent}\n` + diagLog.join('\n');
  const mark = (ok) => {
    if (btn) { const old = btn.dataset.t || (btn.dataset.t = btn.textContent); btn.textContent = ok ? 'Скопировано ✓' : 'Не удалось — сделайте скриншот'; setTimeout(() => { btn.textContent = old; }, 2500); }
  };
  try { await navigator.clipboard.writeText(text); mark(true); return; } catch {}
  try {
    const ta = document.createElement('textarea'); ta.value = text; ta.style.cssText = 'position:fixed;opacity:0;top:0';
    document.body.append(ta); ta.select(); const ok = document.execCommand('copy'); ta.remove();
    if (ok) { mark(true); return; }
  } catch {}
  mark(false);
}
window.addEventListener('error', (e) => diag('ERR ' + e.message));
window.addEventListener('unhandledrejection', (e) => diag('REJ ' + ((e.reason && e.reason.message) || e.reason)));
const withTimeout = (p, ms, label) => new Promise((res, rej) => {
  const t = setTimeout(() => rej(new Error(`${label}: нет ответа ${Math.round(ms / 1000)} с`)), ms);
  Promise.resolve(p).then((v) => { clearTimeout(t); res(v); }, (e) => { clearTimeout(t); rej(e); });
});
const errMsg = (e) => String((e && e.message) || e || 'неизвестная ошибка');

/* ================= Иконки ================= */
const ICONS = {
  zap: '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
  clock: '<circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15 14"/>',
  sliders: '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1"/>',
  flip: '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.5 9a9 9 0 0 1 14.9-3.4L23 10M1 14l4.6 4.4A9 9 0 0 0 20.5 15"/>',
  lens: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/>',
};
$$('[data-ic]').forEach((el) => {
  el.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[el.dataset.ic]}</svg>`;
});

/* ================= Настройки ================= */
const KEY = 'datalaw.camera.v1';
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch {}
const S = Object.assign({
  res: 1080, aspect: '3:4', grid: true, level: true, hist: false, sound: true, volume: true,
  maxphoto: true, mode: 'photo', filter: 0, timer: 0, nativeEngine: false, blur: 2,
  fx: 'bigeyes', funVideo: false, watermark: false,
}, saved);
S.flash = 'off'; S.facing = 'environment'; S.deviceId = null;
const persist = () => {
  try { localStorage.setItem(KEY, JSON.stringify({ ...S, flash: undefined, facing: undefined, deviceId: undefined })); } catch {}
};
// если прошлый запуск CameraX не завершился (зависание/падение) — этот запуск идёт через WebView
let nativeCrashed = false;
// версия настроек 2: CameraX по умолчанию выключен (включается в Настройках), у всех обновившихся — тоже
if (S.cfgVer !== 2) { S.cfgVer = 2; S.nativeEngine = false; delete S.nxPending; persist(); }
if (S.nxPending) {
  nativeCrashed = true; S.nativeEngine = false; delete S.nxPending; persist();
}

const ASP = { '3:4': 3 / 4, '9:16': 9 / 16, '1:1': 1 };
const ASP_LABEL = { '3:4': '4:3', '9:16': '16:9', '1:1': '1:1' };
const ASP_ORDER = ['3:4', '9:16', '1:1'];
const RES = { 720: [1280, 720], 1080: [1920, 1080], 2160: [3840, 2160] };
const BITRATE = { 720: 5e6, 1080: 10e6, 2160: 35e6 };
const FILTERS = [
  { n: 'Оригинал', css: '' },
  { n: 'Ярко', css: 'saturate(1.5) contrast(1.1)' },
  { n: 'Тёплый', css: 'sepia(.25) saturate(1.3) hue-rotate(-10deg) brightness(1.05)' },
  { n: 'Холод', css: 'saturate(1.1) hue-rotate(15deg) brightness(1.02) contrast(1.05)' },
  { n: 'Винтаж', css: 'sepia(.55) contrast(.9) brightness(1.05) saturate(.9)' },
  { n: 'Кино', css: 'contrast(1.15) saturate(.85) sepia(.15) hue-rotate(-8deg)' },
  { n: 'Фейд', css: 'contrast(.85) brightness(1.1) saturate(.8)' },
  { n: 'Ч/Б', css: 'grayscale(1) contrast(1.1)' },
  { n: 'Нуар', css: 'grayscale(1) contrast(1.6) brightness(.85)' },
];
const curCss = () => FILTERS[S.filter].css;
const FX = [
  { id: 'none', e: '🚫', n: 'Без' },
  { id: 'bigeyes', e: '👀', n: 'Глазастик', face: 1 },
  { id: 'fat', e: '🐷', n: 'Пухляш', face: 1 },
  { id: 'alien', e: '👽', n: 'Пришелец', face: 1 },
  { id: 'nose', e: '👃', n: 'Носач', face: 1 },
  { id: 'mouth', e: '👄', n: 'Ротан', face: 1 },
  { id: 'dog', e: '🐶', n: 'Собака', face: 1 },
  { id: 'clown', e: '🤡', n: 'Клоун', face: 1 },
  { id: 'gent', e: '🎩', n: 'Джентльмен', face: 1 },
  { id: 'shades', e: '😎', n: 'Очки', face: 1 },
  { id: 'crown', e: '👑', n: 'Корона', face: 1 },
  { id: 'devil', e: '😈', n: 'Дьявол', face: 1 },
  { id: 'mirror', e: '↔️', n: 'Зеркало' },
  { id: 'kaleido', e: '🔯', n: 'Калейдоскоп' },
  { id: 'fisheye', e: '🐟', n: 'Рыбий глаз' },
  { id: 'swirl', e: '🌀', n: 'Водоворот' },
  { id: 'pixel', e: '👾', n: 'Пиксели' },
  { id: 'thermal', e: '🔥', n: 'Тепловизор' },
  { id: 'comic', e: '💥', n: 'Комикс' },
  { id: 'ghost', e: '👻', n: 'Призрак' },
];
const fxInfo = () => FX.find((f) => f.id === S.fx) || FX[0];
const isVideoMode = () => S.mode === 'video' || (S.mode === 'fun' && S.funVideo);

/* ================= Состояние ================= */
const video = $('#video'), vf = $('#vf'), shutter = $('#shutter');
const Cap = window.Capacitor;
const native = !!(Cap && Cap.isNativePlatform && Cap.isNativePlatform());
const Saver = native && Cap ? (Cap.registerPlugin ? Cap.registerPlugin('MediaSaver') : Cap.Plugins?.MediaSaver) : null;   // запись для веб-движка
const NC = native && Cap ? (Cap.registerPlugin ? Cap.registerPlugin('NativeCamera') : Cap.Plugins?.NativeCamera) : null;     // CameraX

let NX = false;            // сейчас работает нативный движок
let nxFailed = false;      // CameraX не удалось запустить в этой сессии
const wantNative = () => !!(NC && S.nativeEngine && !nxFailed);

let C = { zoom: null, torch: false, ev: null, pro: null, front: false };
let isFront = false, curZoom = 1;
let stream = null, track = null, wcaps = {}, ic = null, backs = [], webReleased = false;
let busy = false, counting = false, cancelCount = false, burstOn = false;
let rec = null, recChunks = [], recTimer = null, recStart = 0, recRaf = 0, recThumb = '';
let saveQueue = Promise.resolve();
const session = [];
const cv = document.createElement('canvas');
const wmImg = new Image(); wmImg.src = 'brand/dl_watermark.png';

/* ================= Утилиты ================= */
let toastT;
function toast(msg, ms = 1800) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('on');
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('on'), ms);
}
const vib = (ms = 12) => { try { navigator.vibrate && navigator.vibrate(ms); } catch {} };
let actx;
function tone(f0, f1, dur, vol = 0.15, type = 'square') {
  if (!S.sound) return;
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    const t = actx.currentTime, o = actx.createOscillator(), g = actx.createGain();
    o.type = type; o.frequency.setValueAtTime(f0, t); o.frequency.exponentialRampToValueAtTime(f1, t + dur);
    g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(g).connect(actx.destination); o.start(t); o.stop(t + dur + 0.02);
  } catch {}
}
const clickSound = () => tone(1800, 300, 0.07);
const beep = (f = 880) => tone(f, f, 0.12, 0.1, 'sine');
const stamp = () => {
  const d = new Date(), p = (n) => String(n).padStart(2, '0');
  return `DataLaw_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
};
const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
const b64 = (blob) => new Promise((res, rej) => {
  const r = new FileReader();
  r.onload = () => res(String(r.result).split(',')[1]);
  r.onerror = rej; r.readAsDataURL(blob);
});
const nearest = (arr, v) => arr.reduce((bi, x, i) => (Math.abs(x - v) < Math.abs(arr[bi] - v) ? i : bi), 0);
const vfRect = () => { const b = vf.getBoundingClientRect(); return { x: b.left, y: b.top, w: b.width, h: b.height }; };

function addSession(item) {
  session.push(item);
  if (session.length > 40) { const old = session.shift(); if (old.url) URL.revokeObjectURL(old.url); }
  if (item.thumb) $('#thumbImg').style.backgroundImage = `url(${item.thumb})`;
}

/* ================= Оверлеи + системная кнопка «Назад» ================= */
const ovStack = [];
function openOv(name) {
  $(name === 'sheet' ? '#sheet' : '#viewer').classList.add('on');
  ovStack.push(name);
  try { history.pushState({ ov: name }, ''); } catch {}
}
function popOv() {
  const n = ovStack.pop(); if (!n) return;
  $(n === 'sheet' ? '#sheet' : '#viewer').classList.remove('on');
  if (n === 'viewer') $('#vbody').innerHTML = '';
}
function closeOv() {
  if (!ovStack.length) return;
  if (history.state && history.state.ov) history.back(); else popOv();
}
window.addEventListener('popstate', () => { if (ovStack.length) popOv(); });

/* ================= Загрузочный экран / ошибки ================= */
let bootT1, bootT2;
function showBoot(note) {
  const b = $('#boot'); b.classList.remove('off'); $('#bootNote').textContent = note || 'Запуск камеры…';
  $('#bootSafe').hidden = true;
  clearTimeout(bootT1); clearTimeout(bootT2);
  bootT1 = setTimeout(() => { $('#bootNote').textContent = 'Камера запускается дольше обычного…'; }, 4000);
  bootT2 = setTimeout(() => { if (native && wantNative()) $('#bootSafe').hidden = false; }, 7000);
}
function hideBoot() {
  clearTimeout(bootT1); clearTimeout(bootT2);
  $('#boot').classList.add('off');
}
function showError(msg) {
  diag('ОШИБКА: ' + String(msg).replace(/\n/g, ' '));
  hideBoot();
  $('#errText').textContent = msg;
  $('#errSafe').hidden = !native;
  $('#err').classList.add('on');
}
function safeMode() {
  S.nativeEngine = false; nxFailed = true; persist(); syncSettingsUI();
  $('#err').classList.remove('on');
  startEngine('safe-mode');
}
$('#errBtn').onclick = () => { $('#err').classList.remove('on'); nxFailed = false; startEngine('retry'); };
$('#errSafe').onclick = safeMode;
$('#bootCopy').onclick = (e) => copyDiag(e.currentTarget);
$('#errCopy').onclick = (e) => copyDiag(e.currentTarget);
$('#bootSafe').onclick = safeMode;

/* ================= Сохранение (веб-движок) ================= */
async function saveBlob(blob, name, mime) {
  if (Saver) {
    const { id } = await Saver.begin({ name, mime });
    try {
      const CH = 3 * 256 * 1024;
      for (let o = 0; o < blob.size; o += CH) await Saver.append({ id, data: await b64(blob.slice(o, o + CH)) });
      return await Saver.finish({ id });
    } catch (e) { try { await Saver.abort({ id }); } catch {} throw e; }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.append(a); a.click(); a.remove();
  return {};
}
function queueSave(blob, name, mime, thumb) {
  const url = URL.createObjectURL(blob);
  addSession({ url, type: mime.startsWith('video') ? 'video' : 'image', thumb: thumb || url });
  saveQueue = saveQueue
    .then(() => saveBlob(blob, name, mime))
    .then(() => { if (!burstOn) toast(native ? 'Сохранено в DCIM/DataLaw' : 'Сохранено'); })
    .catch((e) => { console.error(e); toast('Ошибка сохранения: ' + errMsg(e), 3500); });
}

/* ================= Запуск / остановка движка ================= */
let startToken = 0, starting = false;
const stale = (my) => { if (my !== startToken) throw new Error('SUPERSEDED'); };

async function startEngine(reason = '') {
  const my = ++startToken;
  starting = true;
  diag(`v${DL.version} (${DL.code}) native=${native} cameraX=${S.nativeEngine} mode=${S.mode}`);
  diag(`startEngine(${reason}) native=${wantNative()}`);
  showBoot();
  try {
    await stopEngine(wantNative() && NX);
    stale(my);
    $('#err').classList.remove('on');
    $$('[data-mode="portrait"],[data-mode="fun"]').forEach((b) => { b.hidden = !wantNative(); });
    if ((S.mode === 'portrait' || S.mode === 'fun') && !wantNative()) S.mode = 'photo';

    if (wantNative()) {
      try {
        await startNative(my);
        stale(my); hideBoot(); return;
      } catch (e) {
        if (errMsg(e) === 'SUPERSEDED') return;
        const m = errMsg(e);
        diag('CameraX FAIL: ' + m);
        await cleanupNative();
        if (m.includes('PERMISSION')) {
          showError('Нет доступа к камере.\nРазрешите «Камера» для DataLaw Camera в настройках телефона и нажмите «Повторить».');
          return;
        }
        nxFailed = true;
        S.nativeEngine = false; persist(); syncSettingsUI();   // не пытаться снова при каждом запуске
        toast('CameraX недоступен — включён режим совместимости. Причина в Настройки → Диагностика', 4500);
        $$('[data-mode="portrait"],[data-mode="fun"]').forEach((b) => { b.hidden = true; });
        if (S.mode === 'portrait' || S.mode === 'fun') S.mode = 'photo';
        showBoot('Режим совместимости…');
      }
    }
    stale(my);
    await startWeb(my);
    stale(my); hideBoot();
  } catch (e) {
    if (errMsg(e) === 'SUPERSEDED') return;
    diag('START FAIL: ' + errMsg(e));
    showError('Не удалось открыть камеру.\n' + errMsg(e));
  } finally {
    if (my === startToken) starting = false;
  }
}

async function stopEngine(keepNative) {
  stopWebStream();
  if (keepNative) return;   // при смене режима CameraX просто перепривязывается
  if (NX && NC) { try { await withTimeout(NC.stop(), 3000, 'stop'); } catch {} }
  NX = false;
  document.documentElement.classList.remove('nx');
}
async function cleanupNative() {
  try { if (NC) await withTimeout(NC.stop(), 3000, 'stop'); } catch {}
  NX = false;
  document.documentElement.classList.remove('nx');
  delete S.nxPending; persist();
}

async function startNative(my) {
  // 1) плагин отвечает?
  bootNote('Проверка CameraX…');
  const pong = await withTimeout(NC.ping(), 3000, 'CameraX ping');
  diag(`ping ok: ${pong.device} sdk${pong.sdk}`);
  stale(my);
  // 2) разрешения — отдельным шагом, с человеческим тайм-аутом (диалог ждёт пользователя)
  bootNote('Запрос доступа к камере…');
  const perm = await withTimeout(NC.requestPermissions(), 120000, 'Разрешения');
  diag('perm: ' + JSON.stringify(perm));
  stale(my);
  if (perm.camera !== 'granted') throw new Error('PERMISSION_DENIED');
  // 3) запуск. Метка nxPending: если приложение зависнет/упадёт здесь, следующий запуск пойдёт без CameraX.
  S.nxPending = Date.now(); persist();
  document.documentElement.classList.add('nx');
  NX = true;
  applyLayout();
  await nextFrames();
  stale(my);
  bootNote('Открываем камеру…');
  await withTimeout(NC.funEffect({ id: S.fx }), 3000, 'funEffect').catch(() => {});
  const r = await withTimeout(NC.start({ mode: S.mode, facing: S.facing, res: S.res, css: curCss(), ...vfRect() }), 20000, 'CameraX start');
  stale(my);
  C = {
    zoom: r.zoomMax > r.zoomMin + 0.01 ? { min: r.zoomMin, max: r.zoomMax } : null,
    torch: !!r.torch,
    ev: r.hasEv && r.evMax > r.evMin ? { min: r.evMin, max: r.evMax, step: r.evStep || 0.1 } : null,
    pro: r.pro ? { iso: [r.isoMin, r.isoMax], exp: [r.expMin, r.expMax] } : null,
    front: !!r.front,
  };
  delete S.nxPending; persist();
  diag('CameraX started');
  afterStart();
}
function bootNote(t) { const n = $('#bootNote'); if (n) n.textContent = t; }

async function getStream(v, audio) {
  try { return await navigator.mediaDevices.getUserMedia({ video: v, audio }); }
  catch (e) {
    if (audio) { toast('Микрофон недоступен — видео без звука'); return getStream(v, false); }
    throw e;
  }
}
async function startWeb(my) {
  NX = false;
  document.documentElement.classList.remove('nx');
  applyLayout();
  video.classList.add('swap');
  const mode = S.mode === 'video' ? 'video' : 'photo';
  const dims = mode === 'video' ? RES[S.res] : [1920, 1440];
  const base = { width: { ideal: dims[0] }, height: { ideal: dims[1] }, frameRate: { ideal: 30 } };
  const v = S.deviceId ? { ...base, deviceId: { exact: S.deviceId } } : { ...base, facingMode: { ideal: S.facing } };
  bootNote('Открываем камеру…');
  try {
    try { stream = await withTimeout(getStream(v, mode === 'video'), 20000, 'Доступ к камере'); }
    catch (e1) {
      diag('getUserMedia #1: ' + errMsg(e1));
      stream = await withTimeout(getStream({ facingMode: { ideal: S.facing } }, false), 15000, 'Доступ к камере');
    }
  } catch (e) {
    diag('getUserMedia FAIL: ' + errMsg(e));
    throw new Error('Нет доступа к камере. Разрешите «Камера» в настройках телефона.\n(' + errMsg(e) + ')');
  }
  stale(my);
  track = stream.getVideoTracks()[0]; ic = null; webReleased = false;
  video.srcObject = stream;
  await Promise.race([video.play().catch(() => {}), sleep(2500)]);
  wcaps = track.getCapabilities ? track.getCapabilities() : {};
  const st = track.getSettings ? track.getSettings() : {};
  const front = st.facingMode === 'user' || /front/i.test(track.label || '');
  video.classList.toggle('mirror', front);
  C = {
    zoom: wcaps.zoom && wcaps.zoom.max > wcaps.zoom.min ? { min: wcaps.zoom.min, max: wcaps.zoom.max } : null,
    torch: !!wcaps.torch,
    ev: wcaps.exposureCompensation && wcaps.exposureCompensation.max
      ? { min: wcaps.exposureCompensation.min, max: wcaps.exposureCompensation.max, step: wcaps.exposureCompensation.step || 0.1 } : null,
    // exposureTime в Chrome — в единицах по 100 мкс
    pro: wcaps.iso && wcaps.exposureTime && wcaps.iso.max
      ? { iso: [wcaps.iso.min, wcaps.iso.max], exp: [wcaps.exposureTime.min * 1e-4, wcaps.exposureTime.max * 1e-4] } : null,
    front,
  };
  await refreshDevices(st.deviceId);
  diag('WebView camera started');
  afterStart();
  video.classList.remove('swap');
}
function stopWebStream() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null; track = null;
}
function afterStart() {
  isFront = C.front; curZoom = 1;
  if (NX) $('#btnLens').hidden = true;
  buildZoom(); buildEV(); buildPro(); updateTop(); applyTorch();
}

async function refreshDevices(curId) {
  try {
    const all = (await withTimeout(navigator.mediaDevices.enumerateDevices(), 3000, 'devices')).filter((d) => d.kind === 'videoinput');
    backs = all.filter((d) => /back|rear|environment/i.test(d.label));
    if (!backs.length) backs = all.filter((d) => !/front/i.test(d.label));
    const idx = backs.findIndex((d) => d.deviceId === curId);
    $('#btnLens').hidden = !(backs.length > 1 && !C.front);
    $('#lensLbl').textContent = backs.length > 1 ? String(Math.max(idx, 0) + 1) : '';
  } catch { $('#btnLens').hidden = true; }
}
$('#btnFlip').onclick = () => {
  if (rec || starting) return;
  S.deviceId = null; S.facing = isFront ? 'environment' : 'user';
  if (S.facing === 'user' && S.flash === 'torch') S.flash = 'off';
  vib(); startEngine('flip');
};
$('#btnLens').onclick = () => {
  if (rec || NX || starting || backs.length < 2) return;
  const cur = track && track.getSettings().deviceId;
  const i = (backs.findIndex((d) => d.deviceId === cur) + 1) % backs.length;
  S.deviceId = backs[i].deviceId; S.facing = 'environment'; vib(); startEngine('lens');
};

/* ================= Зум ================= */
let zPending = false, zlT;
function buildZoom() {
  const bar = $('#zoombar'); bar.innerHTML = '';
  const z = C.zoom;
  if (!z) { bar.hidden = true; return; }
  let vals = [z.min, 1, 2, 3, 5, 10].filter((v) => v >= z.min - 1e-6 && v <= z.max + 1e-6);
  vals = [...new Set(vals.map((v) => +v.toFixed(1)))].sort((a, b) => a - b).slice(0, 5);
  vals.forEach((v) => {
    const b = document.createElement('button');
    b.textContent = `${v}×`; b.dataset.z = v;
    b.onclick = () => setZoom(v);
    bar.append(b);
  });
  bar.hidden = vals.length < 2;
  markZoom();
}
function markZoom() {
  let best = null, d = 1e9;
  $$('#zoombar button').forEach((b) => { const k = Math.abs(+b.dataset.z - curZoom); if (k < d) { d = k; best = b; } });
  $$('#zoombar button').forEach((b) => b.classList.toggle('on', b === best && d < 0.35));
}
function setZoom(v) {
  const z = C.zoom; if (!z) return;
  curZoom = Math.min(z.max, Math.max(z.min, v));
  const l = $('#zlabel'); l.textContent = `${curZoom.toFixed(1)}×`; l.classList.add('on');
  clearTimeout(zlT); zlT = setTimeout(() => l.classList.remove('on'), 900);
  markZoom();
  if (zPending) return;
  zPending = true;
  requestAnimationFrame(async () => {
    zPending = false;
    try {
      if (NX) await NC.setZoom({ value: curZoom });
      else if (track) await track.applyConstraints({ advanced: [{ zoom: curZoom }] });
    } catch {}
  });
}

/* ================= Экспозиция (EV) ================= */
function buildEV() {
  const box = $('#ev');
  if (!C.ev) { box.hidden = true; return; }
  const r = $('#evRange');
  r.min = C.ev.min; r.max = C.ev.max; r.step = C.ev.step; r.value = 0;
  $('#evLbl').textContent = 'EV 0';
  box.hidden = PRO.on;
}
async function applyEV(v) {
  try {
    if (NX) await NC.ev({ value: v });
    else if (track) await track.applyConstraints({ advanced: [{ exposureCompensation: v }] });
  } catch {}
}
$('#evRange').addEventListener('input', (e) => {
  const v = +e.target.value;
  $('#evLbl').textContent = 'EV ' + (v > 0 ? '+' : '') + v.toFixed(1);
  applyEV(v);
});
$('#evLbl').addEventListener('click', () => { $('#evRange').value = 0; $('#evLbl').textContent = 'EV 0'; applyEV(0); });

/* ================= PRO: ручные ISO и выдержка ================= */
const PRO = { on: false, iso: [], exp: [], isoI: 0, expI: 0 };
const fmtExp = (s) => (s >= 0.95 ? `${s.toFixed(0)}s` : `1/${Math.round(1 / s)}`);
function buildPro() {
  PRO.on = false;
  const p = C.pro;
  $('#btnPro').hidden = !p;
  if (!p) { $('#pro').hidden = true; return; }
  PRO.iso = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800].filter((v) => v >= p.iso[0] && v <= p.iso[1]);
  if (!PRO.iso.length) PRO.iso = [Math.round(p.iso[0])];
  PRO.exp = [1 / 8000, 1 / 4000, 1 / 2000, 1 / 1000, 1 / 500, 1 / 250, 1 / 125, 1 / 60, 1 / 30, 1 / 15, 1 / 8, 1 / 4, 1 / 2, 1]
    .filter((v) => v >= p.exp[0] * 0.99 && v <= Math.min(p.exp[1], 1) * 1.01);
  if (!PRO.exp.length) PRO.exp = [p.exp[0]];
  PRO.isoI = nearest(PRO.iso, 200); PRO.expI = nearest(PRO.exp, 1 / 60);
  $('#isoR').max = PRO.iso.length - 1; $('#expR').max = PRO.exp.length - 1;
  updatePro();
}
function updatePro() {
  $('#isoR').value = PRO.isoI; $('#expR').value = PRO.expI;
  $('#isoV').textContent = PRO.iso.length ? PRO.iso[PRO.isoI] : '—';
  $('#expV').textContent = PRO.exp.length ? fmtExp(PRO.exp[PRO.expI]) : '—';
  $('#proM').classList.toggle('on', PRO.on);
  $('#proM').textContent = PRO.on ? 'MANUAL' : 'AUTO';
  $('#btnPro').classList.toggle('act', PRO.on);
  if (C.ev) $('#ev').hidden = PRO.on;
}
let manPending = false;
async function applyManual() {
  if (!C.pro || manPending) return;
  manPending = true; await sleep(60); manPending = false;
  const iso = PRO.iso[PRO.isoI], exp = PRO.exp[PRO.expI];
  try {
    if (NX) await NC.manual({ on: PRO.on, iso, expNs: Math.round(exp * 1e9) });
    else if (track) await track.applyConstraints({ advanced: [PRO.on
      ? { exposureMode: 'manual', iso, exposureTime: exp * 1e4 } : { exposureMode: 'continuous' }] });
  } catch (e) {
    diag('manual: ' + errMsg(e));
    toast('Ручной режим не поддерживается на этой камере', 2500);
    PRO.on = false; updatePro();
  }
}
$('#btnPro').onclick = () => { $('#pro').hidden = !$('#pro').hidden; vib(); };
$('#proM').onclick = () => { PRO.on = !PRO.on; updatePro(); applyManual(); vib(); };
$('#isoR').addEventListener('input', (e) => { PRO.isoI = +e.target.value; PRO.on = true; updatePro(); applyManual(); });
$('#expR').addEventListener('input', (e) => { PRO.expI = +e.target.value; PRO.on = true; updatePro(); applyManual(); });

/* ================= Жесты: тап-фокус + щипок ================= */
const ptrs = new Map(); let pinch = null, tap = null;
vf.addEventListener('pointerdown', (e) => {
  if (e.target.closest('.ev')) return;
  vf.setPointerCapture(e.pointerId);
  ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (ptrs.size === 1) tap = { x: e.clientX, y: e.clientY, t: Date.now(), ok: true };
  if (ptrs.size === 2) {
    tap = null;
    const [a, b] = [...ptrs.values()];
    pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), z: curZoom };
  }
});
vf.addEventListener('pointermove', (e) => {
  if (!ptrs.has(e.pointerId)) return;
  ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (tap && Math.hypot(e.clientX - tap.x, e.clientY - tap.y) > 12) tap.ok = false;
  if (pinch && ptrs.size >= 2) {
    const [a, b] = [...ptrs.values()];
    setZoom(pinch.z * Math.hypot(a.x - b.x, a.y - b.y) / pinch.d);
  }
});
const endPtr = (e) => {
  if (!ptrs.has(e.pointerId)) return;
  ptrs.delete(e.pointerId);
  if (tap && tap.ok && Date.now() - tap.t < 400 && e.type === 'pointerup') focusAt(e.clientX, e.clientY);
  tap = null; if (ptrs.size < 2) pinch = null;
};
vf.addEventListener('pointerup', endPtr);
vf.addEventListener('pointercancel', endPtr);

async function focusAt(cx, cy) {
  const r = vf.getBoundingClientRect();
  const nx = (cx - r.left) / r.width, ny = (cy - r.top) / r.height;
  const ring = $('#ring');
  ring.style.left = cx - r.left + 'px'; ring.style.top = cy - r.top + 'px';
  ring.classList.remove('on'); void ring.offsetWidth; ring.classList.add('on');
  vib(6);
  if (NX) { NC.focus({ x: nx, y: ny }).catch(() => {}); return; }
  if (!track || !video.videoWidth) return;
  const fr = video.videoWidth / video.videoHeight, cr = r.width / r.height;
  let fx = nx, fy = ny;
  if (fr > cr) fx = 0.5 + (nx - 0.5) * cr / fr; else fy = 0.5 + (ny - 0.5) * fr / cr;
  if (isFront) fx = 1 - fx;
  const modes = wcaps.focusMode || [];
  try {
    await track.applyConstraints({ advanced: [{
      pointsOfInterest: [{ x: fx, y: fy }],
      focusMode: modes.includes('single-shot') ? 'single-shot' : 'continuous',
    }] });
    if (modes.includes('continuous')) {
      setTimeout(() => { try { track && track.applyConstraints({ advanced: [{ focusMode: 'continuous' }] }); } catch {} }, 3000);
    }
  } catch {}
}

/* ================= Вспышка / фонарик ================= */
async function setTorch(on) {
  if (!C.torch) return;
  try {
    if (NX) await NC.torch({ on });
    else if (track) await track.applyConstraints({ advanced: [{ torch: on }] });
  } catch {}
}
function applyTorch() { setTorch(S.flash === 'torch' || (S.flash === 'flash' && !!rec)); }
$('#btnFlash').onclick = () => {
  const order = isFront ? ['off', 'flash'] : ['off', 'flash', 'torch'];
  if (!isFront && !C.torch) { toast('Вспышка на этой камере недоступна'); return; }
  S.flash = order[(order.indexOf(S.flash) + 1) % order.length];
  vib(); updateTop(); applyTorch();
  toast({ off: 'Вспышка выкл.', flash: 'Вспышка при съёмке', torch: 'Фонарик включён' }[S.flash]);
};

/* ================= Верхняя панель / раскладка ================= */
function updateTop() {
  $('#btnFlash').classList.toggle('act', S.flash !== 'off');
  $('#flashLbl').textContent = { off: '', flash: 'ON', torch: '∞' }[S.flash];
  $('#btnTimer').classList.toggle('act', S.timer > 0);
  $('#timerLbl').textContent = S.timer ? S.timer + 'с' : '';
  $('#btnAspect').textContent = ASP_LABEL[S.aspect];
  $('#btnAspect').style.display = isVideoMode() ? 'none' : '';
  $('#fxwrap').hidden = S.mode !== 'fun';
  if (S.mode !== 'fun') $('#facehint').classList.remove('on');
  $('#funVid').textContent = S.funVideo ? '🎥\nВИДЕО' : '📷\nФОТО';
  $('#funVid').classList.toggle('vid', S.funVideo);
  $('#btnFilters').classList.toggle('act', S.filter > 0);
  applyLayout();
}
let rectT;
function applyLayout() {
  const ar = S.mode === 'video' ? 9 / 16 : (S.mode === 'fun' && S.funVideo ? 3 / 4 : ASP[S.aspect]);
  vf.style.setProperty('--ar', ar);
  vf.classList.toggle('full', ar < 0.6);
  $('#grid').classList.toggle('on', S.grid);
  $('#hist').classList.toggle('on', S.hist && !NX);
  if (!S.level) $('#level').style.display = 'none';
  video.style.filter = curCss();
  shutter.classList.toggle('video', isVideoMode());
  $$('#modes button').forEach((b) => b.classList.toggle('on', b.dataset.mode === S.mode));
  if (NX) { // подогнать нативное превью под видоискатель
    clearTimeout(rectT);
    rectT = setTimeout(async () => { await nextFrames(); NC.setRect(vfRect()).catch(() => {}); }, 30);
  }
}
window.addEventListener('resize', () => { if (NX) applyLayout(); });
$('#btnTimer').onclick = () => { S.timer = { 0: 3, 3: 10, 10: 0 }[S.timer]; persist(); vib(); updateTop(); };
$('#btnAspect').onclick = () => {
  S.aspect = ASP_ORDER[(ASP_ORDER.indexOf(S.aspect) + 1) % ASP_ORDER.length]; persist(); vib(); updateTop();
};
$('#btnFilters').onclick = () => $('#filters').classList.toggle('on');
FILTERS.forEach((f, i) => {
  const b = document.createElement('button');
  b.className = 'fchip'; b.textContent = f.n;
  b.onclick = () => {
    S.filter = i; persist();
    $$('.fchip').forEach((x, k) => x.classList.toggle('on', k === i));
    updateTop();
    if (NX) NC.filter({ css: curCss() }).catch(() => {});
  };
  $('#filters').append(b);
});
$$('.fchip')[S.filter].classList.add('on');

$$('#modes button').forEach((b) => b.onclick = async () => {
  if (rec || busy || starting) return;
  const m = b.dataset.mode; if (m === S.mode) return;
  const restart = NX || (m === 'video') !== (S.mode === 'video');
  S.mode = m; persist(); vib(); updateTop();
  if (m === 'portrait') toast('Портрет: размывается фон за человеком', 2500);
  if (m === 'fun') toast('Выбирайте эффект внизу и снимайте фото или видео', 2500);
  if (restart) await startEngine('mode');
});

/* ================= Фото ================= */
function drawWatermark(x, w, h) {
  if (!S.watermark || !wmImg.complete || !wmImg.naturalWidth) return;
  const m0 = Math.min(w, h), tw = m0 * 0.30, th = tw * wmImg.naturalHeight / wmImg.naturalWidth, mg = m0 * 0.035;
  x.save(); x.filter = 'none'; x.globalAlpha = 0.92;
  x.drawImage(wmImg, w - tw - mg, h - th - mg, tw, th);
  x.restore();
}
async function grabFrame(fast) {   // веб-движок
  let src = null;
  if (!fast && S.maxphoto && 'ImageCapture' in window && track) {
    try {
      ic = ic || new ImageCapture(track);
      let opt;
      try {
        const pc = await withTimeout(ic.getPhotoCapabilities(), 3000, 'photoCaps');
        if (pc.imageWidth && pc.imageHeight) {
          opt = { imageWidth: Math.min(pc.imageWidth.max, 4608), imageHeight: Math.min(pc.imageHeight.max, 3456) };
        }
      } catch {}
      const blob = await withTimeout(ic.takePhoto(opt), 12000, 'takePhoto');
      src = await createImageBitmap(blob);
      if ((src.width > src.height) !== (video.videoWidth > video.videoHeight)) { src.close(); src = null; }
    } catch (e) { diag('takePhoto → кадр из видео: ' + errMsg(e)); src = null; }
  }
  const from = src || video;
  const w0 = src ? src.width : video.videoWidth, h0 = src ? src.height : video.videoHeight;
  if (!w0) throw new Error('Нет кадра');
  const r = ASP[S.aspect];
  let sw = w0, sh = h0;
  if (sw / sh > r) sw = sh * r; else sh = sw / r;
  const sx = (w0 - sw) / 2, sy = (h0 - sh) / 2;
  const k = Math.min(1, (fast ? 2160 : 4608) / Math.max(sw, sh));
  cv.width = Math.round(sw * k); cv.height = Math.round(sh * k);
  const x = cv.getContext('2d');
  x.filter = curCss() || 'none';
  x.drawImage(from, sx, sy, sw, sh, 0, 0, cv.width, cv.height);
  if (src) src.close();
  drawWatermark(x, cv.width, cv.height);
  const blob = await new Promise((res) => cv.toBlob(res, 'image/jpeg', fast ? 0.9 : 0.95));
  const t = document.createElement('canvas'); t.width = 120; t.height = Math.round(120 * cv.height / cv.width);
  t.getContext('2d').drawImage(cv, 0, 0, t.width, t.height);
  return { blob, thumb: t.toDataURL('image/jpeg', 0.7) };
}

function shutterFx(quiet) {
  const b = $('#blink'); b.classList.remove('on'); void b.offsetWidth; b.classList.add('on');
  if (!quiet) { clickSound(); vib(18); }
}

async function takeAndSave(fast, name) {
  const useFlash = S.flash === 'flash' && !fast;
  if (useFlash) {
    if (isFront) { $('#sflash').style.display = 'block'; await sleep(350); }
    else if (!NX) { await setTorch(true); await sleep(450); }
  }
  try {
    if (NX) {
      const portrait = S.mode === 'portrait';
      shutterFx(fast);
      if (portrait) toast('Портрет: обработка…', 5000);
      const r = await withTimeout(NC.photo({
        aspect: ASP[S.aspect], css: S.mode === 'fun' ? '' : curCss(), flash: useFlash && !isFront, fast, portrait,
        blur: S.blur, name: (name || stamp()) + '.jpg', effect: S.fx, watermark: S.watermark,
      }), portrait ? 40000 : 20000, 'Съёмка');
      addSession({ type: 'image', thumb: r.thumb, uri: r.uri });
      if (!fast) toast('Сохранено в DCIM/DataLaw');
    } else {
      const res = await grabFrame(fast);
      shutterFx(fast);
      queueSave(res.blob, (name || stamp()) + '.jpg', 'image/jpeg', res.thumb);
    }
  } finally {
    if (useFlash) {
      if (isFront) $('#sflash').style.display = 'none';
      else if (!NX) await setTorch(S.flash === 'torch');
    }
  }
}

async function countdown() {
  const n = S.timer; if (!n) return true;
  counting = true; cancelCount = false;
  const cd = $('#cd');
  for (let i = n; i > 0 && !cancelCount; i--) {
    cd.textContent = i; cd.classList.remove('on'); void cd.offsetWidth; cd.classList.add('on');
    beep(i === 1 ? 1320 : 880); await sleep(1000);
  }
  cd.classList.remove('on'); counting = false;
  return !cancelCount;
}

async function burstSeries(max = 30) {
  if (burstOn) return;
  burstOn = true; let n = 0; const base = stamp(); const bc = $('#burstCount');
  while (burstOn && n < max) {
    n++; bc.textContent = n;
    try { await takeAndSave(true, `${base}_${String(n).padStart(2, '0')}`); } catch (e) { diag('burst: ' + errMsg(e)); break; }
    await sleep(30);
  }
  burstOn = false; bc.textContent = '';
  toast(`Серия: ${n} кадр.`);
}

/* ================= Видео ================= */
function pickMime() {
  const list = ['video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'];
  return list.find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
}
async function startRecording() {
  if (NX) {
    try { await withTimeout(NC.videoStart({ name: stamp() }), 10000, 'Запись'); }
    catch (e) { toast('Не удалось начать запись: ' + errMsg(e), 3000); return; }
    rec = { native: true };
    if (S.filter && S.mode !== 'fun') toast('В режиме CameraX фильтры не применяются к видео', 2800);
  } else {
    if (!stream) return;
    let rs = stream;
    const css = curCss();
    if (css) {
      const c = document.createElement('canvas');
      const sc = Math.min(1, 1920 / Math.max(video.videoWidth, video.videoHeight));
      c.width = Math.round(video.videoWidth * sc); c.height = Math.round(video.videoHeight * sc);
      const x = c.getContext('2d'); x.filter = css;
      const loop = () => { x.drawImage(video, 0, 0, c.width, c.height); recRaf = requestAnimationFrame(loop); };
      loop();
      rs = c.captureStream(30);
      stream.getAudioTracks().forEach((t) => rs.addTrack(t));
    }
    const mime = pickMime();
    let mr;
    try { mr = new MediaRecorder(rs, { mimeType: mime || undefined, videoBitsPerSecond: BITRATE[S.res] || 8e6 }); }
    catch (e) { toast('Запись не поддерживается: ' + e.message, 3000); cancelAnimationFrame(recRaf); return; }
    recChunks = [];
    mr.ondataavailable = (e) => e.data && e.data.size && recChunks.push(e.data);
    mr.onstop = () => {
      cancelAnimationFrame(recRaf);
      const type = (mr.mimeType || mime || 'video/webm').split(';')[0];
      const blob = new Blob(recChunks, { type });
      rec = null; recChunks = [];
      queueSave(blob, stamp() + (type.includes('mp4') ? '.mp4' : '.webm'), type, recThumb);
    };
    try {
      const t = document.createElement('canvas'); t.width = 120; t.height = Math.round(120 * video.videoHeight / video.videoWidth);
      t.getContext('2d').drawImage(video, 0, 0, t.width, t.height); recThumb = t.toDataURL('image/jpeg', 0.7);
    } catch { recThumb = ''; }
    mr.start(1000);
    rec = mr;
  }
  recStart = Date.now();
  $('#recbar').classList.add('on'); shutter.classList.add('rec');
  $('#recTime').textContent = '00:00';
  recTimer = setInterval(() => { $('#recTime').textContent = mmss(Math.floor((Date.now() - recStart) / 1000)); }, 500);
  applyTorch(); beep(1200); vib(30);
}
async function stopRecording() {
  if (!rec) return;
  const r = rec;
  clearInterval(recTimer);
  $('#recbar').classList.remove('on'); shutter.classList.remove('rec');
  setTorch(S.flash === 'torch'); beep(700); vib(30);
  if (r.native) {
    rec = null;
    try {
      const x = await withTimeout(NC.videoStop(), 20000, 'Остановка записи');
      addSession({ type: 'video', thumb: x.thumb, uri: x.uri });
      toast('Видео сохранено в DCIM/DataLaw');
    } catch (e) { diag('videoStop: ' + errMsg(e)); toast('Ошибка записи видео: ' + errMsg(e), 3000); }
  } else {
    try { r.stop(); } catch {}
  }
}

/* ================= Затвор ================= */
async function onShutter() {
  if (starting) return;
  if (counting) { cancelCount = true; return; }
  if (isVideoMode()) {
    if (rec) return stopRecording();
    if (busy) return;
    busy = true;
    if (await countdown()) await startRecording();
    busy = false; return;
  }
  if (busy || burstOn) return;
  busy = true;
  try { if (await countdown()) await takeAndSave(false); }
  catch (e) { diag('shutter: ' + errMsg(e)); toast('Ошибка съёмки: ' + errMsg(e), 3000); }
  busy = false;
}
shutter.addEventListener('pointerdown', () => {
  if (S.mode === 'burst' && !busy && !counting && !starting) burstSeries();
});
['pointerup', 'pointercancel', 'pointerleave'].forEach((ev) => shutter.addEventListener(ev, () => { burstOn = false; }));
shutter.addEventListener('click', () => { if (S.mode !== 'burst') onShutter(); });
window.addEventListener('volumekey', () => {
  if (!S.volume) return;
  if (S.mode === 'burst') burstSeries(6); else onShutter();
});

/* ================= Гистограмма и уровень ================= */
const hc = $('#hist'), hx = hc.getContext('2d');
const tmp = document.createElement('canvas'); tmp.width = tmp.height = 48;
const tx = tmp.getContext('2d', { willReadFrequently: true });
setInterval(() => {
  if (!S.hist || NX || !video.videoWidth) return;
  try {
    tx.drawImage(video, 0, 0, 48, 48);
    const d = tx.getImageData(0, 0, 48, 48).data, b = new Array(32).fill(0);
    for (let i = 0; i < d.length; i += 4) b[(0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]) >> 3]++;
    const m = Math.max(...b);
    hx.clearRect(0, 0, 96, 40); hx.fillStyle = 'rgba(255,255,255,.85)';
    b.forEach((v, i) => { const h = Math.sqrt(v / m) * 36; hx.fillRect(i * 3, 40 - h - 2, 2, h); });
  } catch {}
}, 250);

const lvl = $('#level');
window.addEventListener('devicemotion', (e) => {
  if (!S.level) return;
  const a = e.accelerationIncludingGravity;
  if (!a || a.x == null) return;
  if (Math.abs(a.y) < 6) { lvl.style.display = 'none'; return; }
  const deg = Math.atan2(a.x, a.y) * 180 / Math.PI;
  lvl.style.display = 'block';
  lvl.style.transform = `translate(-50%,-50%) rotate(${deg}deg)`;
  lvl.classList.toggle('ok', Math.abs(deg) < 1.2);
});

/* ================= Настройки ================= */
function syncSettingsUI() {
  $$('.seg').forEach((seg) => $$('button', seg).forEach((b) => b.classList.toggle('on', String(S[seg.dataset.key]) === b.dataset.val)));
  $$('.tgl').forEach((i) => { i.checked = !!S[i.dataset.key]; });
}
$$('.nonly').forEach((el) => { el.hidden = !native; });
$$('.seg').forEach((seg) => seg.addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  const key = seg.dataset.key; S[key] = isNaN(b.dataset.val) ? b.dataset.val : +b.dataset.val;
  persist(); syncSettingsUI();
  if (key === 'res' && S.mode === 'video') startEngine('res');
}));
$$('.tgl').forEach((i) => i.addEventListener('change', () => {
  const key = i.dataset.key;
  S[key] = i.checked; persist(); applyLayout();
  if (key === 'volume' && Saver) Saver.volumeKeys({ enabled: S.volume }).catch(() => {});
  if (key === 'nativeEngine') { nxFailed = false; startEngine('engine-toggle'); }
}));
$('#btnSettings').onclick = () => { syncSettingsUI(); openOv('sheet'); };
$('#sheetClose').onclick = closeOv;
$('#sheet').addEventListener('click', (e) => { if (e.target.id === 'sheet') closeOv(); });
$('#aboutVer').textContent = `версия ${DL.version} (${DL.code})${DL.build && DL.build !== 'dev' ? '' : ' · dev'}`;
$('#aboutPrivacy').href = DL.privacyUrl || '#';
$('#diagCopy').onclick = (e) => copyDiag(e.currentTarget);

/* ================= Миниатюра / просмотр ================= */
let vi = 0;
const viewList = () => session.filter((s) => s.url);
function renderViewer() {
  const list = viewList(), it = list[vi], body = $('#vbody'); body.innerHTML = '';
  if (!it) return;
  let el;
  if (it.type === 'video') { el = document.createElement('video'); el.controls = true; el.autoplay = true; el.playsInline = true; }
  else el = document.createElement('img');
  el.src = it.url; body.append(el);
  $('#vCount').textContent = `${vi + 1} / ${list.length}`;
}
$('#btnThumb').onclick = () => {
  const last = session[session.length - 1];
  if (!last) { toast(native ? 'Снимки лежат в DCIM/DataLaw' : 'Пока пусто'); return; }
  if (last.uri) { // нативный элемент: открыть в системной галерее/плеере
    NC.open({ uri: last.uri, mime: last.type === 'video' ? 'video/mp4' : 'image/jpeg' })
      .catch(() => toast('Не удалось открыть'));
    return;
  }
  vi = viewList().length - 1; renderViewer(); openOv('viewer');
};
$('#vPrev').onclick = () => { const n = viewList().length; if (n) { vi = (vi - 1 + n) % n; renderViewer(); } };
$('#vNext').onclick = () => { const n = viewList().length; if (n) { vi = (vi + 1) % n; renderViewer(); } };
$('#vClose').onclick = closeOv;

/* ================= Смешные эффекты ================= */
function updateFaceHint(n) {
  const show = S.mode === 'fun' && fxInfo().face && n === 0;
  $('#facehint').classList.toggle('on', !!show);
}
FX.forEach((f) => {
  const b = document.createElement('button');
  b.className = 'fxc' + (f.id === S.fx ? ' on' : ''); b.dataset.id = f.id;
  b.innerHTML = `<i>${f.e}</i>${f.n}`;
  b.onclick = () => {
    S.fx = f.id; persist(); vib(8);
    $$('.fxc').forEach((x) => x.classList.toggle('on', x.dataset.id === f.id));
    if (NX) NC.funEffect({ id: f.id }).catch(() => {});
    updateFaceHint(1);
  };
  $('#fx').append(b);
});
$('#funVid').onclick = () => {
  if (rec || busy) return;
  S.funVideo = !S.funVideo; persist(); vib(); updateTop();
};
if (NC) { try { NC.addListener('faces', (e) => updateFaceHint(e.n)).catch(() => {}); } catch {} }

/* ================= Жизненный цикл ================= */
document.addEventListener('visibilitychange', () => {
  diag(document.hidden ? 'приложение свёрнуто' : 'приложение развёрнуто');
  if (document.hidden) {
    burstOn = false;
    if (rec) stopRecording();
    if (!NX && !starting && stream) { stopWebStream(); webReleased = true; }   // отпускаем камеру в фоне
  } else if (!NX && !starting && webReleased) {
    startEngine('resume');
  }
});

/* ================= Старт ================= */
if (Saver) Saver.volumeKeys({ enabled: S.volume }).catch(() => {});
updateTop(); syncSettingsUI(); renderDiag();
if (nativeCrashed) {
  diag('Прошлый запуск CameraX не завершился — включён режим совместимости');
  setTimeout(() => toast('CameraX был отключён после сбоя. Включить: Настройки', 5000), 1500);
}
if (native && !S.hintShown && !nativeCrashed) {
  S.hintShown = true; persist();
  setTimeout(() => toast('Больше возможностей: Настройки → «Расширенный режим CameraX»', 5000), 4000);
}
startEngine('boot');
})();
