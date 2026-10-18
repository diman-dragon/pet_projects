// Keeps the screen on while a document is open: held while the reader touches the screen, released after the idle limit.
import { plugin } from './platform.js';
import { awakeMinutes } from './awake-pref.js';
import { on } from './bus.js';
const EVENTS = ['pointerdown', 'touchstart', 'scroll', 'keydown', 'wheel'];
let running = false, held = false, timer = 0, lock = null;
async function hold(v) {
  if (held === v) return;
  held = v;
  const native = plugin('Screen');
  try {
    if (native) await native.keepOn({ on: v });
    else if (v) lock = await navigator.wakeLock?.request('screen');
    else { await lock?.release(); lock = null; }
  } catch { /* screen stays under system control */ }
}
function bump() {
  if (!running) return;
  clearTimeout(timer);
  const mins = awakeMinutes();
  if (!mins) { hold(false); return; }
  hold(true);
  if (Number.isFinite(mins)) timer = setTimeout(() => hold(false), mins * 60000);
}
const onVisible = () => { if (document.visibilityState === 'visible') { held = false; bump(); } };
export function startAwake() {
  if (running) return;
  running = true;
  EVENTS.forEach((n) => document.addEventListener(n, bump, { capture: true, passive: true }));
  document.addEventListener('visibilitychange', onVisible);
  bump();
}
export function stopAwake() {
  if (!running) return;
  running = false;
  clearTimeout(timer);
  EVENTS.forEach((n) => document.removeEventListener(n, bump, { capture: true }));
  document.removeEventListener('visibilitychange', onVisible);
  hold(false);
}
on('awakechange', bump);
