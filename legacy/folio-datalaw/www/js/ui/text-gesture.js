// Touch gestures on page text: long press selects a word, handles extend it, text tools mark by dragging.
import { lineBoxes } from './sel-chars.js';
import { wordAt } from './sel-hit.js';
import { pageText, hit } from './page-hit.js';
import { handleSide, paintSel, removeSel } from './sel-view.js';
import { getSel, select, clearSel, placeMenu, applyMark } from './text-select.js';
import { getDrawTool } from './viewer-draw.js';
import { hideMenu } from './text-menu.js';
import { toast } from './toast.js';
import { t } from '../core/i18n.js';
import { on } from '../core/bus.js';
const pages = document.getElementById('viewer-pages');
const scroller = document.getElementById('viewer-scroll');
const LONG = 420, SLOP = 10, REACH = 30;
let press = null, drag = null;
function cancelPress() { clearTimeout(press?.timer); clearTimeout(press?.warm); press = null; }
async function fire(p) {
  const T = await (p.job ??= pageText(p.el));
  if (press !== p || !T) return;
  if (T.chars.length < 20) { toast(t('ocr.noText')); return; } // scanned page: it needs recognition first
  const h = hit(p.el, T, p.x, p.y);
  if (h.i < 0 || h.css > REACH) return;
  p.fired = true;
  const [a, b] = wordAt(T.chars, h.i);
  select(p.el, T, a, b);
  navigator.vibrate?.(12);
}
function startPress(e) {
  const el = e.target.closest?.('.pdf-page');
  if (!el?.dataset.ready || getDrawTool()) return;
  cancelPress();
  press = { el, x: e.clientX, y: e.clientY, pid: e.pointerId, fired: false, job: null };
  press.warm = setTimeout((p) => { p.job = pageText(p.el); }, 160, press); // text geometry is built only if the finger stays down
  press.timer = setTimeout(fire, LONG, press);
}
// A drag is either a selection handle (extends the selection) or a text tool (marks the range on release).
function startDrag(kind, e, init) {
  drag = { kind, pid: e.pointerId, pt: [e.clientX, e.clientY], raf: 0, anchor: -1, T: null, range: null, ...init };
  try { (kind === 'handle' ? e.target : pages).setPointerCapture(e.pointerId); } catch { /* ignore */ }
  e.preventDefault();
  return drag;
}
function step(d) {
  d.raf = 0;
  if (!d.T || d.anchor < 0) return;
  const h = hit(d.el, d.T, d.pt[0], d.pt[1] - (d.kind === 'handle' ? 22 : 0));
  if (h.i < 0) return;
  let a = Math.min(d.anchor, h.i), b = Math.max(d.anchor, h.i);
  if (d.kind === 'mark' && a === b) [a, b] = wordAt(d.T.chars, a);
  d.range = [a, b];
  if (d.kind === 'handle') select(d.el, d.T, a, b, { menu: false });
  else paintSel(d.el, d.T, lineBoxes(d.T.chars, a, b), { handles: false });
}
function startMark(e, tool) {
  const el = e.target.closest?.('.pdf-page');
  if (!el?.dataset.ready) return;
  clearSel();
  const d = startDrag('mark', e, { el, tool });
  const pt = [e.clientX, e.clientY];
  d.ready = pageText(el).then((T) => {
    d.T = T;
    if (!T) return;
    const h = hit(el, T, ...pt);
    d.anchor = h.css <= REACH * 1.5 ? h.i : -1;
    if (!d.ended) step(d);
  });
}
async function finishMark(d) {
  await d.ready;
  step(d);
  removeSel();
  if (d.range) applyMark(d.tool === 'underline' ? 'tu' : 'th', d.el, d.T, ...d.range);
}
pages.addEventListener('pointerdown', (e) => {
  if (drag) return;
  const side = handleSide(e.target), s = getSel(), tool = getDrawTool();
  if (side && s) { hideMenu(); startDrag('handle', e, { el: s.el, T: s.T, anchor: side === 'start' ? s.b : s.a }); }
  else if (tool === 'text-select' || tool === 'underline') startMark(e, tool);
  else if (!tool) startPress(e);
});
pages.addEventListener('pointermove', (e) => {
  if (press && !press.fired && Math.hypot(e.clientX - press.x, e.clientY - press.y) > SLOP) cancelPress();
  if (drag && e.pointerId === drag.pid) { drag.pt = [e.clientX, e.clientY]; drag.raf ||= requestAnimationFrame(() => step(drag)); }
});
function end(e) {
  if (press && e.pointerId === press.pid) {
    const tap = !press.fired && e.type === 'pointerup';
    cancelPress();
    if (tap && getSel()) clearSel();
  }
  if (!drag || e.pointerId !== drag.pid) return;
  const d = drag;
  drag = null;
  d.ended = true;
  cancelAnimationFrame(d.raf);
  if (d.kind === 'mark') { finishMark(d); return; }
  step(d);
  placeMenu();
}
pages.addEventListener('pointerup', end);
pages.addEventListener('pointercancel', end);
pages.addEventListener('contextmenu', (e) => { if (!e.target.closest('.fb2')) e.preventDefault(); }); // FB2 text uses the native selection
pages.addEventListener('pagehide', (e) => { if (getSel()?.el === e.target) clearSel(); });
scroller.addEventListener('scroll', () => { cancelPress(); if (getSel() && !drag) requestAnimationFrame(placeMenu); }, { passive: true });
on('toolchange', () => { cancelPress(); drag = null; clearSel(); });
