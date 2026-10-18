// Active tool / colour state and the freehand tools (pen with pressure, marker, eraser); strokes are stored as vectors.
import { ensureLayers } from './annot-layers.js';
import { addItem, recordErase } from './annot-store.js';
import { svgNode, register, outlineD, HL } from './annot-svg.js';
import { eraseSegment } from './annot-erase.js';
import { liveCanvas, segment } from './pen-live.js';
import { stylusOnly, penSeen } from './stylus.js';
import { emit } from '../core/bus.js';
const pages = document.getElementById('viewer-pages');
const root = document.getElementById('viewer');
const WIDTH = { pen: 2.8, highlighter: 20, eraser: 14 };
const KIND = { pen: 'pen', highlighter: 'hl', eraser: 'er' };
let tool = null, color = '#2563eb', g = null, noticed = false;
export const getDrawTool = () => tool;
export const getDrawColor = () => color;
export const setDrawColor = (c) => { color = c; };
export function setDrawTool(next) {
  finish();
  tool = next;
  root.classList.toggle('draw-active', Boolean(next));
  root.dataset.tool = next ?? '';
  emit('toolchange', next);
}
const norm = (v) => Math.round(Math.min(1, Math.max(0, v)) * 10000) / 10000;
function point(e) { // pressure is smoothed a little so the line width does not flicker
  g.p = g.p === undefined ? e.pressure : g.p * 0.6 + e.pressure * 0.4;
  return [norm((e.clientX - g.r.left) / g.r.width), norm((e.clientY - g.r.top) / g.r.height), Math.round(g.p * 100) / 100];
}
function down(e) {
  const pen = e.pointerType === 'pen';
  if (g && e.pointerId !== g.pid) {
    if (g.pen) return; // palm or finger while the pen is drawing: ignored
    if (pen) drop(); else { abort(); return; } // the pen always wins over a touch stroke
  }
  const el = e.target.closest?.('.pdf-page');
  const kind = pen && e.buttons & 34 ? 'eraser' : tool; // eraser end / barrel button of the pen
  if (!KIND[kind] || !(e.isPrimary || pen) || !el?.dataset.ready || (!pen && stylusOnly())) return;
  const { mark, ink, R } = ensureLayers(el);
  const r = el.getBoundingClientRect();
  g = { pid: e.pointerId, pen, v: pen && kind === 'pen', el, r, R, k: kind, w: WIDTH[kind] / r.width, pts: [], n: 1, raf: 0, removed: [], i: Number(el.dataset.i) };
  g.pts.push(point(e));
  if (g.v) { g.live = liveCanvas(el, color); segment(g.live, g.pts[0], g.pts[0], g.w); }
  else if (kind !== 'eraser') {
    const hl = kind === 'highlighter';
    g.node = svgNode('path', { fill: 'none', stroke: color, 'stroke-width': g.w, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', ...(hl ? { 'stroke-opacity': HL } : {}) });
    (hl ? mark : ink).append(g.node);
    g.d = `M${g.pts[0][0]} ${g.pts[0][1] * R}`;
    g.node.setAttribute('d', `${g.d}L${g.pts[0][0] + 0.0001} ${g.pts[0][1] * R}`);
  }
  try { pages.setPointerCapture(e.pointerId); } catch { /* ignore */ }
}
function flush() {
  if (!g) return;
  g.raf = 0;
  for (; g.n < g.pts.length; g.n++) {
    const c = g.pts[g.n - 1], p = g.pts[g.n];
    if (g.v) segment(g.live, c, p, g.w);
    else if (g.k === 'eraser') g.removed.push(...eraseSegment(g.i, c, p, g.R, g.w));
    else g.d += `Q${c[0]} ${c[1] * g.R} ${(c[0] + p[0]) / 2} ${((c[1] + p[1]) / 2) * g.R}`;
  }
  if (g.node) g.node.setAttribute('d', `${g.d}L${g.pts.at(-1)[0]} ${g.pts.at(-1)[1] * g.R}`);
}
function move(e) {
  if (!g || e.pointerId !== g.pid) return;
  const list = e.getCoalescedEvents?.() ?? [];
  (list.length ? list : [e]).forEach((ev) => {
    const last = g.pts.at(-1), p = point(ev);
    // dense pen samples (240 Hz) add nothing below a fraction of a pixel
    if (Math.hypot((p[0] - last[0]) * g.r.width, (p[1] - last[1]) * g.r.height) >= 0.7 || Math.abs(p[2] - last[2]) > 0.06) g.pts.push(p);
  });
  g.raf ||= requestAnimationFrame(flush);
}
function finish() {
  if (!g) return;
  cancelAnimationFrame(g.raf);
  if (g.k === 'eraser' && g.pts.length === 1) g.pts.push(g.pts[0]);
  flush();
  if (g.k === 'eraser') recordErase(g.removed);
  else {
    const item = { k: KIND[g.k], c: color, w: Math.round(g.w * 1e5) / 1e5, p: g.pts, ...(g.v ? { v: 1 } : {}) };
    if (g.v) {
      g.live.c.remove();
      g.node = svgNode('path', { d: outlineD(g.pts, g.R, item.w), fill: color, stroke: 'none' });
      ensureLayers(g.el).ink.append(g.node);
    }
    register(item, g.node);
    addItem(g.i, item);
  }
  g = null;
}
function drop() {
  if (!g) return;
  cancelAnimationFrame(g.raf);
  if (g.k === 'eraser') recordErase(g.removed);
  g.node?.remove();
  g.live?.c.remove();
  g = null;
}
function abort() { // a second finger means pinch / pan: a barely started stroke is dropped, a real one kept
  if (!g) return;
  const far = g.pts.some((p) => Math.hypot((p[0] - g.pts[0][0]) * g.r.width, (p[1] - g.pts[0][1]) * g.r.height) > 14);
  if (far || g.k === 'eraser') finish(); else drop();
}
const up = (e) => { if (g && e.pointerId === g.pid) finish(); };
pages.addEventListener('pointerdown', (e) => { if (e.pointerType === 'pen' && !noticed) { noticed = true; penSeen(); } down(e); });
pages.addEventListener('pointermove', move, { passive: true });
pages.addEventListener('pointerup', up);
pages.addEventListener('pointercancel', up);
