// Stylus mode: one finger scrolls the page (with a fling) and swipes between pages in "Pages" mode; the pen never does.
import { stylusOnly } from './stylus.js';
import { getActiveHandle } from './viewer-handle.js';
import { getReadMode } from '../core/readmode.js';
import { on } from '../core/bus.js';
const scroller = document.getElementById('viewer-scroll');
let g = null, fling = 0, page = 1, penDown = false, raf = 0, dx = 0, dy = 0;
on('pagechange', (p) => { page = p.n; });
const stop = () => { cancelAnimationFrame(fling); fling = 0; };
function glide(vx, vy) {
  if (Math.hypot(vx, vy) < 0.05) return;
  scroller.scrollLeft -= vx * 16;
  scroller.scrollTop -= vy * 16;
  fling = requestAnimationFrame(() => glide(vx * 0.94, vy * 0.94));
}
const apply = () => { raf = 0; scroller.scrollLeft -= dx; scroller.scrollTop -= dy; dx = dy = 0; };
scroller.addEventListener('pointerdown', (e) => {
  if (e.pointerType === 'pen') penDown = true; // a palm next to a drawing pen never scrolls
  if (e.pointerType !== 'touch' || penDown || !stylusOnly()) return;
  stop();
  if (g) { g = null; return; }
  g = { pid: e.pointerId, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, t: performance.now(), vx: 0, vy: 0, from: page };
  try { scroller.setPointerCapture(e.pointerId); } catch { /* ignore */ }
});
scroller.addEventListener('pointermove', (e) => {
  if (!g || e.pointerId !== g.pid) return;
  const now = performance.now(), ddt = Math.max(1, now - g.t), mx = e.clientX - g.x, my = e.clientY - g.y;
  g.vx = 0.7 * g.vx + 0.3 * (mx / ddt); g.vy = 0.7 * g.vy + 0.3 * (my / ddt);
  dx += mx; dy += my;
  g.x = e.clientX; g.y = e.clientY; g.t = now;
  raf ||= requestAnimationFrame(apply);
}, { passive: true });
function end(e) {
  if (e.pointerType === 'pen') penDown = false;
  if (!g || e.pointerId !== g.pid) return;
  const d = g;
  g = null;
  cancelAnimationFrame(raf);
  apply();
  const mx = d.x - d.x0, my = d.y - d.y0;
  if (getReadMode() === 'paged' && !scroller.classList.contains('zoomed') && getActiveHandle()?.caps?.mode !== false) {
    const flip = Math.abs(mx) > 50 && Math.abs(mx) > Math.abs(my) * 1.5;
    getActiveHandle()?.goTo(d.from + (flip ? (mx < 0 ? 1 : -1) : 0), true);
  } else glide(d.vx, d.vy);
}
scroller.addEventListener('pointerup', end);
scroller.addEventListener('pointercancel', end);
