// Drag logic of the floating tool palette: moves it with the finger, keeps it inside the viewer, remembers the position.
import { getPref, setPref } from '../core/prefs.js';
const root = document.getElementById('viewer');
export function initPanelDrag(panel, isLocked) {
  let pos = getPref('panelpos', { fx: 1, fy: 0.12 });
  const area = () => { const r = root.getBoundingClientRect(), top = root.querySelector('.bar').offsetHeight + 6; return { w: r.width - panel.offsetWidth - 6, h: r.height - top - panel.offsetHeight - 10, top }; };
  const place = () => {
    const a = area();
    panel.style.transform = `translate(${3 + pos.fx * Math.max(0, a.w)}px,${a.top + pos.fy * Math.max(0, a.h)}px)`;
  };
  let g = null;
  panel.addEventListener('pointerdown', (e) => {
    if (isLocked() || e.target.closest('button')) return;
    const a = area(), r = panel.getBoundingClientRect(), v = root.getBoundingClientRect();
    g = { pid: e.pointerId, dx: e.clientX - r.left, dy: e.clientY - r.top, a, v };
    panel.setPointerCapture(e.pointerId);
    panel.classList.add('dragging');
  });
  panel.addEventListener('pointermove', (e) => {
    if (!g || e.pointerId !== g.pid) return;
    const x = e.clientX - g.dx - g.v.left - 3, y = e.clientY - g.dy - g.v.top - g.a.top;
    pos = { fx: Math.min(1, Math.max(0, x / (g.a.w || 1))), fy: Math.min(1, Math.max(0, y / (g.a.h || 1))) };
    place();
  });
  const end = (e) => { if (g && e.pointerId === g.pid) { g = null; panel.classList.remove('dragging'); setPref('panelpos', pos); } };
  panel.addEventListener('pointerup', end);
  panel.addEventListener('pointercancel', end);
  window.addEventListener('resize', place);
  return place;
}
