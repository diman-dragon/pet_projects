// Double tap on the scroller toggles zoom (fit <-> 2.5x) around the tapped point.
export function attachDoubleTap({ scroller, zoom }) {
  let last = 0, lx = 0, ly = 0, sx = 0, sy = 0;
  const start = (e) => {
    if (e.touches.length !== 1) { last = 0; return; }
    sx = e.touches[0].clientX; sy = e.touches[0].clientY;
  };
  const end = (e) => {
    if (e.touches.length || e.changedTouches.length !== 1 || e.changedTouches[0].touchType === 'stylus' || scroller.closest('.draw-active')) return;
    const { clientX: x, clientY: y } = e.changedTouches[0];
    if (Math.hypot(x - sx, y - sy) > 10) { last = 0; return; }
    const now = Date.now();
    if (now - last < 300 && Math.hypot(x - lx, y - ly) < 40) { last = 0; zoom.toggleAt(x, y); } else { last = now; lx = x; ly = y; }
  };
  scroller.addEventListener('touchstart', start, { passive: true });
  scroller.addEventListener('touchend', end, { passive: true });
  return () => { scroller.removeEventListener('touchstart', start); scroller.removeEventListener('touchend', end); };
}
