// Two-finger gesture on the scroller (works with any tool): pinch zoom + pan follow the fingers; a plain swipe is reported.
import { hold } from './render-queue.js';
export function attachPinch({ scroller, zoom, onSwipe, onStart = () => {} }) {
  let g = null, d0 = 0, raf = 0, last = null;
  const dist = (t) => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY) || 1;
  const mid = (t) => [(t[0].clientX + t[1].clientX) / 2, (t[0].clientY + t[1].clientY) / 2];
  const start = (e) => {
    if (e.touches.length !== 2 || g || [...e.touches].some((t) => t.touchType === 'stylus')) return; // pen + palm is not a pinch
    d0 = dist(e.touches);
    g = zoom.begin(...mid(e.touches));
    scroller.addEventListener('touchmove', move, { passive: false }); // only while two fingers are down
    hold(true); // no rendering while the fingers move
    onStart();
  };
  const move = (e) => {
    if (!g || e.touches.length !== 2) return;
    e.preventDefault();
    last = [dist(e.touches) / d0, ...mid(e.touches)];
    raf ||= requestAnimationFrame(() => { raf = 0; if (g && last) zoom.live(g, ...last); });
  };
  const end = (e) => {
    if (!g || e.touches.length >= 2) return;
    const done = g;
    g = null;
    scroller.removeEventListener('touchmove', move);
    cancelAnimationFrame(raf);
    raf = 0;
    if (last) zoom.live(done, ...last);
    last = null;
    zoom.end(done);
    hold(false);
    if (Math.abs(done.k - 1) < 0.06) onSwipe({ dx: done.x - done.x0, dy: done.y - done.y0, dt: Date.now() - done.t0 });
  };
  scroller.addEventListener('touchstart', start, { passive: true });
  scroller.addEventListener('touchend', end, { passive: true });
  scroller.addEventListener('touchcancel', end, { passive: true });
  return () => {
    scroller.removeEventListener('touchstart', start);
    scroller.removeEventListener('touchmove', move);
    scroller.removeEventListener('touchend', end);
    scroller.removeEventListener('touchcancel', end);
  };
}
