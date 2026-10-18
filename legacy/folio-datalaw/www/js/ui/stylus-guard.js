// Stylus mode: the pen's touches are cancelled for scrolling (pointer events still reach the drawing tools).
import { on } from '../core/bus.js';
import { stylusOnly } from './stylus.js';
const scroller = document.getElementById('viewer-scroll');
let lastPen = -1e9;
const mark = (e) => { if (e.pointerType === 'pen') lastPen = performance.now(); };
// touchType says it directly; the pen pointer event fired right before the touch event is the fallback for WebViews without it.
const isPen = (e) => [...e.changedTouches].some((t) => t.touchType === 'stylus') || performance.now() - lastPen < 250;
const guard = (e) => { if (isPen(e) && e.cancelable) e.preventDefault(); };
function attach(on_) {
  const fn = on_ ? 'addEventListener' : 'removeEventListener';
  scroller[fn]('pointerdown', mark, { capture: true, passive: true });
  scroller[fn]('pointermove', mark, { capture: true, passive: true });
  scroller[fn]('touchstart', guard, { passive: false });
  scroller[fn]('touchmove', guard, { passive: false });
}
on('stylus', attach);
if (stylusOnly()) attach(true);
