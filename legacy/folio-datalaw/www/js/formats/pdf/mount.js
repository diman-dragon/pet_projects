// PDF module entry: composes document, page list, zoom and gestures into a viewer handle.
import { openPdf } from './open-doc.js';
import { createPageList } from './page-list.js';
import { createZoom } from './zoom.js';
import { attachPinch } from './pinch.js';
import { attachDoubleTap } from './double-tap.js';
import { getPageText } from './page-chars.js';
import { renderThumb } from './thumb.js';
import { renderForOcr } from './ocr-source.js';
import { ocrText } from '../../core/ocr-geom.js';
const TEXT_MIN = 20; // fewer chars than this (page numbers, stamps) does not make a page "text"
export async function mountPdf({ container, scroller, blob, startPage = 1, mode = 'scroll', onPage }) {
  const doc = await openPdf(blob);
  let cur = startPage, curMode = mode;
  const ocr = new Map(); // page index -> recognized words (see core/ocr-geom.js)
  const list = await createPageList({ doc, container, scroller, mode, onPage: (n) => { cur = n; onPage(n, doc.numPages); } });
  const zoom = createZoom({
    scroller, content: container, lay: list.lay, entries: list.entries, getMode: () => curMode,
    onChange: (s) => { scroller.classList.toggle('zoomed', s !== 1); list.setScale(s); },
    onSettle: () => list.vp.sync(),
  });
  let from = cur;
  const swipe = ({ dx, dy }) => {
    if (curMode !== 'paged' || zoom.get() > 1.01) return;
    const flip = Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5;
    list.goTo(from + (flip ? (dx < 0 ? 1 : -1) : 0), true);
  };
  const offPinch = attachPinch({ scroller, zoom, onSwipe: swipe, onStart: () => { from = cur; } });
  const offTap = attachDoubleTap({ scroller, zoom });
  list.goTo(startPage);
  list.vp.sync();
  let lastW = scroller.clientWidth, lastH = scroller.clientHeight, timer;
  const ro = new ResizeObserver(() => {
    const w = scroller.clientWidth, h = scroller.clientHeight;
    const changed = Math.abs(w - lastW) > 2 || (curMode === 'paged' && Math.abs(h - lastH) > 2);
    lastW = w; lastH = h;
    if (changed) { clearTimeout(timer); timer = setTimeout(list.relayout, 100); }
  });
  ro.observe(scroller);
  return {
    doc,
    zoomBy: zoom.by,
    goTo: list.goTo,
    // Native text first; a page without text uses its recognized layer.
    getText: async (i) => {
      const T = await doc.getPage(i + 1).then(getPageText);
      return T.chars.length >= TEXT_MIN || !ocr.has(i) ? T : ocrText(ocr.get(i));
    },
    hasText: async (i) => ocr.has(i) || (await doc.getPage(i + 1).then(getPageText)).chars.length >= TEXT_MIN,
    setOcr: (i, data) => ocr.set(i, data),
    renderForOcr: (i) => renderForOcr(doc, i + 1),
    thumb: (n, canvas, w, near) => renderThumb(list, doc, n, canvas, w, near),
    setMode(m) {
      const keep = cur;
      curMode = m;
      zoom.reset();
      scroller.classList.remove('zoomed');
      list.setMode(m);
      list.goTo(keep);
      list.vp.sync();
    },
    destroy() { clearTimeout(timer); ro.disconnect(); offPinch(); offTap(); list.destroy(); doc.loadingTask.destroy(); },
  };
}
