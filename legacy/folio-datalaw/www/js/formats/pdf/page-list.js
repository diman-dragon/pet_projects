// Page list of a pdf.js document: in-flow page boxes (ribbon or one page per screen), virtualized by the viewport controller.
import { createLayout } from './layout.js';
import { createViews } from './page-view.js';
import { createViewport } from './viewport.js';
import { clearQueue } from './render-queue.js';
export async function createPageList({ doc, container, scroller, mode = 'scroll', onPage }) {
  const first = await doc.getPage(1);
  const v = first.getViewport({ scale: 1 });
  const entries = Array.from({ length: doc.numPages }, (_, i) => {
    const el = document.createElement('div');
    el.className = 'pdf-page';
    el.dataset.i = i;
    return { i, el, ratio: v.height / v.width, page: i === 0 ? first : null, mounted: false, base: null, baseW: 0, baseJob: 0, detail: null };
  });
  const lay = createLayout({ scroller, container, entries, getMode: () => mode });
  const views = createViews({ doc, lay });
  const vp = createViewport({ scroller, lay, entries, views, getMode: () => mode, onPage });
  const setClass = () => scroller.classList.toggle('paged', mode === 'paged');
  entries.forEach((e) => container.append(e.el));
  setClass();
  lay.all();
  return {
    entries, lay, views, vp,
    setScale: (s) => lay.setScale(s),
    relayout() { lay.all(); vp.sync(); },
    setMode(m) { mode = m; setClass(); lay.setScale(1); },
    goTo(n, smooth = false) {
      const el = entries[Math.min(Math.max(n, 1), entries.length) - 1].el;
      el.scrollIntoView({ behavior: smooth ? 'smooth' : 'instant', block: mode === 'paged' ? 'nearest' : 'start', inline: 'center' });
    },
    destroy() {
      vp.destroy();
      clearQueue();
      [...views.mounted].forEach(views.hide);
      scroller.classList.remove('paged', 'zoomed');
      container.style.paddingInline = '';
    },
  };
}
