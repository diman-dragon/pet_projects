// One page's canvases: a base canvas (whole page, size-capped) plus a detail canvas (visible window) while zoomed in.
import { renderPage, baseCap, detailCap } from './render-page.js';
import { enqueue, cancel } from './render-queue.js';
const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
const canvas = (cls) => Object.assign(document.createElement('canvas'), { className: cls });
const fire = (e, name) => e.el.dispatchEvent(new CustomEvent(name, { bubbles: true, detail: { i: e.i } }));
export function createViews({ doc, lay }) {
  const mounted = new Set();
  const pageOf = async (e) => (e.page ??= await doc.getPage(e.i + 1));
  function drawBase(e, prio) {
    const w = lay.widthOf(e);
    if (e.baseW === w || e.baseJob === w) return;
    e.baseJob = w;
    enqueue(`b${e.i}`, prio, async (ctl) => {
      try {
        const page = await pageOf(e);
        if (ctl.dead) return;
        const v = page.getViewport({ scale: 1 });
        if (Math.abs(e.ratio - v.height / v.width) > 0.001) { e.ratio = v.height / v.width; lay.place(e); lay.measure(); }
        const c = canvas('pv pv-base');
        ctl.task = renderPage(page, c, lay.widthOf(e), { maxPx: baseCap() });
        await ctl.task.promise;
        if (ctl.dead || !e.mounted) return;
        e.base?.remove();
        e.el.prepend(c);
        e.base = c;
        e.baseW = w;
        if (!e.el.dataset.ready) { e.el.dataset.ready = '1'; fire(e, 'pageready'); }
      } finally { if (e.baseJob === w) e.baseJob = 0; }
    });
  }
  function clearDetail(e) { cancel(`d${e.i}`); e.detail?.canvas.remove(); e.detail = null; }
  // Draws the visible window of a zoomed page at full resolution (the base canvas is size-capped).
  function detail(e, view, prio) {
    const w = lay.widthOf(e), d = dpr();
    if (w * w * e.ratio * d * d <= baseCap()) { if (e.detail) clearDetail(e); return; }
    const r = e.el.getBoundingClientRect();
    const box = (m) => [Math.max(0, view.left - m * view.width - r.left), Math.max(0, view.top - m * view.height - r.top), Math.min(r.width, view.right + m * view.width - r.left), Math.min(r.height, view.bottom + m * view.height - r.top)];
    const [vx0, vy0, vx1, vy1] = box(0), [x0, y0, x1, y1] = box(0.3);
    if (vx1 <= vx0 || vy1 <= vy0) return;
    const c = e.detail;
    if (c && c.w === w && c.x <= vx0 + 1 && c.y <= vy0 + 1 && c.x + c.cw >= vx1 - 1 && c.y + c.ch >= vy1 - 1) return;
    enqueue(`d${e.i}`, 100 + prio, async (ctl) => {
      const page = await pageOf(e);
      if (ctl.dead) return;
      const cv = canvas('pv pv-detail');
      ctl.task = renderPage(page, cv, w, { maxPx: detailCap(), rect: { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } });
      await ctl.task.promise;
      if (ctl.dead || !e.mounted || !e.base) return;
      cv.style.cssText = `left:${(x0 / w) * 100}%;top:${(y0 / (w * e.ratio)) * 100}%;width:${((x1 - x0) / w) * 100}%;height:${((y1 - y0) / (w * e.ratio)) * 100}%`;
      e.detail?.canvas.remove();
      e.base.after(cv);
      e.detail = { canvas: cv, w, x: x0, y: y0, cw: x1 - x0, ch: y1 - y0 };
    });
  }
  function show(e, prio) {
    if (!e.mounted) { e.mounted = true; mounted.add(e); }
    drawBase(e, prio);
  }
  function hide(e) {
    if (!e.mounted) return;
    e.mounted = false;
    mounted.delete(e);
    cancel(`b${e.i}`); cancel(`d${e.i}`);
    fire(e, 'pagehide');
    e.el.replaceChildren();
    delete e.el.dataset.ready;
    e.base = e.detail = null;
    e.baseW = e.baseJob = 0;
    e.page?.cleanup();
  }
  return { show, hide, detail, mounted };
}
