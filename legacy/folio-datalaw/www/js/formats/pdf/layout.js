// Page sizes and positions: pages are in-flow boxes; sizes are written in one pass, then positions are measured (one reflow).
const MARGIN = 16;
export function createLayout({ scroller, container, entries, getMode }) {
  let cw = 0, ch = 0, scale = 1;
  const n = entries.length;
  const starts = new Float64Array(n), sizes = new Float64Array(n);
  const paged = () => getMode() === 'paged';
  const widthOf = (e) => Math.max(120, (paged() ? Math.min(cw, ch / e.ratio) : cw) * scale);
  const place = (e) => { const w = widthOf(e); e.el.style.width = `${w}px`; e.el.style.height = `${w * e.ratio}px`; };
  function measure() {
    const h = paged(), o = h ? container.offsetLeft : container.offsetTop;
    entries.forEach((e, i) => { starts[i] = o + (h ? e.el.offsetLeft : e.el.offsetTop); sizes[i] = h ? e.el.offsetWidth : e.el.offsetHeight; });
  }
  function all() {
    cw = scroller.clientWidth - MARGIN;
    ch = scroller.clientHeight - MARGIN;
    const pad = !paged() ? 0 : scale <= 1 ? Math.max(8, (scroller.clientWidth - widthOf(entries[0])) / 2) : 8;
    entries.forEach(place);
    container.style.paddingInline = paged() ? `${pad}px` : '';
    measure();
  }
  function indexAt(pos) {
    let lo = 0, hi = n - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (starts[m] <= pos) lo = m; else hi = m - 1; }
    return lo;
  }
  function range(pos, len, margin) {
    const a = indexAt(pos - margin);
    let b = a;
    while (b + 1 < n && starts[b + 1] < pos + len + margin) b++;
    return [a, b];
  }
  return { widthOf, place, all, measure, indexAt, range, start: (i) => starts[i], size: (i) => sizes[i], setScale(s) { scale = s; all(); } };
}
