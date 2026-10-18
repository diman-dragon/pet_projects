// Per-page vector annotation layers (marks blended under ink): created on demand, rebuilt from the store.
import { getItems } from './annot-store.js';
import { svgNode, buildNode, layerOf } from './annot-svg.js';
export function ensureLayers(el) {
  const R = el.offsetHeight / (el.offsetWidth || 1);
  let mark = el.querySelector(':scope > .annot-mark'), ink = el.querySelector(':scope > .annot-ink');
  if (!ink) {
    const mk = (cls) => svgNode('svg', { class: `annot ${cls}`, preserveAspectRatio: 'none' });
    mark = mk('annot-mark');
    ink = mk('annot-ink');
    el.append(mark, ink);
  }
  [mark, ink].forEach((s) => s.setAttribute('viewBox', `0 0 1 ${R}`));
  return { mark, ink, R };
}
export function repaintPage(el) {
  const { mark, ink, R } = ensureLayers(el);
  mark.replaceChildren();
  ink.replaceChildren();
  getItems(Number(el.dataset.i)).forEach((it) => (layerOf(it) === 'mark' ? mark : ink).append(buildNode(it, R)));
}
export const addNode = (el, item) => { const { mark, ink, R } = ensureLayers(el); (layerOf(item) === 'mark' ? mark : ink).append(buildNode(item, R)); };
// Pages come back when they scroll into view: bring their annotations with them.
document.getElementById('viewer-pages').addEventListener('pageready', (e) => {
  if (getItems(e.detail.i).length) repaintPage(e.target);
});
