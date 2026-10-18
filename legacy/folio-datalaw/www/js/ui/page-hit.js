// Page text helpers for gestures: fetch a page's char table and map screen points to page units.
import { getActiveHandle } from './viewer-handle.js';
import { hitChar } from './sel-hit.js';
export const pageText = (el) => getActiveHandle()?.getText?.(Number(el.dataset.i)).catch(() => null) ?? Promise.resolve(null);
// Screen point -> page units; k = page units per css px.
export function local(el, T, x, y) {
  const r = el.getBoundingClientRect(), k = T.w / r.width;
  return { x: (x - r.left) * k, y: (y - r.top) * k, k };
}
export const hit = (el, T, x, y) => { const p = local(el, T, x, y), h = hitChar(T.chars, p.x, p.y); return { i: h.i, css: h.dist / p.k }; };
