// Annotations per page (0..1 page coordinates, so they scale with zoom) + one undo/redo history across pages.
// Items: { k: 'pen'|'hl', c, w, p:[[x,y]..] } strokes; { k: 'th'|'tu', c, r:[[x,y,w,h]..] } text highlight / underline.
import { emit } from '../core/bus.js';
const pages = new Map();
let done = [], undone = [];
const changed = () => emit('annotchange', { undo: done.length > 0, redo: undone.length > 0 });
const list = (i) => { let l = pages.get(i); if (!l) pages.set(i, (l = [])); return l; };
export const getItems = (i) => pages.get(i) ?? [];
export function addItem(i, item) { list(i).push(item); done.push({ t: 'add', i, item }); undone = []; changed(); }
// Removes items right away (live erasing) and returns what was removed; the history entry is written by recordErase().
export function takeItems(i, items) {
  const l = list(i), out = [];
  items.forEach((item) => { const idx = l.indexOf(item); if (idx >= 0) { l.splice(idx, 1); out.push({ i, item, idx }); } });
  return out;
}
export function recordErase(removed) { if (removed.length) { done.push({ t: 'del', removed }); undone = []; changed(); } }
function step(op, undoing) {
  const pgs = new Set();
  if (op.t === 'add') {
    if (undoing) takeItems(op.i, [op.item]); else list(op.i).push(op.item);
    pgs.add(op.i);
  } else if (undoing) [...op.removed].reverse().forEach((r) => { list(r.i).splice(r.idx, 0, r.item); pgs.add(r.i); });
  else op.removed.forEach((r) => { takeItems(r.i, [r.item]); pgs.add(r.i); });
  return [...pgs];
}
export function undoOp() { const op = done.pop(); if (!op) return []; const p = step(op, true); undone.push(op); changed(); return p; }
export function redoOp() { const op = undone.pop(); if (!op) return []; const p = step(op, false); done.push(op); changed(); return p; }
export function clearAnnots() { pages.clear(); done = []; undone = []; changed(); }
