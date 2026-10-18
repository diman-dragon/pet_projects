// Selection overlay inside a page (percent units, so it follows zoom): translucent line boxes plus two drag handles.
let layer = null, boxes = null, h0 = null, h1 = null;
const handle = (side) => Object.assign(document.createElement('div'), { className: `sel-handle ${side}` });
function build(pageEl) {
  layer = Object.assign(document.createElement('div'), { className: 'sel-layer' });
  boxes = document.createElement('div');
  h0 = handle('start');
  h1 = handle('end');
  h0.dataset.side = 'start';
  h1.dataset.side = 'end';
  layer.append(boxes, h0, h1);
  pageEl.append(layer);
}
const pct = (v, t) => `${(v / t) * 100}%`;
export function paintSel(pageEl, T, lines, { handles = true } = {}) {
  if (!layer || layer.parentNode !== pageEl) { removeSel(); build(pageEl); }
  boxes.replaceChildren(...lines.map((l) => {
    const d = document.createElement('div');
    d.className = 'sel-box';
    d.style.cssText = `left:${pct(l.x, T.w)};top:${pct(l.y, T.h)};width:${pct(l.w, T.w)};height:${pct(l.h, T.h)}`;
    return d;
  }));
  const f = lines[0], l = lines.at(-1);
  h0.style.cssText = `left:${pct(f.x, T.w)};top:${pct(f.y + f.h, T.h)}`;
  h1.style.cssText = `left:${pct(l.x + l.w, T.w)};top:${pct(l.y + l.h, T.h)}`;
  h0.hidden = h1.hidden = !handles;
}
export function removeSel() { layer?.remove(); layer = boxes = h0 = h1 = null; }
export const handleSide = (el) => el.closest?.('.sel-handle')?.dataset.side ?? null;
