// Helpers on a page's char table (page units): line boxes, normalized rects and plain text of a char range.
export const sameLine = (p, q) => Math.abs(p.y + p.h / 2 - (q.y + q.h / 2)) < Math.max(p.h, q.h) * 0.6;
export function lineBoxes(chars, a, b) {
  const lines = [];
  for (let i = a; i <= b; i++) {
    const c = chars[i], L = lines.at(-1);
    if (L && sameLine(L.ref, c)) {
      L.x0 = Math.min(L.x0, c.x); L.x1 = Math.max(L.x1, c.x + c.w);
      L.y0 = Math.min(L.y0, c.y); L.y1 = Math.max(L.y1, c.y + c.h);
    } else lines.push({ ref: c, x0: c.x, x1: c.x + c.w, y0: c.y, y1: c.y + c.h });
  }
  return lines.map((L) => ({ x: L.x0, y: L.y0, w: L.x1 - L.x0, h: L.y1 - L.y0 }));
}
// Line boxes -> 0..1 page coordinates, slightly shrunk so highlights of neighbour lines do not touch.
export const normRects = (lines, T) => lines.map((l) => [l.x / T.w, (l.y + l.h * 0.06) / T.h, l.w / T.w, (l.h * 0.88) / T.h]);
export function textOf(chars, a, b) {
  let s = '';
  for (let i = a; i <= b; i++) {
    const c = chars[i], p = chars[i - 1];
    if (p && i > a) {
      if (!sameLine(p, c)) s += '\n';
      else if (c.x - (p.x + p.w) > p.h * 0.3 && c.ch !== ' ' && p.ch !== ' ') s += ' ';
    }
    s += c.ch;
  }
  return s.trim();
}
