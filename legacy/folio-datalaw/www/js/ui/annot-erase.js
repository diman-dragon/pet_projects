// Stroke eraser: removes every annotation touched by the eraser segment (units: x 0..1, y 0..R) and reports what it removed.
import { getItems, takeItems } from './annot-store.js';
import { nodeOf } from './annot-svg.js';
function segDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  const t = l2 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0;
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}
function touches(item, a, b, R, rad) {
  if (item.p) return item.p.some(([x, y]) => segDist(x, y * R, a[0], a[1] * R, b[0], b[1] * R) <= rad + item.w / 2);
  return item.r.some(([x, y, w, h]) => [0, 0.25, 0.5, 0.75, 1].some((t) => {
    const px = a[0] + (b[0] - a[0]) * t, py = (a[1] + (b[1] - a[1]) * t) * R;
    return px >= x - rad && px <= x + w + rad && py >= y * R - rad && py <= (y + h) * R + rad;
  }));
}
export function eraseSegment(i, a, b, R, rad) {
  const hit = getItems(i).filter((it) => touches(it, a, b, R, rad));
  hit.forEach((it) => nodeOf(it)?.remove());
  return takeItems(i, hit);
}
