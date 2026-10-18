// SVG nodes for annotation items (page units: x 0..1, y 0..R). Vector, so they stay crisp at any zoom and cost no bitmap memory.
const NS = 'http://www.w3.org/2000/svg';
export const HL = 0.4;
const nodes = new WeakMap();
export const nodeOf = (item) => nodes.get(item);
export const register = (item, node) => nodes.set(item, node);
export const layerOf = (item) => (item.k === 'hl' || item.k === 'th' ? 'mark' : 'ink');
const f = (v) => +v.toFixed(4);
export function pathD(pts, R) {
  const [x0, y0] = pts[0];
  let d = `M${x0} ${f(y0 * R)}`;
  if (pts.length === 1) return `${d}L${f(x0 + 0.0001)} ${f(y0 * R)}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [x, y] = pts[i], [nx, ny] = pts[i + 1];
    d += `Q${x} ${f(y * R)} ${f((x + nx) / 2)} ${f(((y + ny) / 2) * R)}`;
  }
  const l = pts.at(-1);
  return `${d}L${l[0]} ${f(l[1] * R)}`;
}
// Pressure-sensitive stroke as a filled outline: pts are [x, y, pressure 0..1]; width follows the pressure (0.25x .. 1.75x).
export function outlineD(pts, R, w) {
  const P = [];
  pts.forEach(([x, y, p = 0.5]) => {
    const q = P.at(-1), r = (w / 2) * (0.25 + 1.5 * p);
    if (!q || Math.hypot(x - q[0], y * R - q[1]) > 0.0006) P.push([x, y * R, r]); else q[2] = Math.max(q[2], r);
  });
  const n = P.length;
  if (n === 1) { const [x, y, r] = P[0]; return `M${f(x - r)} ${f(y)}a${f(r)} ${f(r)} 0 1 0 ${f(2 * r)} 0a${f(r)} ${f(r)} 0 1 0 ${f(-2 * r)} 0z`; }
  const A = [], B = [];
  P.forEach(([x, y, r], i) => {
    const a = P[Math.max(0, i - 1)], b = P[Math.min(n - 1, i + 1)];
    let tx = b[0] - a[0], ty = b[1] - a[1];
    const l = Math.hypot(tx, ty) || 1;
    tx /= l; ty /= l;
    A.push([x - ty * r, y + tx * r]);
    B.push([x + ty * r, y - tx * r]);
  });
  const run = (L) => L.slice(1).map(([x, y]) => `L${f(x)} ${f(y)}`).join('');
  const [r0, r1] = [P[0][2], P[n - 1][2]];
  return `M${f(A[0][0])} ${f(A[0][1])}${run(A)}A${f(r1)} ${f(r1)} 0 0 0 ${f(B[n - 1][0])} ${f(B[n - 1][1])}${run([...B].reverse())}A${f(r0)} ${f(r0)} 0 0 0 ${f(A[0][0])} ${f(A[0][1])}z`;
}
export function svgNode(tag, attrs) {
  const n = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
  return n;
}
export function buildNode(item, R) {
  let n;
  if (item.v) {
    n = svgNode('path', { d: outlineD(item.p, R, item.w), fill: item.c, stroke: 'none' });
  } else if (item.k === 'pen' || item.k === 'hl') {
    n = svgNode('path', { d: pathD(item.p, R), fill: 'none', stroke: item.c, 'stroke-width': item.w, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', ...(item.k === 'hl' ? { 'stroke-opacity': HL } : {}) });
  } else if (item.k === 'th') {
    n = svgNode('path', { d: item.r.map(([x, y, w, h]) => `M${f(x)} ${f(y * R)}h${f(w)}v${f(h * R)}h${f(-w)}z`).join(''), fill: item.c, 'fill-opacity': HL });
  } else {
    const lw = Math.max(0.003, item.r[0][3] * R * 0.08);
    n = svgNode('path', { d: item.r.map(([x, y, w, h]) => `M${f(x)} ${f((y + h) * R - lw * 0.55)}h${f(w)}`).join(''), fill: 'none', stroke: item.c, 'stroke-width': f(lw), 'stroke-linecap': 'round' });
  }
  nodes.set(item, n);
  return n;
}
