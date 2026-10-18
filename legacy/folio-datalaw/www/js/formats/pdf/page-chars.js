// Text geometry of a page straight from pdf.js text content (no DOM): every char box in page units (scale 1, y down).
const cache = new WeakMap();
const widths = new Map();
const ctx2 = document.createElement('canvas').getContext('2d');
function charW(family, ch) {
  const k = `${family}|${ch}`;
  let w = widths.get(k);
  if (w === undefined) { ctx2.font = `100px ${family}`; w = ctx2.measureText(ch).width; widths.set(k, w); }
  return w;
}
const mul = (m, t) => [m[0] * t[0] + m[2] * t[1], m[1] * t[0] + m[3] * t[1], m[0] * t[2] + m[2] * t[3], m[1] * t[2] + m[3] * t[3], m[0] * t[4] + m[2] * t[5] + m[4], m[1] * t[4] + m[3] * t[5] + m[5]];
async function build(page) {
  const vp = page.getViewport({ scale: 1 });
  const { items, styles } = await page.getTextContent();
  const chars = [];
  for (const it of items) {
    if (!it.str) continue;
    const [a, b, c, d, e, f] = mul(vp.transform, it.transform);
    const fh = Math.hypot(c, d), len = Math.hypot(a, b);
    if (!fh || !len) continue;
    const st = styles[it.fontName] ?? {};
    const asc = st.ascent ?? 0.88, desc = st.descent ?? -0.22;
    const ws = [...it.str].map((ch) => charW(st.fontFamily || 'sans-serif', ch));
    const sum = ws.reduce((x, y) => x + y, 0);
    const total = it.width;
    const ux = a / len, uy = b / len, vx = c / fh, vy = d / fh;
    let s = 0;
    [...it.str].forEach((ch, k) => {
      const w = sum ? (ws[k] / sum) * total : total / it.str.length;
      const xs = [], ys = [];
      for (const [along, up] of [[s, asc], [s + w, asc], [s, desc], [s + w, desc]]) { xs.push(e + ux * along + vx * up * fh); ys.push(f + uy * along + vy * up * fh); }
      const x = Math.min(...xs), y = Math.min(...ys);
      chars.push({ ch, x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y });
      s += w;
    });
  }
  return { w: vp.width, h: vp.height, chars };
}
export function getPageText(page) {
  if (!cache.has(page)) cache.set(page, build(page));
  return cache.get(page);
}
