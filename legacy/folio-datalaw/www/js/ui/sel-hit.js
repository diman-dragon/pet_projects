// Hit testing on the char table: nearest char to a point, and word expansion around a char.
const isWord = (ch) => /[\p{L}\p{N}_]/u.test(ch);
const near = (p, q) => Math.abs(p.y + p.h / 2 - (q.y + q.h / 2)) < p.h * 0.6 && q.x - (p.x + p.w) < p.h * 0.4;
export function hitChar(chars, x, y) {
  let best = -1, bd = Infinity;
  for (let i = 0; i < chars.length; i++) {
    const c = chars[i];
    const dx = Math.max(c.x - x, 0, x - (c.x + c.w));
    const dy = Math.max(c.y - y, 0, y - (c.y + c.h));
    const d = dx * dx + 4 * dy * dy;
    if (d < bd) { bd = d; best = i; if (!d) break; }
  }
  return { i: best, dist: Math.sqrt(bd) };
}
export function wordAt(chars, i) {
  let k = i;
  if (!isWord(chars[k].ch)) {
    if (chars[k + 1] && isWord(chars[k + 1].ch)) k++;
    else if (chars[k - 1] && isWord(chars[k - 1].ch)) k--;
    else return [i, i];
  }
  let a = k, b = k;
  while (a > 0 && isWord(chars[a - 1].ch) && near(chars[a - 1], chars[a])) a--;
  while (b < chars.length - 1 && isWord(chars[b + 1].ch) && near(chars[b], chars[b + 1])) b++;
  return [a, b];
}
