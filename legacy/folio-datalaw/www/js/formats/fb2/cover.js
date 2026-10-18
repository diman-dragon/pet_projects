// Library cover of a FB2 book: the embedded cover picture, or a generated title card when the book has none.
import { loadBook } from './load.js';
const W = 320, H = 452;
const hue = (s) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7);
function wrap(ctx, text, max) {
  const lines = [];
  let line = '';
  for (const w of text.split(/\s+/)) {
    const t = line ? `${line} ${w}` : w;
    if (ctx.measureText(t).width > max && line) { lines.push(line); line = w; } else line = t;
  }
  if (line) lines.push(line);
  return lines.slice(0, 7);
}
function card(book) {
  const c = Object.assign(document.createElement('canvas'), { width: W, height: H });
  const ctx = c.getContext('2d'), h = hue(book.title + book.authors.join());
  const g = ctx.createLinearGradient(0, 0, W, H);
  g.addColorStop(0, `hsl(${h} 45% 34%)`);
  g.addColorStop(1, `hsl(${(h + 40) % 360} 50% 18%)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(255,255,255,.35)';
  ctx.strokeRect(16, 16, W - 32, H - 32);
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'center';
  ctx.font = '600 20px Georgia, serif';
  const author = wrap(ctx, book.authors.join(', '), W - 70);
  author.forEach((l, i) => ctx.fillText(l, W / 2, 70 + i * 26));
  ctx.font = '700 30px Georgia, serif';
  const lines = wrap(ctx, book.title || 'FB2', W - 70);
  lines.forEach((l, i) => ctx.fillText(l, W / 2, 190 + i * 38));
  return c;
}
async function picture(book) {
  const b = book.binaries.get(book.coverId);
  if (!b) return null;
  const bytes = Uint8Array.from(atob(b.b64), (ch) => ch.charCodeAt(0));
  const bmp = await createImageBitmap(new Blob([bytes], { type: b.type }));
  const c = Object.assign(document.createElement('canvas'), { width: W, height: Math.round((W * bmp.height) / bmp.width) });
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close?.();
  return c;
}
export async function fb2Cover(blob) {
  const book = await loadBook(blob);
  const canvas = (await picture(book).catch(() => null)) ?? card(book);
  const out = await new Promise((res) => canvas.toBlob(res, 'image/jpeg', 0.85));
  return { blob: out, pages: book.pages };
}
