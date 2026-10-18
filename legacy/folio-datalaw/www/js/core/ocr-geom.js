// Turns OCR words (page units) into the char table the selection engine uses: each word box is shared by letter width.
const NARROW = /[iIlj1!|.,:;'`]/, WIDE = /[mwMW@%]/, CAPS = /[A-ZА-ЯЂЉЊЋЏЖШЩЫЮČĆŠŽĐ]/;
const weight = (ch) => (NARROW.test(ch) ? 0.45 : WIDE.test(ch) ? 1.5 : CAPS.test(ch) ? 1.15 : 1);
export function wordsToChars(words) {
  const out = [];
  let prev = null;
  for (const [text, x, y, w, h] of words) {
    if (prev && Math.abs(prev.y + prev.h / 2 - (y + h / 2)) < Math.max(prev.h, h) * 0.6 && x > prev.x + prev.w) {
      out.push({ ch: ' ', x: prev.x + prev.w, y: prev.y, w: x - (prev.x + prev.w), h: prev.h });
    }
    const cs = [...text], ws = cs.map(weight), sum = ws.reduce((a, b) => a + b, 0);
    let at = x;
    cs.forEach((ch, i) => { const cw = (ws[i] / sum) * w; out.push({ ch, x: at, y, w: cw, h }); at += cw; });
    prev = { x, y, w, h };
  }
  return out;
}
export const ocrText = (data) => ({ w: data.w, h: data.h, chars: wordsToChars(data.words) });
