// Minimal PDF writer: one JPEG picture per page (DCTDecode), no dependencies. pages: [{ jpeg: Uint8Array, w, h }].
const enc = new TextEncoder();
const PAGE_W = 595.28;
export function buildPdf(pages) {
  const parts = [], offsets = [];
  let len = 0;
  const push = (x) => { const b = typeof x === 'string' ? enc.encode(x) : x; parts.push(b); len += b.length; };
  const begin = (n) => { offsets[n] = len; push(`${n} 0 obj\n`); };
  push(new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d, 0x31, 0x2e, 0x34, 0x0a, 0x25, 0xe2, 0xe3, 0xcf, 0xd3, 0x0a])); // %PDF-1.4 + binary marker
  begin(1);
  push('<< /Type /Catalog /Pages 2 0 R >>\nendobj\n');
  begin(2);
  push(`<< /Type /Pages /Count ${pages.length} /Kids [${pages.map((_, k) => `${3 + 3 * k} 0 R`).join(' ')}] >>\nendobj\n`);
  pages.forEach((p, k) => {
    const H = +(PAGE_W * (p.h / p.w)).toFixed(2), W = PAGE_W, id = 3 + 3 * k;
    begin(id);
    push(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${W} ${H}] /Resources << /XObject << /Im0 ${id + 2} 0 R >> >> /Contents ${id + 1} 0 R >>\nendobj\n`);
    const content = `q ${W} 0 0 ${H} 0 0 cm /Im0 Do Q\n`;
    begin(id + 1);
    push(`<< /Length ${content.length} >>\nstream\n${content}endstream\nendobj\n`);
    begin(id + 2);
    push(`<< /Type /XObject /Subtype /Image /Width ${p.w} /Height ${p.h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${p.jpeg.length} >>\nstream\n`);
    push(p.jpeg);
    push('\nendstream\nendobj\n');
  });
  const count = 3 + 3 * pages.length, xref = len;
  push(`xref\n0 ${count}\n0000000000 65535 f \n`);
  for (let n = 1; n < count; n++) push(`${String(offsets[n]).padStart(10, '0')} 00000 n \n`);
  push(`trailer\n<< /Size ${count} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`);
  const out = new Uint8Array(len);
  let at = 0;
  parts.forEach((b) => { out.set(b, at); at += b.length; });
  return out;
}
