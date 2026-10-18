// Offline OCR engine: tesseract.js + wasm core + language data, fetched into www/vendor/tesseract by scripts/fetch-ocr.mjs.
const base = new URL('../../vendor/tesseract/', import.meta.url).href;
let info, worker = null, key = '', progress = null;
export async function ocrInfo() {
  info ??= await fetch(`${base}ready.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  return info;
}
export async function stopOcr() {
  const w = worker;
  worker = null;
  key = '';
  try { await w?.terminate(); } catch { /* already gone */ }
}
const SIMD = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
// Exact core file (SIMD when the WebView supports it) so the library does not have to guess names.
function corePath(meta) {
  const want = `tesseract-core${WebAssembly.validate(SIMD) ? '-simd' : ''}-lstm.wasm.js`;
  return meta.core?.includes(want) ? `${base}core/${want}` : `${base}core`;
}
// Starts the worker; if the first way fails (workers from a plain URL, then from a blob) the next one is tried.
async function start(meta, langs) {
  const { createWorker } = await import(`${base}tesseract.esm.min.js`);
  const errors = [];
  for (const blob of [false, true]) {
    try {
      return await createWorker(langs, 1, {
        workerPath: `${base}worker.min.js`, corePath: corePath(meta), langPath: `${base}lang`, gzip: true, cacheMethod: 'none', workerBlobURL: blob,
        logger: (m) => { if (m.status === 'recognizing text') progress?.(m.progress); },
      });
    } catch (err) { errors.push(String(err?.message ?? err)); }
  }
  throw new Error(`engine: ${errors.join(' | ')}`);
}
const wordsOf = (d) => d.words ?? (d.blocks ?? []).flatMap((b) => b.paragraphs.flatMap((p) => p.lines.flatMap((l) => l.words)));
const toBlob = (canvas) => new Promise((res, rej) => canvas.toBlob((b) => (b ? res(b) : rej(new Error('canvas.toBlob failed'))), 'image/png'));
// Recognizes a canvas; resolves with words [{ text, x0, y0, x1, y1 }] in canvas px.
export async function recognize(canvas, langs, onProgress) {
  const meta = await ocrInfo();
  if (!meta) throw new Error('ocr-missing');
  const use = langs.filter((l) => meta.langs.includes(l));
  if (!use.length) throw new Error('ocr-lang');
  progress = onProgress;
  if (worker && key !== use.join('+')) await stopOcr();
  if (!worker) { key = use.join('+'); worker = await start(meta, key); }
  const { data } = await worker.recognize(await toBlob(canvas));
  return wordsOf(data)
    .filter((w) => w.text?.trim() && w.confidence >= 35)
    .map((w) => ({ text: w.text.trim(), x0: w.bbox.x0, y0: w.bbox.y0, x1: w.bbox.x1, y1: w.bbox.y1 }));
}
