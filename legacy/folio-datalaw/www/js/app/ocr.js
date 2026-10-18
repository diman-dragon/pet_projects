// OCR orchestration: render a page for recognition, run the engine, keep the result as a selectable text layer and in the library.
import { recognize } from '../io/ocr-engine.js';
import { saveOcr } from '../data/ocr-store.js';
export async function recognizePage(handle, docId, i, langs, onProgress) {
  const src = await handle.renderForOcr(i);
  const words = await recognize(src.canvas, langs, onProgress);
  const k = src.w / src.canvas.width;
  src.canvas.width = 0;
  const data = { w: src.w, h: src.h, words: words.map((x) => [x.text, x.x0 * k, x.y0 * k, (x.x1 - x.x0) * k, (x.y1 - x.y0) * k]) };
  handle.setOcr(i, data);
  await saveOcr(docId, i, data);
  return data;
}
// Pages that already have text (native or recognized) are skipped. Returns how many pages were recognized.
export async function recognizePages(handle, docId, pages, langs, { onPage, cancelled }) {
  let done = 0;
  for (let n = 0; n < pages.length && !cancelled(); n++) {
    const i = pages[n];
    onPage(n + 1, pages.length, 0);
    if (await handle.hasText(i)) continue;
    await recognizePage(handle, docId, i, langs, (p) => onPage(n + 1, pages.length, p));
    done++;
  }
  return done;
}
