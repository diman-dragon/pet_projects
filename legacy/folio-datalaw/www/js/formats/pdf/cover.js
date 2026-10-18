// First-page thumbnail of a PDF as a JPEG blob (library cover) plus the page count.
import { openPdf } from './open-doc.js';
export async function pdfCover(blob, { width = 320 } = {}) {
  const doc = await openPdf(blob, { askPassword: false });
  try {
    const page = await doc.getPage(1);
    const vp = page.getViewport({ scale: width / page.getViewport({ scale: 1 }).width });
    const canvas = document.createElement('canvas');
    canvas.width = Math.floor(vp.width);
    canvas.height = Math.floor(vp.height);
    await page.render({ canvas, viewport: vp, background: '#ffffff' }).promise;
    const out = await new Promise((res) => canvas.toBlob(res, 'image/jpeg', 0.82));
    canvas.width = canvas.height = 0;
    return { blob: out, pages: doc.numPages };
  } finally {
    doc.loadingTask.destroy();
  }
}
