// Page thumbnail: scaled copy of the page's base canvas when it is on screen, otherwise a low-priority queued render.
import { renderPage } from './render-page.js';
import { enqueue } from './render-queue.js';
export function renderThumb({ entries }, doc, n, canvas, cssWidth, near) {
  const e = entries[n - 1];
  canvas.width = Math.round(cssWidth * Math.min(window.devicePixelRatio || 1, 2));
  if (e.base && e.baseW) {
    canvas.height = Math.round((canvas.width * e.base.height) / e.base.width);
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(e.base, 0, 0, canvas.width, canvas.height);
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => enqueue(`t${n}`, 1000 + Math.abs(n - near), async (ctl) => {
    try {
      const page = await doc.getPage(n);
      if (ctl.dead) return reject(new Error('cancelled'));
      ctl.task = renderPage(page, canvas, cssWidth, { maxPx: 400_000 });
      await ctl.task.promise;
      resolve();
    } catch (err) { reject(err); }
  }));
}
