// Renders a page into a canvas sized for text recognition (about 150 dpi of an A4 page); sizes are in page units.
import { enqueue } from './render-queue.js';
const TARGET_W = 1600;
export function renderForOcr(doc, n) {
  return new Promise((resolve, reject) => enqueue(`o${n}`, 5, async (ctl) => {
    try {
      const page = await doc.getPage(n);
      const base = page.getViewport({ scale: 1 });
      const viewport = page.getViewport({ scale: Math.min(TARGET_W / base.width, 2400 / base.height) });
      const canvas = Object.assign(document.createElement('canvas'), { width: Math.floor(viewport.width), height: Math.floor(viewport.height) });
      ctl.task = page.render({ canvas, viewport });
      await ctl.task.promise;
      resolve({ canvas, w: base.width, h: base.height });
    } catch (err) { reject(err); }
  }));
}
