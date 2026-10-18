// Turns picture files into one PDF (a page per picture): decode, fix orientation, shrink, JPEG, assemble.
import { buildPdf } from '../data/pdf-writer.js';
const MAX_SIDE = 2600, QUALITY = 0.88;
async function decode(file) {
  try { return await createImageBitmap(file, { imageOrientation: 'from-image' }); } catch { /* older WebView: plain image decoder */ }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    return img;
  } finally { URL.revokeObjectURL(url); }
}
async function toJpeg(file) {
  const src = await decode(file);
  const sw = src.width || src.naturalWidth, sh = src.height || src.naturalHeight;
  const k = Math.min(1, MAX_SIDE / Math.max(sw, sh));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(sw * k));
  c.height = Math.max(1, Math.round(sh * k));
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(src, 0, 0, c.width, c.height);
  src.close?.();
  const blob = await new Promise((res) => c.toBlob(res, 'image/jpeg', QUALITY));
  if (!blob) throw new Error('jpeg encode failed');
  return { jpeg: new Uint8Array(await blob.arrayBuffer()), w: c.width, h: c.height };
}
export async function imagesToPdf(files, onProgress) {
  const pages = [];
  for (let i = 0; i < files.length; i++) {
    pages.push(await toJpeg(files[i]));
    onProgress?.(i + 1, files.length);
  }
  return new Blob([buildPdf(pages)], { type: 'application/pdf' });
}
