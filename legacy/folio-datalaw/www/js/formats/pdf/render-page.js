// Renders a pdf.js page (whole, or just a css-px window of it) into a canvas at device resolution, capped by pixel count.
const screenPx = () => innerWidth * innerHeight * Math.min(window.devicePixelRatio || 1, 2) ** 2;
// Caps scale with the screen: a phone needs ~1.7 MP per canvas, a tablet several.
export const baseCap = () => Math.min(8e6, Math.max(1.5e6, screenPx() * 1.4));
export const detailCap = () => Math.min(8e6, Math.max(2e6, screenPx() * 1.7));
export function renderPage(page, canvas, cssWidth, { maxPx = baseCap(), rect = null } = {}) {
  const base = page.getViewport({ scale: 1 });
  const k = cssWidth / base.width;
  const w = rect ? rect.w : cssWidth;
  const h = rect ? rect.h : base.height * k;
  let d = Math.min(window.devicePixelRatio || 1, 2);
  if (w * h * d * d > maxPx) d = Math.sqrt(maxPx / (w * h));
  const viewport = page.getViewport({ scale: k * d, offsetX: rect ? -rect.x * d : 0, offsetY: rect ? -rect.y * d : 0 });
  canvas.width = Math.max(1, Math.floor(w * d));
  canvas.height = Math.max(1, Math.floor(h * d));
  return page.render({ canvas, viewport });
}
