// Live drawing of a pressure pen stroke on a light bitmap (cheap per segment); the stroke becomes a vector path when the pen lifts.
export const gain = (p) => 0.25 + 1.5 * p; // pressure -> width factor (same rule as annot-svg outlineD)
export function liveCanvas(el, color) {
  const d = Math.min(window.devicePixelRatio || 1, 2, Math.sqrt(4e6 / Math.max(1, el.offsetWidth * el.offsetHeight)));
  const c = Object.assign(document.createElement('canvas'), { className: 'annot-live-ink', width: Math.round(el.offsetWidth * d), height: Math.round(el.offsetHeight * d) });
  el.append(c);
  const ctx = c.getContext('2d');
  Object.assign(ctx, { lineCap: 'round', strokeStyle: color });
  return { c, ctx };
}
// a, b: [x, y, pressure] in 0..1 page coordinates; w: base width (fraction of the page width).
export function segment({ c, ctx }, a, b, w) {
  ctx.lineWidth = w * c.width * gain((a[2] + b[2]) / 2);
  ctx.beginPath();
  ctx.moveTo(a[0] * c.width, a[1] * c.height);
  ctx.lineTo(b[0] * c.width, b[1] * c.height);
  ctx.stroke();
}
