export type CanvasRect = {
  left: number;
  right: number;
  top: number;
  bottom: number;
};

export type CanvasPoint = { x: number; y: number };

export function segmentIntersectsRect(start: CanvasPoint, end: CanvasPoint, rect: CanvasRect) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  let entry = 0;
  let exit = 1;
  for (const [origin, direction, minimum, maximum] of [
    [start.x, dx, rect.left, rect.right],
    [start.y, dy, rect.top, rect.bottom],
  ] as const) {
    if (Math.abs(direction) < 0.0001) {
      if (origin <= minimum || origin >= maximum) return false;
      continue;
    }
    const first = (minimum - origin) / direction;
    const second = (maximum - origin) / direction;
    entry = Math.max(entry, Math.min(first, second));
    exit = Math.min(exit, Math.max(first, second));
    if (entry >= exit) return false;
  }
  return exit > 0 && entry < 1;
}
