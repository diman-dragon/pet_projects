import type { CanvasPoint } from "@/lib/canvas/segmentIntersectsRect";

export function getPathMidpoint(path: CanvasPoint[]) {
  const lengths = path.slice(1).map((point, index) => Math.hypot(point.x - path[index].x, point.y - path[index].y));
  const half = lengths.reduce((sum, length) => sum + length, 0) / 2;
  let passed = 0;
  for (let index = 0; index < lengths.length; index += 1) {
    if (passed + lengths[index] >= half) {
      const progress = lengths[index] ? (half - passed) / lengths[index] : 0;
      return { x: path[index].x + (path[index + 1].x - path[index].x) * progress, y: path[index].y + (path[index + 1].y - path[index].y) * progress };
    }
    passed += lengths[index];
  }
  return path[path.length - 1];
}
