import { segmentIntersectsRect, type CanvasPoint, type CanvasRect } from "@/lib/canvas/segmentIntersectsRect";

export function getSafeMarriageRoute(start: CanvasPoint, end: CanvasPoint, obstacles: CanvasRect[]): CanvasPoint[] | undefined {
  if (!obstacles.some((rect) => segmentIntersectsRect(start, end, rect))) return [start, end];
  const nodes = [start, end];
  for (const rect of obstacles) nodes.push({ x: rect.left, y: rect.top }, { x: rect.right, y: rect.top }, { x: rect.right, y: rect.bottom }, { x: rect.left, y: rect.bottom });
  const distances = nodes.map((_, index) => index === 0 ? 0 : Number.POSITIVE_INFINITY);
  const previous: Array<number | undefined> = nodes.map(() => undefined);
  const visited = new Set<number>();
  while (visited.size < nodes.length) {
    let current = -1;
    let best = Number.POSITIVE_INFINITY;
    for (let index = 0; index < nodes.length; index += 1) {
      if (!visited.has(index) && distances[index] < best) { current = index; best = distances[index]; }
    }
    if (current < 0) break;
    visited.add(current);
    for (let next = 0; next < nodes.length; next += 1) {
      if (visited.has(next) || next === current || obstacles.some((rect) => segmentIntersectsRect(nodes[current], nodes[next], rect))) continue;
      const distance = best + Math.hypot(nodes[next].x - nodes[current].x, nodes[next].y - nodes[current].y);
      if (distance < distances[next]) { distances[next] = distance; previous[next] = current; }
    }
  }
  if (!Number.isFinite(distances[1])) return undefined;
  const path: CanvasPoint[] = [];
  for (let current: number | undefined = 1; current !== undefined; current = previous[current]) path.unshift(nodes[current]);
  return path;
}
