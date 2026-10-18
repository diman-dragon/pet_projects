import type { CSSProperties } from "react";
import "@/components/Canvas/RelationOverlays.css";

type Point = { x: number; y: number };
type RelationPathProps = { points: Point[] };

export function RelationPath({ points }: RelationPathProps) {
  if (points.length < 2) return null;
  const left = Math.min(...points.map((point) => point.x));
  const top = Math.min(...points.map((point) => point.y));
  const right = Math.max(...points.map((point) => point.x));
  const bottom = Math.max(...points.map((point) => point.y));
  const style: CSSProperties = { left, top, width: Math.max(1, right - left), height: Math.max(1, bottom - top) };
  return <svg className="relation-path" style={style} width={style.width} height={style.height} aria-hidden="true"><polyline className="relation-path__line" points={points.map((point) => `${point.x - left},${point.y - top}`).join(" ")} /></svg>;
}
