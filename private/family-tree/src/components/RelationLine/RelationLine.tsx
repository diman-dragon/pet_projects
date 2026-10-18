import type { CSSProperties } from "react";
import "@/components/Canvas/RelationOverlays.css";

type Point = { x: number; y: number };

type RelationLineProps = {
  start: Point;
  end: Point;
  className?: string;
};

export function RelationLine({ start, end, className = "" }: RelationLineProps) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy);
  const angle = Math.atan2(dy, dx);
  const style: CSSProperties = {
    left: start.x,
    top: start.y,
    width: length,
    transform: `rotate(${angle}rad)`,
  };
  return <span className={`relation-line ${className}`} style={style} aria-hidden="true" />;
}
