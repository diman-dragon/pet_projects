import type { CSSProperties, MouseEventHandler } from "react";
import "@/components/Canvas/MarriageOverlays.css";

type RelationHeartProps = {
  variant?: "solid" | "broken";
  className?: string;
  style?: CSSProperties;
  size?: number;
  onContextMenu?: MouseEventHandler<SVGSVGElement>;
  onClick?: MouseEventHandler<SVGSVGElement>;
};

export function RelationHeart({ variant = "solid", className = "", style, size = 56, onContextMenu, onClick }: RelationHeartProps) {
  return (
    <svg
      className={`relation-heart ${className}`}
      style={{ ...style, width: size, height: size }}
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
      onContextMenu={onContextMenu}
      onClick={onClick}
    >
      <path className="relation-heart__part" d="M12 21s-7-4.35-9.5-8.75C.7 8.4 2.6 4.5 6.5 4.5c2.1 0 3.7 1.2 5.5 3.1 1.8-1.9 3.4-3.1 5.5-3.1 3.9 0 5.8 3.9 4 7.75C19 16.65 12 21 12 21Z" />
      {variant === "broken" && <path className="relation-heart__slash" d="M4.2 4.8 19.8 19.2" />}
    </svg>
  );
}
