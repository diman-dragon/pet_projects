import type { CanvasCardData } from "@/types/canvasCard";

export type ParenthoodGroup = {
  children: CanvasCardData[];
  parents: CanvasCardData[];
  autoAlign: boolean;
};

export type HubBaseLayout = {
  hubY: number;
  childIsBelow: boolean;
  parentEdgeY: number;
  upperBoundary: number;
  lowerBoundary: number;
};
