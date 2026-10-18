import { GENERATION_HEIGHT } from "@/lib/canvas/generationConstants";

export function getGenerationCenterY(worldY: number): number {
  return Math.round(worldY / GENERATION_HEIGHT) * GENERATION_HEIGHT;
}
