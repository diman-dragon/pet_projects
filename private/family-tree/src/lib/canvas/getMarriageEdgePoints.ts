import type { CanvasCardData } from "@/types/canvasCard";
import { PERSON_CARD_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import type { CanvasPoint } from "@/lib/canvas/segmentIntersectsRect";

export function getMarriageEdgePoints(first: CanvasCardData, second: CanvasCardData): [CanvasPoint, CanvasPoint] {
  const firstCenter = { x: first.x + PERSON_CARD_WIDTH / 2, y: first.y };
  const secondCenter = { x: second.x + PERSON_CARD_WIDTH / 2, y: second.y };
  const dx = secondCenter.x - firstCenter.x;
  const dy = secondCenter.y - firstCenter.y;
  const clearance = 10;
  const firstHorizontal = Math.abs(dx) >= Math.abs(dy);
  const firstPoint = firstHorizontal
    ? { x: firstCenter.x + Math.sign(dx || 1) * (PERSON_CARD_WIDTH / 2 + clearance), y: firstCenter.y }
    : { x: firstCenter.x, y: firstCenter.y + Math.sign(dy || 1) * (PERSON_CARD_HEIGHT / 2 + clearance) };
  const secondPoint = firstHorizontal
    ? { x: secondCenter.x - Math.sign(dx || 1) * (PERSON_CARD_WIDTH / 2 + clearance), y: secondCenter.y }
    : { x: secondCenter.x, y: secondCenter.y - Math.sign(dy || 1) * (PERSON_CARD_HEIGHT / 2 + clearance) };
  return [firstPoint, secondPoint];
}
