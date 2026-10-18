import type { CanvasCardData } from "@/types/canvasCard";
import { PERSON_CARD_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";

export type CardsBounds = { left: number; right: number; top: number; bottom: number };

export function getCardsBounds(cards: CanvasCardData[]): CardsBounds | null {
  if (!cards.length) return null;
  return {
    left: Math.min(...cards.map((card) => card.x)),
    right: Math.max(...cards.map((card) => card.x + PERSON_CARD_WIDTH)),
    top: Math.min(...cards.map((card) => card.y - PERSON_CARD_HEIGHT / 2)),
    bottom: Math.max(...cards.map((card) => card.y + PERSON_CARD_HEIGHT / 2)),
  };
}
