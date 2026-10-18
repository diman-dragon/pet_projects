import type { CanvasCardData } from "@/types/canvasCard";
import { PERSON_CARD_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";

export function cardsOverlap(first: CanvasCardData, second: CanvasCardData) {
  return first.x < second.x + PERSON_CARD_WIDTH && first.x + PERSON_CARD_WIDTH > second.x && first.y - PERSON_CARD_HEIGHT / 2 < second.y + PERSON_CARD_HEIGHT / 2 && first.y + PERSON_CARD_HEIGHT / 2 > second.y - PERSON_CARD_HEIGHT / 2;
}
