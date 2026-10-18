import { getGenerationCenterY } from "@/lib/canvas/getGenerationCenterY";
import { MIN_CARD_GAP, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import type { CanvasCardData } from "@/types/canvasCard";

export function getNonOverlappingCardX(
  cards: CanvasCardData[],
  cardId: string,
  desiredX: number,
  generationY: number,
  excludedIds: string[] = [cardId],
): number {
  const cardsInGeneration = cards.filter(
    (card) =>
      !excludedIds.includes(card.id) &&
      getGenerationCenterY(card.y) === getGenerationCenterY(generationY),
  );
  const cardStep = PERSON_CARD_WIDTH + MIN_CARD_GAP;
  const candidates = [
    desiredX,
    ...cardsInGeneration.flatMap((card) => [
      card.x - cardStep,
      card.x + cardStep,
    ]),
  ];

  const safeCandidate = candidates
    .filter((candidate) =>
      cardsInGeneration.every(
        (card) =>
          candidate + cardStep <= card.x ||
          candidate >= card.x + cardStep,
      ),
    )
    .sort(
      (firstCandidate, secondCandidate) =>
        Math.abs(firstCandidate - desiredX) -
        Math.abs(secondCandidate - desiredX),
    )[0];
  if (safeCandidate !== undefined) return safeCandidate;

  // There must always be a finite destination. If the local candidate set is
  // blocked on both sides, place the card next to the nearest outer edge
  // instead of returning undefined and silently breaking the drag operation.
  const left = Math.min(...cardsInGeneration.map((card) => card.x)) - cardStep;
  const right = Math.max(...cardsInGeneration.map((card) => card.x)) + cardStep;
  return Math.abs(left - desiredX) <= Math.abs(right - desiredX) ? left : right;
}
