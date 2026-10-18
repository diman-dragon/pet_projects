import { MIN_CARD_GAP, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import type { CanvasCardData } from "@/types/canvasCard";
import { loadStoredLockedAreas, type LockedAreaBounds } from "@/lib/canvas/lockedAreaStorage";

export type LockedArea = { id: string; cards: CanvasCardData[]; bounds?: LockedAreaBounds };

export function getLockedAreas(cards: CanvasCardData[]): LockedArea[] {
  const locked = cards.filter((card) => card.locked).sort((first, second) => first.y - second.y || first.x - second.x);
  const areas: LockedArea[] = [];
  const lockedById = new Map(locked.map((card) => [card.id, card]));
  const storedCardIds = new Set<string>();
  for (const area of loadStoredLockedAreas()) {
    const areaCards = area.cardIds.map((id) => lockedById.get(id)).filter((card): card is CanvasCardData => Boolean(card));
    if (!areaCards.length) continue;
    areaCards.forEach((card) => storedCardIds.add(card.id));
    areas.push({ id: area.id, cards: areaCards, bounds: area.bounds });
  }
  for (const card of locked) {
    if (storedCardIds.has(card.id)) continue;
    const area = areas.find((candidate) => {
      const anchor = candidate.cards[0];
      return Math.abs(anchor.y - card.y) < 0.01 && candidate.cards.some((item) => Math.abs(item.x - card.x) <= PERSON_CARD_WIDTH + MIN_CARD_GAP);
    });
    if (area) area.cards.push(card);
    else areas.push({ id: `locked-area-${areas.length + 1}`, cards: [card] });
  }
  return areas;
}
