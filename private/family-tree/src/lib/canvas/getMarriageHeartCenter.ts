import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationData } from "@/types/marriage";
import { PERSON_CARD_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import { getMarriageEdgePoints } from "@/lib/canvas/getMarriageEdgePoints";
import { getPathMidpoint } from "@/lib/canvas/getPathMidpoint";
import { getSafeMarriageRoute } from "@/lib/canvas/getSafeMarriageRoute";
import type { CanvasPoint } from "@/lib/canvas/segmentIntersectsRect";

export function getMarriageHeartCenter(relation: RelationData, cards: CanvasCardData[]): CanvasPoint | undefined {
  if (relation.type !== "marriage") return undefined;
  const first = cards.find((card) => card.id === relation.personAId);
  const second = cards.find((card) => card.id === relation.personBId);
  if (!first || !second) return undefined;
  const [start, end] = getMarriageEdgePoints(first, second);
  const obstacles = cards
    .filter((card) => card.id !== first.id && card.id !== second.id)
    .map((card) => ({
      left: card.x - 12,
      right: card.x + PERSON_CARD_WIDTH + 12,
      top: card.y - PERSON_CARD_HEIGHT / 2 - 12,
      bottom: card.y + PERSON_CARD_HEIGHT / 2 + 12,
    }));
  const route = getSafeMarriageRoute(start, end, obstacles);
  return route ? getPathMidpoint(route) : undefined;
}
