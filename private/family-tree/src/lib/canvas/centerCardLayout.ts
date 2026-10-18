import { getGenerationCenterY } from "@/lib/canvas/getGenerationCenterY";
import { MIN_CARD_GAP, PERSON_CARD_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationData } from "@/types/marriage";

type CardTarget = { x: number; y: number };
const MAX_LAYOUT_ATTEMPTS = 120;

export function getCenteredCardTargets(
  cards: CanvasCardData[],
  _relations: RelationData[],
  ids: string[],
  targetCenterX: number,
  targetCenterY?: number,
  anchorIds: string[] = ids,
) {
  const selected = cards.filter((card) => ids.includes(card.id));
  const anchors = selected.filter((card) => anchorIds.includes(card.id));
  if (!selected.length || !anchors.length) return undefined;

  const anchorLeft = Math.min(...anchors.map((card) => card.x));
  const anchorRight = Math.max(...anchors.map((card) => card.x + PERSON_CARD_WIDTH));
  const currentAnchorCenterX = (anchorLeft + anchorRight) / 2;
  const currentAnchorCenterY = anchors.reduce((sum, card) => sum + card.y, 0) / anchors.length;
  const minimumStep = PERSON_CARD_WIDTH + MIN_CARD_GAP;
  const deltaX = targetCenterX - currentAnchorCenterX;
  const deltaY = targetCenterY === undefined ? 0 : targetCenterY - currentAnchorCenterY;
  const stationary = cards.filter((card) => !ids.includes(card.id));
  const createPositions = (extraX = 0) => {
    const targets = new Map<string, CardTarget>();
    for (const card of selected) {
      targets.set(card.id, {
        x: card.x + deltaX + extraX,
        y: getGenerationCenterY(card.y + deltaY),
      });
    }
    return targets;
  };
  const isSafe = (targets: Map<string, CardTarget>) => {
    const moved = selected.map((card) => ({ ...card, ...(targets.get(card.id) ?? { x: card.x, y: card.y }) }));
    for (let index = 0; index < moved.length; index += 1) {
      for (let next = index + 1; next < moved.length; next += 1) {
        if (getGenerationCenterY(moved[index].y) !== getGenerationCenterY(moved[next].y)) continue;
        if (Math.abs(moved[index].x - moved[next].x) < minimumStep
          && Math.abs(moved[index].y - moved[next].y) < PERSON_CARD_HEIGHT + MIN_CARD_GAP) return false;
      }
      for (const other of stationary) {
        if (getGenerationCenterY(moved[index].y) !== getGenerationCenterY(other.y)) continue;
        if (Math.abs(moved[index].x - other.x) < minimumStep
          && Math.abs(moved[index].y - other.y) < PERSON_CARD_HEIGHT + MIN_CARD_GAP) return false;
      }
    }
    return true;
  };

  let targets = createPositions();
  let safeLayoutFound = isSafe(targets);
  for (let attempt = 1; attempt <= MAX_LAYOUT_ATTEMPTS && !safeLayoutFound; attempt += 1) {
    const direction = attempt % 2 === 0 ? 1 : -1;
    const distance = Math.ceil(attempt / 2) * MIN_CARD_GAP;
    targets = createPositions(direction * distance);
    safeLayoutFound = isSafe(targets);
  }
  // Неразрешимая геометрия не должна блокировать UI бесконечным циклом.
  // Вызывающий код просто сохранит текущую раскладку и позволит исправить её вручную.
  return safeLayoutFound ? targets : undefined;
}
