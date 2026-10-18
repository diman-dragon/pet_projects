import { getGenerationCenterY } from "@/lib/canvas/getGenerationCenterY";
import { MIN_CARD_GAP, PERSON_CARD_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import type { CanvasCardData } from "@/types/canvasCard";

function groupDeltaIsSafe(cards: CanvasCardData[], memberIds: string[], deltaX: number, deltaY: number) {
  const members = cards.filter((card) => memberIds.includes(card.id));
  const others = cards.filter((card) => !memberIds.includes(card.id));
  for (let index = 0; index < members.length; index += 1) {
    const first = members[index];
    const firstX = first.x + deltaX;
    const firstY = first.y + deltaY;
    for (let secondIndex = index + 1; secondIndex < members.length; secondIndex += 1) {
      const second = members[secondIndex];
      const secondX = second.x + deltaX;
      if (getGenerationCenterY(firstY) !== getGenerationCenterY(second.y + deltaY)) continue;
      if (Math.abs(firstX - secondX) < PERSON_CARD_WIDTH + MIN_CARD_GAP
        && Math.abs(firstY - (second.y + deltaY)) < PERSON_CARD_HEIGHT + MIN_CARD_GAP) return false;
    }
    for (const other of others) {
      if (getGenerationCenterY(firstY) !== getGenerationCenterY(other.y)) continue;
      if (Math.abs(firstX - other.x) < PERSON_CARD_WIDTH + MIN_CARD_GAP
        && Math.abs(firstY - other.y) < PERSON_CARD_HEIGHT + MIN_CARD_GAP) return false;
    }
  }
  return true;
}

export function getSafeGroupDelta(cards: CanvasCardData[], memberIds: string[], desiredX: number, desiredY: number, activeId: string) {
  const active = cards.find((card) => card.id === activeId);
  if (!active) return { x: desiredX, y: desiredY };
  const desiredDeltaX = desiredX - active.x;
  const desiredDeltaY = desiredY - active.y;
  if (groupDeltaIsSafe(cards, memberIds, desiredDeltaX, desiredDeltaY)) return { x: desiredDeltaX, y: desiredDeltaY };
  const candidates = [0];
  const members = cards.filter((card) => memberIds.includes(card.id));
  const others = cards.filter((card) => !memberIds.includes(card.id));
  for (const member of members) for (const other of others) {
    if (getGenerationCenterY(member.y + desiredDeltaY) !== getGenerationCenterY(other.y)) continue;
    candidates.push(other.x - member.x - PERSON_CARD_WIDTH - MIN_CARD_GAP - desiredDeltaX);
    candidates.push(other.x + PERSON_CARD_WIDTH + MIN_CARD_GAP - member.x - desiredDeltaX);
  }
  const groupStep = PERSON_CARD_WIDTH + MIN_CARD_GAP;
  for (let index = 1; index <= 200; index += 1) candidates.push(index * groupStep, -index * groupStep);
  const safeOffset = candidates.filter((candidate) => groupDeltaIsSafe(cards, memberIds, desiredDeltaX + candidate, desiredDeltaY)).sort((first, second) => Math.abs(first) - Math.abs(second))[0];
  if (safeOffset !== undefined) return { x: desiredDeltaX + safeOffset, y: desiredDeltaY };

  if (!others.length) return { x: desiredDeltaX, y: desiredDeltaY };

  // A rigid group can be blocked between cards on both sides. Move it beyond
  // the outer edge rather than cancelling the user's drag. This keeps the
  // minimum gap invariant without imposing a finite horizontal canvas limit.
  const memberLeft = Math.min(...members.map((card) => card.x));
  const memberRight = Math.max(...members.map((card) => card.x + PERSON_CARD_WIDTH));
  const otherLeft = Math.min(...others.map((card) => card.x));
  const otherRight = Math.max(...others.map((card) => card.x + PERSON_CARD_WIDTH));
  const leftDelta = otherLeft - MIN_CARD_GAP - memberRight - desiredDeltaX;
  const rightDelta = otherRight + MIN_CARD_GAP - memberLeft - desiredDeltaX;
  const fallback = Math.abs(leftDelta) <= Math.abs(rightDelta) ? leftDelta : rightDelta;
  return { x: desiredDeltaX + fallback, y: desiredDeltaY };
}
