import { getGenerationCenterY } from "@/lib/canvas/getGenerationCenterY";
import { GENERATION_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationData } from "@/types/marriage";

export type ParenthoodAlignment = {
  effectiveParents: string[];
  children: string[];
  groupIds: string[];
  targetCenter?: number;
  targetGenerationY?: number;
  requiresMarriage: boolean;
};

export function getParenthoodAlignment(
  cards: CanvasCardData[],
  relations: RelationData[],
  childId: string,
  role: "father" | "mother",
  changedParentId: string,
  preserveGeneration = false,
  includeParentsInGroup = false,
): ParenthoodAlignment {
  const parentRelations = relations.filter((relation) => relation.type === "parenthood" && relation.personBId === childId);
  const fatherId = parentRelations.find((relation) => relation.details?.role === "father")?.personAId ?? (role === "father" ? changedParentId : "");
  const motherId = parentRelations.find((relation) => relation.details?.role === "mother")?.personAId ?? (role === "mother" ? changedParentId : "");
  const effectiveParents = [fatherId, motherId].filter(Boolean);
  const marriage = fatherId && motherId ? relations.find((relation) => relation.type === "marriage" && ((relation.personAId === fatherId && relation.personBId === motherId) || (relation.personAId === motherId && relation.personBId === fatherId))) : undefined;
  const parentSet = new Set(effectiveParents);
  const childCard = cards.find((card) => card.id === childId);
  const childGeneration = childCard ? getGenerationCenterY(childCard.y) : undefined;
  const candidateChildIds = [...new Set(relations.filter((relation) => relation.type === "parenthood" && parentSet.has(relation.personAId)).map((relation) => relation.personBId))];
  const children = candidateChildIds.filter((candidateId) => {
    const candidateCard = cards.find((card) => card.id === candidateId);
    if (!candidateCard || childGeneration === undefined || getGenerationCenterY(candidateCard.y) !== childGeneration) return false;
    const candidateParents = new Set(relations.filter((relation) => relation.type === "parenthood" && relation.personBId === candidateId).map((relation) => relation.personAId));
    return candidateParents.size === parentSet.size && [...candidateParents].every((parentId) => parentSet.has(parentId));
  });
  const groupIds = new Set(children);
  if (includeParentsInGroup) {
    for (const parentId of effectiveParents) groupIds.add(parentId);
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (const relation of relations) {
      if (relation.type !== "marriage" || relation.details?.status === "dissolved") continue;
      if (groupIds.has(relation.personAId) && !groupIds.has(relation.personBId)) { groupIds.add(relation.personBId); changed = true; }
      if (groupIds.has(relation.personBId) && !groupIds.has(relation.personAId)) { groupIds.add(relation.personAId); changed = true; }
    }
  }
  if (!groupIds.size) groupIds.add(childId);
  const parentCards = effectiveParents.map((id) => cards.find((card) => card.id === id)).filter((card): card is CanvasCardData => Boolean(card));
  const targetCenter = marriage && parentCards.length === 2
    ? (parentCards[0].x + PERSON_CARD_WIDTH / 2 + parentCards[1].x + PERSON_CARD_WIDTH / 2) / 2
    : parentCards[0] ? parentCards[0].x + PERSON_CARD_WIDTH / 2 : undefined;
  const targetGenerationY = preserveGeneration ? undefined : parentCards.length ? Math.max(...parentCards.map((card) => card.y)) + GENERATION_HEIGHT : undefined;
  return { effectiveParents, children, groupIds: [...groupIds], targetCenter, targetGenerationY, requiresMarriage: Boolean(fatherId && motherId && !marriage && !preserveGeneration) };
}
