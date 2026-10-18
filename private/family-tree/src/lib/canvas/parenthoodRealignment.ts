import { getGenerationCenterY } from "@/lib/canvas/getGenerationCenterY";
import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationData } from "@/types/marriage";

export function getParenthoodGroupKey(
  cards: CanvasCardData[],
  relations: RelationData[],
  childId: string,
): string {
  const parentIds = relations
    .filter((relation) => relation.type === "parenthood" && relation.personBId === childId)
    .map((relation) => relation.personAId)
    .sort();
  const child = cards.find((card) => card.id === childId);
  return `${parentIds.join("|")}:${child ? getGenerationCenterY(child.y) : "unknown"}`;
}

export function getChildrenForParents(relations: RelationData[], parentIds: Set<string>): string[] {
  return [...new Set(relations
    .filter((relation) => relation.type === "parenthood" && parentIds.has(relation.personAId))
    .map((relation) => relation.personBId))];
}

export function getActiveMarriageMemberIds(relations: RelationData[], personId: string): Set<string> {
  const memberIds = new Set([personId]);
  const marriage = relations.find((relation) => relation.type === "marriage"
    && relation.details?.status !== "dissolved"
    && (relation.personAId === personId || relation.personBId === personId));
  if (marriage) {
    memberIds.add(marriage.personAId);
    memberIds.add(marriage.personBId);
  }
  return memberIds;
}

export function getParentRoleForCard(
  relations: RelationData[],
  childId: string,
  cardId: string,
): "father" | "mother" {
  const parentRelation = relations.find((relation) => relation.type === "parenthood" && relation.personBId === childId && relation.personAId === cardId);
  if (parentRelation) return parentRelation.details?.role === "mother" ? "mother" : "father";
  return "father";
}
