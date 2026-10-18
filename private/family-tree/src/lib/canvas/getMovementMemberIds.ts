import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationData } from "@/types/marriage";

type CanvasGeometryState = { cards: CanvasCardData[]; relations: RelationData[] };

export function getMovementMemberIds(state: CanvasGeometryState, id: string, moveAsGroup: boolean) {
  const ids = new Set<string>([id]);
  if (!moveAsGroup) return [...ids];
  const marriage = state.relations.find((relation) => relation.type === "marriage"
    && relation.details?.status !== "dissolved"
    && (relation.personAId === id || relation.personBId === id));
  if (marriage) {
    ids.add(marriage.personAId);
    ids.add(marriage.personBId);
  }
  return [...ids];
}
