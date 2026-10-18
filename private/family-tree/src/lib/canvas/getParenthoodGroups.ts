import type { CanvasCardData } from "@/types/canvasCard";
import type { RelationData } from "@/types/marriage";
import { GENERATION_HEIGHT } from "@/lib/canvas/generationConstants";
import type { ParenthoodGroup } from "@/lib/canvas/parenthoodTypes";

export function getParenthoodGroups(cards: CanvasCardData[], relations: RelationData[]) {
  const parenthood = relations.filter((relation) => relation.type === "parenthood");
  const groups = new Map<string, ParenthoodGroup>();
  for (const child of cards) {
    const parentIds = [...new Set(parenthood.filter((relation) => relation.personBId === child.id).map((relation) => relation.personAId))].sort();
    const parents = parentIds.map((id) => cards.find((card) => card.id === id)).filter((card): card is CanvasCardData => Boolean(card));
    if (!parents.length || parents.length !== parentIds.length) continue;
    const key = `${parentIds.join("|")}:${Math.round(child.y / GENERATION_HEIGHT)}`;
    const groupRelations = parenthood.filter((relation) => parentIds.includes(relation.personAId) && relation.personBId === child.id);
    const group = groups.get(key);
    if (group) {
      group.children.push(child);
      if (groupRelations.some((relation) => relation.details?.autoAlign === false)) group.autoAlign = false;
    } else {
      groups.set(key, { children: [child], parents, autoAlign: !groupRelations.some((relation) => relation.details?.autoAlign === false) });
    }
  }
  return groups;
}
