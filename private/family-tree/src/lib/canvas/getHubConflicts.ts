import { PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import { getBaseHubY } from "@/lib/canvas/getBaseHubY";
import type { ParenthoodGroup } from "@/lib/canvas/parenthoodTypes";

export function getHubConflicts(groups: Map<string, ParenthoodGroup>, hubLanes: Map<string, number>) {
  const buckets = new Map<number, Array<{ key: string; start: number; end: number }>>();
  for (const [key, group] of groups) {
    const hubY = getBaseHubY(group).hubY + (hubLanes.get(key) ?? 0);
    const centers = group.children.map((child) => child.x + PERSON_CARD_WIDTH / 2);
    const bucket = Math.round(hubY / 10);
    const items = buckets.get(bucket) ?? [];
    items.push({ key, start: Math.min(...centers), end: Math.max(...centers) });
    buckets.set(bucket, items);
  }
  return [...buckets.values()].flatMap((items) => {
    const conflicts: Array<{ key: string }> = [];
    for (let index = 0; index < items.length; index += 1) {
      for (let next = index + 1; next < items.length; next += 1) {
        if (items[index].start <= items[next].end && items[index].end >= items[next].start) conflicts.push({ key: items[next].key });
      }
    }
    return conflicts;
  });
}
