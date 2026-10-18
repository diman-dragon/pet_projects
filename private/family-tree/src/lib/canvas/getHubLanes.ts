import { GENERATION_HEIGHT, GENERATION_HUB_GAP } from "@/lib/canvas/generationConstants";
import { getBaseHubY } from "@/lib/canvas/getBaseHubY";
import type { ParenthoodGroup } from "@/lib/canvas/parenthoodTypes";

export function getHubLanes(groups: Map<string, ParenthoodGroup>, createdHubLanes: Record<string, number>) {
  const result = new Map<string, number>();
  const groupsByGenerationGap = new Map<string, Array<{ key: string; group: ParenthoodGroup }>>();
  for (const [key, group] of groups) {
    const base = getBaseHubY(group);
    const parentGeneration = Math.round(group.parents.reduce((sum, parent) => sum + parent.y, 0) / group.parents.length / GENERATION_HEIGHT);
    const childGeneration = Math.round(group.children.reduce((sum, child) => sum + child.y, 0) / group.children.length / GENERATION_HEIGHT);
    const gapKey = `${parentGeneration}:${childGeneration}:${base.childIsBelow ? "down" : "up"}`;
    const gapGroups = groupsByGenerationGap.get(gapKey) ?? [];
    gapGroups.push({ key, group });
    groupsByGenerationGap.set(gapKey, gapGroups);
  }

  for (const gapGroups of groupsByGenerationGap.values()) {
    const manual = gapGroups.filter((item) => !item.group.autoAlign);
    const automatic = gapGroups.filter((item) => item.group.autoAlign);
    if (!gapGroups.length) continue;
    const base = getBaseHubY(gapGroups[0].group);
    const manualTargets = manual.map((_item, index) => {
      const reverseIndex = manual.length - index - 1;
      return base.childIsBelow
        ? base.upperBoundary + GENERATION_HUB_GAP + reverseIndex * GENERATION_HUB_GAP
        : base.upperBoundary - GENERATION_HUB_GAP - reverseIndex * GENERATION_HUB_GAP;
    });
    manual.forEach((item, index) => {
      result.set(item.key, manualTargets[index] - getBaseHubY(item.group).hubY + (createdHubLanes[item.key] ?? 0) * GENERATION_HUB_GAP);
    });
    if (automatic.length) {
      const manualBoundary = manualTargets.length
        ? (base.childIsBelow ? Math.max(...manualTargets) : Math.min(...manualTargets))
        : base.upperBoundary;
      const automaticTarget = (manualBoundary + base.lowerBoundary) / 2;
      for (const item of automatic) result.set(item.key, automaticTarget - getBaseHubY(item.group).hubY);
    }
  }
  return result;
}
