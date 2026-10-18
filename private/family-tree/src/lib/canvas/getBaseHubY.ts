import { GENERATION_VERTICAL_PADDING, PERSON_CARD_HEIGHT } from "@/lib/canvas/generationConstants";
import type { ParenthoodGroup, HubBaseLayout } from "@/lib/canvas/parenthoodTypes";

export function getBaseHubY(group: ParenthoodGroup): HubBaseLayout {
  const childCenterY = group.children.reduce((sum, child) => sum + child.y, 0) / group.children.length;
  const parentCenterY = group.parents.reduce((sum, parent) => sum + parent.y, 0) / group.parents.length;
  const childIsBelow = childCenterY >= parentCenterY;
  const childEdgeY = childIsBelow
    ? Math.min(...group.children.map((child) => child.y - PERSON_CARD_HEIGHT / 2))
    : Math.max(...group.children.map((child) => child.y + PERSON_CARD_HEIGHT / 2));
  const parentEdgeY = childIsBelow
    ? Math.max(...group.parents.map((parent) => parent.y + PERSON_CARD_HEIGHT / 2))
    : Math.min(...group.parents.map((parent) => parent.y - PERSON_CARD_HEIGHT / 2));
  const upperBoundary = childIsBelow ? parentEdgeY + GENERATION_VERTICAL_PADDING : parentEdgeY - GENERATION_VERTICAL_PADDING;
  const lowerBoundary = childIsBelow ? childEdgeY - GENERATION_VERTICAL_PADDING : childEdgeY + GENERATION_VERTICAL_PADDING;
  return { hubY: (upperBoundary + lowerBoundary) / 2, childIsBelow, parentEdgeY, upperBoundary, lowerBoundary };
}
