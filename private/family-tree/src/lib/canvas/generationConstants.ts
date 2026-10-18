export const PERSON_CARD_WIDTH = 250;
export const PERSON_CARD_HEIGHT = 350;
export const GENERATION_VERTICAL_PADDING = 40;
export const GENERATION_HUB_GAP = 20;
export const MIN_CARD_GAP = 100;

export function getIntergenerationGap(hubCount = 0): number {
  return GENERATION_VERTICAL_PADDING * 2 + Math.max(0, hubCount - 1) * GENERATION_HUB_GAP;
}

export function getGenerationHeight(hubCount = 0): number {
  return PERSON_CARD_HEIGHT + GENERATION_VERTICAL_PADDING * 2 + getIntergenerationGap(hubCount);
}

export const GENERATION_HEIGHT = getGenerationHeight();
