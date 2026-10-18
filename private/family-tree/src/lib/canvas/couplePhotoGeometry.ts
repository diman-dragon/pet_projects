import { GENERATION_VERTICAL_PADDING, PERSON_CARD_HEIGHT, PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import { getGenerationCenterY } from "@/lib/canvas/getGenerationCenterY";

export const COUPLE_PHOTO_GAP = 10;
export const MARRIAGE_HEART_SIZE = 56;

export function getGenerationProtectiveTop(generationCenter: number): number {
  return generationCenter - PERSON_CARD_HEIGHT / 2 - GENERATION_VERTICAL_PADDING;
}

export function getCouplePhotoLayout(heartCenterY: number, cardY: number) {
  const protectiveTop = getGenerationProtectiveTop(getGenerationCenterY(cardY));
  const heartTop = heartCenterY - MARRIAGE_HEART_SIZE / 2;
  const size = Math.max(0, Math.min(PERSON_CARD_WIDTH, heartTop - COUPLE_PHOTO_GAP - protectiveTop));
  return { size, top: protectiveTop };
}
