import type { CanvasCardData } from "@/types/canvasCard";
import {
  GENERATION_HEIGHT,
  GENERATION_VERTICAL_PADDING,
  PERSON_CARD_HEIGHT,
} from "@/lib/canvas/generationConstants";
import { getGenerationCenterY } from "@/lib/canvas/getGenerationCenterY";
import "@/components/Canvas/GenerationGuides.css";

type GenerationGuidesProps = {
  cards: CanvasCardData[];
  panY: number;
  zoom: number;
  visible?: boolean;
};

type GenerationBounds = {
  top: number;
  bottom: number;
};

function getGenerationBounds(cards: CanvasCardData[]) {
  const groups = new Map<number, GenerationBounds>();

  for (const card of cards) {
    const generationY = getGenerationCenterY(card.y);
    const generation = Math.round(getGenerationCenterY(generationY) / GENERATION_HEIGHT);
    const cardTop = generationY - PERSON_CARD_HEIGHT / 2;
    const cardBottom = generationY + PERSON_CARD_HEIGHT / 2;
    const current = groups.get(generation);

    if (!current) {
      groups.set(generation, {
        top: cardTop - GENERATION_VERTICAL_PADDING,
        bottom: cardBottom + GENERATION_VERTICAL_PADDING,
      });
      continue;
    }

    current.top = Math.min(current.top, cardTop - GENERATION_VERTICAL_PADDING);
    current.bottom = Math.max(current.bottom, cardBottom + GENERATION_VERTICAL_PADDING);
  }

  return [...groups.values()].sort((first, second) => first.top - second.top);
}

export function GenerationGuides({ cards, panY, zoom, visible = true }: GenerationGuidesProps) {
  return (
    <>
      {getGenerationBounds(cards).map((bounds, index) => {
        const top = `calc(50% + ${panY + bounds.top * zoom}px)`;
        const height = `${(bounds.bottom - bounds.top) * zoom}px`;
        return (
          <div
            className={`generation-guides${visible ? "" : " generation-guides--hidden"}`}
            style={{ top, height }}
            key={`${bounds.top}-${index}`}
          >
            <div className="generation-guides__line generation-guides__line--top" />
            <div className="generation-guides__line generation-guides__line--bottom" />
          </div>
        );
      })}
    </>
  );
}
