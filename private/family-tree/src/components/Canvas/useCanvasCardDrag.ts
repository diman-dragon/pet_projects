import { useRef } from "react";
import type { PointerEvent } from "react";
import { useCanvasCardsStore } from "@/store/useCanvasCardsStore";
import { getGenerationCenterY } from "@/lib/canvas/getGenerationCenterY";
import type { CanvasCardData } from "@/types/canvasCard";
import type { CanvasViewport } from "@/types/canvasViewport";

type UseCanvasCardDragOptions = {
  card: CanvasCardData;
  viewport: CanvasViewport;
  locked: boolean;
  onMoveFinished: (cardId: string, movedAsGroup: boolean) => Promise<void>;
  onSelectionToggle: (cardId: string, clientX: number, clientY: number) => void;
};

export function useCanvasCardDrag({ card, viewport, locked, onMoveFinished, onSelectionToggle }: UseCanvasCardDragOptions) {
  const moveCard = useCanvasCardsStore((state) => state.moveCard);
  const beginMoveCard = useCanvasCardsStore((state) => state.beginMoveCard);
  const cancelMoveCard = useCanvasCardsStore((state) => state.cancelMoveCard);
  const finishMoveCard = useCanvasCardsStore((state) => state.finishMoveCard);
  const selectCard = useCanvasCardsStore((state) => state.selectCard);
  const toggleCardSelection = useCanvasCardsStore((state) => state.toggleCardSelection);
  const dragOffset = useRef({ x: 0, y: 0 });
  const dragZoom = useRef(1);
  const moveAsGroup = useRef(true);
  const didMove = useRef(false);
  const pointerStart = useRef({ x: 0, y: 0 });

  function getWorldPoint(event: PointerEvent<HTMLDivElement>) {
    const canvas = event.currentTarget.closest(".canvas") as HTMLDivElement | null;
    const bounds = canvas?.getBoundingClientRect();
    return bounds
      ? { x: (event.clientX - (bounds.left + bounds.width / 2) - viewport.pan.x) / dragZoom.current, y: (event.clientY - (bounds.top + bounds.height / 2) - viewport.pan.y) / dragZoom.current }
      : { x: event.clientX / dragZoom.current, y: event.clientY / dragZoom.current };
  }

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    event.stopPropagation();
    if (event.ctrlKey || event.metaKey) {
      event.preventDefault();
      toggleCardSelection(card.id);
      onSelectionToggle(card.id, event.clientX, event.clientY);
      return;
    }
    selectCard(card.id);
    if (locked) return;
    didMove.current = false;
    pointerStart.current = { x: event.clientX, y: event.clientY };
    moveAsGroup.current = !event.altKey;
    beginMoveCard(card.id, moveAsGroup.current);
    dragZoom.current = viewport.zoom;
    const worldPoint = getWorldPoint(event);
    dragOffset.current = { x: worldPoint.x - card.x, y: worldPoint.y - card.y };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    if (event.buttons !== 1) return;
    if (Math.abs(event.clientX - pointerStart.current.x) > 2 || Math.abs(event.clientY - pointerStart.current.y) > 2) didMove.current = true;
    const worldPoint = getWorldPoint(event);
    moveCard(card.id, worldPoint.x - dragOffset.current.x, getGenerationCenterY(worldPoint.y - dragOffset.current.y), moveAsGroup.current);
  }

  async function handlePointerUp(event: PointerEvent<HTMLDivElement>) {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (!didMove.current) {
      cancelMoveCard(card.id);
      return;
    }
    const worldPoint = getWorldPoint(event);
    await finishMoveCard(card.id, worldPoint.x - dragOffset.current.x, getGenerationCenterY(worldPoint.y - dragOffset.current.y), moveAsGroup.current);
    await onMoveFinished(card.id, moveAsGroup.current);
  }

  return { handlePointerDown, handlePointerMove, handlePointerUp, didMove };
}
