import { useCallback, useRef, useState } from "react";
import type { PointerEvent } from "react";
import { PERSON_CARD_WIDTH } from "@/lib/canvas/generationConstants";
import type { CanvasCardData } from "@/types/canvasCard";

type CanvasPoint = { x: number; y: number };
type ScreenPoint = { x: number; y: number };

type AreaSelection = {
  left: number;
  top: number;
  width: number;
  height: number;
  cardIds: string[];
  worldBounds: { left: number; top: number; right: number; bottom: number } | null;
};

type CanvasElement = HTMLDivElement;

type UseCanvasAreaSelectionOptions = {
  canvasRef: React.RefObject<CanvasElement | null>;
  cards: CanvasCardData[];
  screenToWorld: (point: ScreenPoint) => CanvasPoint | null;
  onPanPointerDown: (event: PointerEvent<CanvasElement>) => void;
  onPanPointerMove: (event: PointerEvent<CanvasElement>) => void;
  onPanPointerUp: (event: PointerEvent<CanvasElement>) => void;
};

export function useCanvasAreaSelection({
  canvasRef,
  cards,
  screenToWorld,
  onPanPointerDown,
  onPanPointerMove,
  onPanPointerUp,
}: UseCanvasAreaSelectionOptions) {
  const [areaSelection, setAreaSelection] = useState<AreaSelection | null>(null);
  const selectionDrag = useRef<ScreenPoint | null>(null);
  const selectionClickGuard = useRef(false);

  const getCanvasLocalPoint = useCallback((event: PointerEvent<CanvasElement>) => {
    const bounds = canvasRef.current?.getBoundingClientRect();
    return bounds
      ? { x: event.clientX - bounds.left, y: event.clientY - bounds.top }
      : { x: event.clientX, y: event.clientY };
  }, [canvasRef]);

  const handlePointerDown = useCallback((event: PointerEvent<CanvasElement>) => {
    if (event.button === 0 && event.shiftKey && event.target === event.currentTarget) {
      event.preventDefault();
      event.stopPropagation();
      selectionDrag.current = { x: event.clientX, y: event.clientY };
      setAreaSelection(null);
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }
    onPanPointerDown(event);
  }, [onPanPointerDown]);

  const handlePointerMove = useCallback((event: PointerEvent<CanvasElement>) => {
    const start = selectionDrag.current;
    if (!start) {
      onPanPointerMove(event);
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const point = getCanvasLocalPoint(event);
    const origin = getCanvasLocalPoint({ ...event, clientX: start.x, clientY: start.y } as PointerEvent<CanvasElement>);
    setAreaSelection((current) => ({
      left: Math.min(origin.x, point.x),
      top: Math.min(origin.y, point.y),
      width: Math.abs(point.x - origin.x),
      height: Math.abs(point.y - origin.y),
      cardIds: current?.cardIds ?? [],
      worldBounds: current?.worldBounds ?? null,
    }));
  }, [getCanvasLocalPoint, onPanPointerMove]);

  const handlePointerUp = useCallback((event: PointerEvent<CanvasElement>) => {
    const start = selectionDrag.current;
    if (!start) {
      onPanPointerUp(event);
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const end = { x: event.clientX, y: event.clientY };
    const topLeft = screenToWorld({ x: Math.min(start.x, end.x), y: Math.min(start.y, end.y) });
    const bottomRight = screenToWorld({ x: Math.max(start.x, end.x), y: Math.max(start.y, end.y) });
    const cardIds = topLeft && bottomRight
      ? cards
        .filter((card) => (
          card.x + PERSON_CARD_WIDTH / 2 >= Math.min(topLeft.x, bottomRight.x)
          && card.x + PERSON_CARD_WIDTH / 2 <= Math.max(topLeft.x, bottomRight.x)
          && card.y >= Math.min(topLeft.y, bottomRight.y)
          && card.y <= Math.max(topLeft.y, bottomRight.y)
        ))
        .map((card) => card.id)
      : [];
    const localStart = getCanvasLocalPoint({ ...event, clientX: start.x, clientY: start.y } as PointerEvent<CanvasElement>);
    const localEnd = getCanvasLocalPoint(event);
    setAreaSelection({
      left: Math.min(localStart.x, localEnd.x),
      top: Math.min(localStart.y, localEnd.y),
      width: Math.abs(localEnd.x - localStart.x),
      height: Math.abs(localEnd.y - localStart.y),
      cardIds,
      worldBounds: topLeft && bottomRight ? { left: Math.min(topLeft.x, bottomRight.x), top: Math.min(topLeft.y, bottomRight.y), right: Math.max(topLeft.x, bottomRight.x), bottom: Math.max(topLeft.y, bottomRight.y) } : null,
    });
    selectionClickGuard.current = true;
    selectionDrag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }, [cards, getCanvasLocalPoint, onPanPointerUp, screenToWorld]);

  const clearAreaSelection = useCallback(() => setAreaSelection(null), []);
  const consumeSelectionClick = useCallback(() => {
    if (!selectionClickGuard.current) return false;
    selectionClickGuard.current = false;
    return true;
  }, []);

  return { areaSelection, handlePointerDown, handlePointerMove, handlePointerUp, clearAreaSelection, consumeSelectionClick };
}
