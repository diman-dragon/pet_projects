import { useRef, useState } from "react";
import type { PointerEvent, RefObject, WheelEvent } from "react";
import type { CanvasPoint, CanvasViewport } from "@/types/canvasViewport";
import type { CardsBounds } from "@/lib/canvas/getCardsBounds";

const MIN_ZOOM = 0.05;
const MAX_ZOOM = 20;
const ZOOM_STEP = 1.2;

type UseCanvasViewportResult = {
  canvasRef: RefObject<HTMLDivElement | null>;
  viewport: CanvasViewport;
  transform: string;
  consumePanClick: () => boolean;
  handlePointerDown: (event: PointerEvent<HTMLDivElement>) => void;
  handlePointerMove: (event: PointerEvent<HTMLDivElement>) => void;
  handlePointerUp: (event: PointerEvent<HTMLDivElement>) => void;
  handleWheel: (event: WheelEvent<HTMLDivElement>) => void;
  centerView: (worldCenter: CanvasPoint) => void;
  fitViewToBounds: (bounds: CardsBounds | null) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  zoomTo100: () => void;
  screenToWorld: (point: CanvasPoint) => CanvasPoint | null;
};

export function useCanvasViewport(): UseCanvasViewportResult {
  const canvasRef = useRef<HTMLDivElement>(null);
  const panStart = useRef<CanvasPoint | null>(null);
  const panAtStart = useRef<CanvasPoint>({ x: 0, y: 0 });
  const didPan = useRef(false);
  const [viewport, setViewport] = useState<CanvasViewport>({
    pan: { x: 0, y: 0 },
    zoom: 1,
  });

  function screenToWorld(point: CanvasPoint): CanvasPoint | null {
    const bounds = canvasRef.current?.getBoundingClientRect();
    if (!bounds) return null;

    return {
      x:
        (point.x - (bounds.left + bounds.width / 2) - viewport.pan.x) /
        viewport.zoom,
      y:
        (point.y - (bounds.top + bounds.height / 2) - viewport.pan.y) /
        viewport.zoom,
    };
  }

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || event.target !== event.currentTarget) return;

    panStart.current = { x: event.clientX, y: event.clientY };
    panAtStart.current = viewport.pan;
    didPan.current = false;
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    if (!panStart.current) return;

    const dx = event.clientX - panStart.current.x;
    const dy = event.clientY - panStart.current.y;
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) didPan.current = true;

    setViewport((current) => ({
      ...current,
      pan: {
        x: panAtStart.current.x + dx,
        y: panAtStart.current.y + dy,
      },
    }));
  }

  function handlePointerUp(event: PointerEvent<HTMLDivElement>) {
    panStart.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function handleWheel(event: WheelEvent<HTMLDivElement>) {
    event.preventDefault();
    const direction = event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
    const nextZoom = Math.min(
      MAX_ZOOM,
      Math.max(MIN_ZOOM, viewport.zoom * direction),
    );
    if (nextZoom === viewport.zoom) return;

    const bounds = canvasRef.current?.getBoundingClientRect();
    if (!bounds) return;

    const cursor = {
      x: event.clientX - (bounds.left + bounds.width / 2),
      y: event.clientY - (bounds.top + bounds.height / 2),
    };
    setViewport({
      zoom: nextZoom,
      pan: {
        x: cursor.x - (cursor.x - viewport.pan.x) * (nextZoom / viewport.zoom),
        y: cursor.y - (cursor.y - viewport.pan.y) * (nextZoom / viewport.zoom),
      },
    });
  }

  function changeZoom(factor: number) {
    setViewport((current) => ({
      ...current,
      zoom: Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current.zoom * factor)),
    }));
  }

  function zoomTo100() {
    setViewport((current) => ({
      zoom: 1,
      pan: { x: current.pan.x / current.zoom, y: current.pan.y / current.zoom },
    }));
  }

  function centerView(worldCenter: CanvasPoint) {
    setViewport((current) => ({
      ...current,
      pan: { x: -worldCenter.x * current.zoom, y: -worldCenter.y * current.zoom },
    }));
  }

  function fitViewToBounds(bounds: CardsBounds | null) {
    const canvasBounds = canvasRef.current?.getBoundingClientRect();
    if (!canvasBounds || !bounds) return;
    const width = Math.max(1, bounds.right - bounds.left);
    const height = Math.max(1, bounds.bottom - bounds.top);
    const padding = 56;
    const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Math.min((canvasBounds.width - padding * 2) / width, (canvasBounds.height - padding * 2) / height)));
    const center = { x: (bounds.left + bounds.right) / 2, y: (bounds.top + bounds.bottom) / 2 };
    setViewport({ zoom, pan: { x: -center.x * zoom, y: -center.y * zoom } });
  }

  function consumePanClick() {
    if (!didPan.current) return false;
    didPan.current = false;
    return true;
  }

  return {
    canvasRef,
    viewport,
    transform: `translate(${viewport.pan.x}px, ${viewport.pan.y}px) scale(${viewport.zoom})`,
    consumePanClick,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    handleWheel,
    centerView,
    fitViewToBounds,
    zoomIn: () => changeZoom(ZOOM_STEP),
    zoomOut: () => changeZoom(1 / ZOOM_STEP),
    zoomTo100,
    screenToWorld,
  };
}
