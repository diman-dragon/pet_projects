import { useRef } from "react";
import type { KeyboardEvent, MouseEvent, PointerEvent } from "react";
import { PersonCard } from "@/components/PersonCard/PersonCard";
import { useCanvasCardsStore } from "@/store/useCanvasCardsStore";
import type { CanvasCardData } from "@/types/canvasCard";
import type { CanvasViewport } from "@/types/canvasViewport";
import type { ContextMenuPosition } from "@/types/contextMenu";
import { useCanvasCardDrag } from "@/components/Canvas/useCanvasCardDrag";
import "@/components/Canvas/CanvasCard.css";

type CanvasCardProps = {
  card: CanvasCardData;
  viewport: CanvasViewport;
  onDeleteRequest: (cardId: string) => void;
  onEditRequest: (cardId: string) => void;
  onContextMenu: (cardId: string, position: ContextMenuPosition) => void;
  locked?: boolean;
  showAuxiliaryLines?: boolean;
  onMoveFinished: (cardId: string, movedAsGroup: boolean) => Promise<void>;
  onSelectionToggle: (cardId: string, clientX: number, clientY: number) => void;
};

export function CanvasCard({ card, viewport, onDeleteRequest, onEditRequest, onContextMenu, onMoveFinished, onSelectionToggle, locked = card.locked, showAuxiliaryLines = true }: CanvasCardProps) {
  const selectCard = useCanvasCardsStore((state) => state.selectCard);
  const selectedCardId = useCanvasCardsStore((state) => state.selectedCardId);
  const selectedCardIds = useCanvasCardsStore((state) => state.selectedCardIds);
  const personName = [card.person.firstName, card.person.patronymic, card.person.lastName].filter(Boolean).join(" ");
  const multiSelectPointer = useRef(false);
  const { handlePointerDown: handleDragPointerDown, handlePointerMove, handlePointerUp, didMove } = useCanvasCardDrag({ card, viewport, locked: locked ?? false, onMoveFinished, onSelectionToggle });

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    multiSelectPointer.current = event.ctrlKey || event.metaKey;
    handleDragPointerDown(event);
  }

  function handleClick(event: MouseEvent<HTMLDivElement>) {
    event.stopPropagation();
    if (multiSelectPointer.current) {
      multiSelectPointer.current = false;
      return;
    }
    if (didMove.current) {
      didMove.current = false;
      return;
    }
    selectCard(card.id);
  }

  function handleContextMenu(event: MouseEvent<HTMLDivElement>) {
    event.preventDefault();
    event.stopPropagation();
    onContextMenu(card.id, { x: event.clientX, y: event.clientY });
  }

  function handleDoubleClick(event: MouseEvent<HTMLDivElement>) {
    event.preventDefault(); event.stopPropagation(); onEditRequest(card.id);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectCard(card.id);
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      onDeleteRequest(card.id);
    }
  }

  return (
    <div
      className={`canvas-card${locked ? " canvas-card--locked" : ""}${showAuxiliaryLines ? "" : " canvas-card--auxiliary-hidden"}`}
      tabIndex={0}
      role="button"
      aria-label={personName ? `Карточка человека ${personName}` : `Карточка человека ${card.id}`}
      style={{ left: card.x, top: card.y }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onContextMenu={handleContextMenu}
      onKeyDown={handleKeyDown}
    >
      <PersonCard
        person={card.person}
        selected={selectedCardIds.includes(card.id) || selectedCardId === card.id}
      />
    </div>
  );
}
