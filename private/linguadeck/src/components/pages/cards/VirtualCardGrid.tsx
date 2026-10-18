import { memo, useEffect, useRef, useState } from "react";
import type { Card } from "../../../types/domain";
import { CardTile } from "../../cards/CardTile";

interface VirtualCardGridProps {
  cards: Card[];
  selectedSet: Set<number>;
  onToggle: (id: number) => void;
  onEdit: (card: Card) => void;
  onDelete: (id: number) => void;
}

const CARD_HEIGHT = 220;
const COLS = 3;
const OVERSCAN_ROWS = 3;

export const VirtualCardGrid = memo(function VirtualCardGrid({
  cards,
  selectedSet,
  onToggle,
  onEdit,
  onDelete,
}: VirtualCardGridProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(800);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewportHeight(el.clientHeight));
    ro.observe(el);
    setViewportHeight(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  const totalRows = Math.ceil(cards.length / COLS);
  const totalHeight = totalRows * CARD_HEIGHT;

  const firstRow = Math.max(0, Math.floor(scrollTop / CARD_HEIGHT) - OVERSCAN_ROWS);
  const visibleRows = Math.ceil(viewportHeight / CARD_HEIGHT) + OVERSCAN_ROWS * 2;
  const lastRow = Math.min(totalRows, firstRow + visibleRows);

  const visibleCards = cards.slice(firstRow * COLS, lastRow * COLS);

  return (
    <div
      ref={containerRef}
      className="overflow-auto"
      style={{ height: "calc(100vh - 340px)", minHeight: 400 }}
      onScroll={(e) => setScrollTop((e.target as HTMLDivElement).scrollTop)}
    >
      <div style={{ height: totalHeight, position: "relative" }}>
        <div
          style={{
            position: "absolute",
            top: firstRow * CARD_HEIGHT,
            left: 0,
            right: 0,
          }}
          className="grid gap-4 md:grid-cols-2 xl:grid-cols-3"
        >
          {visibleCards.map((card) => (
            <CardTile
              key={card.id}
              card={card}
              selected={selectedSet.has(card.id)}
              onToggle={() => onToggle(card.id)}
              onEdit={() => onEdit(card)}
              onDelete={() => onDelete(card.id)}
            />
          ))}
        </div>
      </div>
    </div>
  );
});
