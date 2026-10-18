import type { Card } from "../../../types/domain";
import { CardTile } from "../../cards/CardTile";
import { VirtualCardGrid } from "../cards/VirtualCardGrid";

interface CardsGridContainerProps {
  cards: Card[];
  selectedSet: Set<number>;
  useVirtual: boolean;
  onToggle: (id: number) => void;
  onEdit: (card: Card) => void;
  onDelete: (id: number) => void;
}

export function CardsGridContainer({ 
  cards, selectedSet, useVirtual, onToggle, onEdit, onDelete 
}: CardsGridContainerProps) {
  if (useVirtual) {
    return (
      <VirtualCardGrid
        cards={cards}
        selectedSet={selectedSet}
        onToggle={onToggle}
        onEdit={onEdit}
        onDelete={onDelete}
      />
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {cards.map((card) => (
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
  );
}
