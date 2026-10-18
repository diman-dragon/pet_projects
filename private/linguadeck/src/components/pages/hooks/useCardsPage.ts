import { useCallback, useMemo } from "react";
import type { Card } from "../../../types/domain";

interface UseCardsPageProps {
  selectedCardIds: number[];
  onToggleCard: (cardId: number) => void;
  onEditCard: (card: Card) => void;
  onDeleteCard: (cardId: number) => void;
}

export function useCardsPage({
  selectedCardIds,
  onToggleCard,
  onEditCard,
  onDeleteCard,
}: UseCardsPageProps) {
  const selectedSet = useMemo(() => new Set(selectedCardIds), [selectedCardIds]);

  const handleToggle = useCallback((id: number) => onToggleCard(id), [onToggleCard]);
  const handleEdit = useCallback((card: Card) => onEditCard(card), [onEditCard]);
  const handleDelete = useCallback((id: number) => onDeleteCard(id), [onDeleteCard]);

  return {
    selectedSet,
    handleToggle,
    handleEdit,
    handleDelete,
  };
}
