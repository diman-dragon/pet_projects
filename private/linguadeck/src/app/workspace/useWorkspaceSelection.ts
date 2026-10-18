import { useState } from "react";

export function useWorkspaceSelection() {
  const [selectedDeckId, setSelectedDeckId] = useState<number | null>(null);
  const [selectedCardIds, setSelectedCardIds] = useState<number[]>([]);

  function openDeckCards(deckId: number, onViewCards: () => void) {
    setSelectedDeckId(deckId);
    setSelectedCardIds([]);
    onViewCards();
  }

  function selectDeck(deckId: number | null) {
    setSelectedDeckId(deckId);
    setSelectedCardIds([]);
  }

  return {
    selectedDeckId,
    setSelectedDeckId,
    selectedCardIds,
    setSelectedCardIds,
    openDeckCards,
    selectDeck
  };
}
