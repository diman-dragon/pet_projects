import { useEffect, useMemo, useRef, useState } from "react";
import type { Card, Deck } from "../../types/domain";

/** Debounce a value by `ms` milliseconds */
function useDebounced<T>(value: T, ms = 150): T {
  const [debounced, setDebounced] = useState(value);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setDebounced(value), ms);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [value, ms]);

  return debounced;
}

export function useWorkspaceFilters(decks: Deck[], cards: Card[]) {
  const [deckSearch, setDeckSearch] = useState("");
  const [cardSearch, setCardSearch] = useState("");

  // Debounce so filtering doesn't run on every keystroke with 1000+ cards
  const debouncedCardSearch = useDebounced(cardSearch, 150);
  const debouncedDeckSearch = useDebounced(deckSearch, 100);

  const visibleDecks = useMemo(() => {
    const query = debouncedDeckSearch.trim().toLowerCase();
    if (!query) return decks;
    return decks.filter(
      (d) =>
        d.name.toLowerCase().includes(query) ||
        (d.description ?? "").toLowerCase().includes(query)
    );
  }, [decks, debouncedDeckSearch]);

  const visibleCards = useMemo(() => {
    const query = debouncedCardSearch.trim().toLowerCase();
    if (!query) return cards;
    return cards.filter((card) =>
      (card.front + " " + card.back + " " + (card.context ?? "") + " " + (card.tags ?? ""))
        .toLowerCase()
        .includes(query)
    );
  }, [cards, debouncedCardSearch]);

  return {
    deckSearch,
    setDeckSearch,
    cardSearch,
    setCardSearch,
    visibleDecks,
    visibleCards,
  };
}
