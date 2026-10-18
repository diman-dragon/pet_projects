import { useMemo, useState } from "react";
import type { Deck } from "../../../types/domain";

export function useDecksPage(decks: Deck[]) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const totals = useMemo(() => {
    return decks.reduce(
      (acc, deck) => {
        acc.cards += deck.cards_count;
        acc.newCards += deck.new_count;
        acc.dueCards += deck.due_count;
        return acc;
      },
      { cards: 0, newCards: 0, dueCards: 0 }
    );
  }, [decks]);

  const groups = useMemo(() => {
    const map = new Map<string, Deck[]>();
    for (const deck of decks) {
      const key = deck.language?.trim() || "";
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(deck);
    }
    const entries = [...map.entries()].sort(([a], [b]) => {
      if (a === "" && b !== "") return 1;
      if (a !== "" && b === "") return -1;
      return a.localeCompare(b);
    });
    return entries.map(([lang, ds]) => ({ lang, decks: ds }));
  }, [decks]);

  function toggleGroup(lang: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(lang)) next.delete(lang);
      else next.add(lang);
      return next;
    });
  }

  const isSingleUngrouped = groups.length === 1 && groups[0].lang === "";

  return {
    totals,
    groups,
    expanded,
    toggleGroup,
    isSingleUngrouped,
  };
}
