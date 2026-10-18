import { ChevronDown, ChevronRight } from "lucide-react";
import type { Deck } from "../../../types/domain";
import { DeckTile } from "../../decks/DeckTile";
import { getLanguageOption } from "../../../app/languageConfig";

interface LanguageGroupProps {
  lang: string;
  decks: Deck[];
  expanded: boolean;
  onToggle: () => void;
  onOpenCards: (deckId: number) => void;
  onStudy: (deck: Deck) => void;
  onEditDeck: (deck: Deck) => void;
  onDeleteDeck: (deckId: number) => void;
}

export function LanguageGroup({
  lang,
  decks,
  expanded,
  onToggle,
  onOpenCards,
  onStudy,
  onEditDeck,
  onDeleteDeck,
}: LanguageGroupProps) {
  const opt = getLanguageOption(lang);
  const label = lang
    ? `${opt.flag ? opt.flag + " " : ""}${opt.label || lang}`
    : "Без языка";
  const accentColor = opt.color || "#9ca3af";
  const dueCount = decks.reduce((s, d) => s + d.due_count, 0);
  const newCount = decks.reduce((s, d) => s + d.new_count, 0);

  return (
    <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-3 px-5 py-4 text-left transition-colors hover:bg-gray-50"
      >
        <span
          className="h-3 w-3 shrink-0 rounded-full"
          style={{ backgroundColor: accentColor }}
        />
        <span className="flex-1 text-base font-semibold text-gray-800">{label}</span>
        <div className="flex items-center gap-2 text-xs">
          <span className="rounded-full bg-gray-100 px-2.5 py-1 font-medium text-gray-600">
            {decks.length} {decks.length === 1 ? "колода" : "колод"}
          </span>
          {dueCount > 0 && (
            <span className="rounded-full bg-amber-100 px-2.5 py-1 font-medium text-amber-700">
              {dueCount} на повтор
            </span>
          )}
          {newCount > 0 && (
            <span className="rounded-full bg-blue-100 px-2.5 py-1 font-medium text-blue-700">
              {newCount} новых
            </span>
          )}
        </div>
        {expanded ? (
          <ChevronDown size={18} className="shrink-0 text-gray-400" />
        ) : (
          <ChevronRight size={18} className="shrink-0 text-gray-400" />
        )}
      </button>

      {expanded && (
        <div className="border-t border-gray-100 px-5 pb-5 pt-4">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {decks.map((deck) => (
              <DeckTile
                key={deck.id}
                deck={deck}
                onDelete={() => onDeleteDeck(deck.id)}
                onEdit={() => onEditDeck(deck)}
                onOpenCards={() => onOpenCards(deck.id)}
                onStudy={() => onStudy(deck)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
