import { Plus, Trash2, Upload } from "lucide-react";
import type { Deck } from "../../../types/domain";
import { Button } from "../../ui/Button";
import { Checkbox } from "../../ui/Checkbox";
import { SearchField } from "../../ui/SearchField";

interface CardsToolbarProps {
  decks: Deck[];
  selectedDeckId: number | null;
  cardsCount: number;
  search: string;
  selectedCardIds: number[];
  onSelectDeck: (deckId: number | null) => void;
  onChangeSearch: (value: string) => void;
  onDeleteSelected: () => void;
  onImport: () => void;
  onCreateCard: () => void;
}

export function CardsToolbar({
  decks,
  selectedDeckId,
  cardsCount,
  search,
  selectedCardIds,
  onSelectDeck,
  onChangeSearch,
  onDeleteSelected,
  onImport,
  onCreateCard,
}: CardsToolbarProps) {
  const hasDeck = Boolean(selectedDeckId);

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <select
        className="rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm outline-none focus:border-[#0078D4] focus:ring-2 focus:ring-[#0078D4]/20 min-w-[180px]"
        value={selectedDeckId ? String(selectedDeckId) : ""}
        onChange={(e) => onSelectDeck(e.target.value ? Number(e.target.value) : null)}
      >
        <option value="">Выберите колоду...</option>
        {decks.map((d) => (
          <option key={d.id} value={String(d.id)}>
            {d.name} ({d.cards_count})
          </option>
        ))}
      </select>

      <div className="h-6 w-px bg-gray-200" />

      <SearchField
        value={search}
        onChange={onChangeSearch}
        placeholder="Поиск по слову, переводу..."
        className="flex-1 min-w-[200px]"
      />

      <div className="h-6 w-px bg-gray-200" />

      <div className="text-sm text-gray-500">
        {cardsCount.toLocaleString()} карточек
      </div>

      {selectedCardIds.length > 0 && (
        <>
          <div className="h-6 w-px bg-gray-200" />
          <div className="flex items-center gap-2">
            <Checkbox
              checked
              onChange={() => undefined}
              label={`${selectedCardIds.length} выбрано`}
            />
            <button
              onClick={onDeleteSelected}
              className="rounded-lg bg-red-50 px-3 py-1.5 text-sm font-medium text-red-600 hover:bg-red-100"
              type="button"
            >
              <Trash2 size={14} className="inline mr-1" />
              Удалить
            </button>
          </div>
        </>
      )}

      <div className="ml-auto flex items-center gap-2">
        <Button disabled={!hasDeck} onClick={onImport}>
          <Upload size={16} />
          Импорт
        </Button>
        <Button disabled={!hasDeck} onClick={onCreateCard} variant="primary">
          <Plus size={16} />
          Новая
        </Button>
      </div>
    </div>
  );
}
