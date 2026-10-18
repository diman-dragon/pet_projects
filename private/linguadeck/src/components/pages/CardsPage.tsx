import { Plus, Search } from "lucide-react";
import type { Card, Deck } from "../../types/domain";
import { Button } from "../ui/Button";
import { StatCard } from "../ui/StatCard";
import { CardsToolbar } from "./cards/CardsToolbar";
import { CardsGridContainer } from "./cards/CardsGridContainer";
import { useCardsPage } from "./hooks/useCardsPage";

interface CardsPageProps {
  decks: Deck[];
  selectedDeck: Deck | null;
  selectedDeckId: number | null;
  cards: Card[];
  selectedCardIds: number[];
  search: string;
  onSelectDeck: (deckId: number | null) => void;
  onChangeSearch: (value: string) => void;
  onToggleCard: (cardId: number) => void;
  onCreateCard: () => void;
  onEditCard: (card: Card) => void;
  onDeleteCard: (cardId: number) => void;
  onDeleteSelected: () => void;
  onImport: () => void;
}

export function CardsPage(props: CardsPageProps) {
  const hasDeck = Boolean(props.selectedDeckId);
  const deck = props.selectedDeck;
  const cards = props.cards;

  const { selectedSet, handleToggle, handleEdit, handleDelete } = useCardsPage({
    selectedCardIds: props.selectedCardIds,
    onToggleCard: props.onToggleCard,
    onEditCard: props.onEditCard,
    onDeleteCard: props.onDeleteCard,
  });

  return (
    <section className="space-y-6 p-2">
      <div className="py-6">
        <h1 className="text-2xl font-bold text-gray-900">Карточки</h1>
        <p className="text-sm text-gray-500">
          {deck
            ? `Управление карточками колоды "${deck.name}"`
            : "Выберите колоду для управления карточками"}
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <StatCard
          title="Колода"
          value={deck?.name || "—"}
          subtitle={deck ? `${deck.cards_count} карточек` : "Не выбрана"}
          accent={Boolean(deck)}
        />
        <StatCard title="Всего" value={deck?.cards_count ?? 0} subtitle="В колоде" />
        <StatCard title="Новые" value={deck?.new_count ?? 0} subtitle="Для изучения" />
        <StatCard title="На повтор" value={deck?.due_count ?? 0} subtitle="Очередь" />
      </div>

      <CardsToolbar
        decks={props.decks}
        selectedDeckId={props.selectedDeckId}
        cardsCount={cards.length}
        search={props.search}
        selectedCardIds={props.selectedCardIds}
        onSelectDeck={props.onSelectDeck}
        onChangeSearch={props.onChangeSearch}
        onDeleteSelected={props.onDeleteSelected}
        onImport={props.onImport}
        onCreateCard={props.onCreateCard}
      />

      {!hasDeck ? (
        <div className="text-center py-20 text-gray-400">
          <Plus size={48} className="mx-auto mb-4" />
          <h3 className="text-lg font-medium">Выберите колоду</h3>
          <p>Для просмотра карточек выберите колоду из списка выше</p>
        </div>
      ) : cards.length === 0 ? (
        <div className="text-center py-20 text-gray-400">
          <Search size={48} className="mx-auto mb-4" />
          <h3 className="text-lg font-medium">{props.search ? "Ничего не найдено" : "Нет карточек"}</h3>
          <p className="mb-4">{props.search ? "Попробуйте изменить запрос" : "Создайте первую карточку в этой колоде"}</p>
          {!props.search && (
            <Button onClick={props.onCreateCard} variant="primary">
              <Plus size={16} />
              Создать карточку
            </Button>
          )}
        </div>
      ) : (
        <CardsGridContainer
          cards={cards}
          selectedSet={selectedSet}
          useVirtual={cards.length > 60}
          onToggle={handleToggle}
          onEdit={handleEdit}
          onDelete={handleDelete}
        />
      )}
    </section>
  );
}
