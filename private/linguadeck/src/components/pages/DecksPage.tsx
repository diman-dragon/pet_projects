import { Layers, Plus } from "lucide-react";
import type { Deck, ImportProgress } from "../../types/domain";
import { DeckTile } from "../decks/DeckTile";
import { Button } from "../ui/Button";
import { StatCard } from "../ui/StatCard";
import { ImportProgressBanner } from "./decks/ImportProgressBanner";
import { LanguageGroup } from "./decks/LanguageGroup";
import { DecksToolbar } from "./decks/DecksToolbar";
import { useDecksPage } from "./hooks/useDecksPage";

interface DecksPageProps {
  decks: Deck[];
  search: string;
  importProgress?: ImportProgress;
  onChangeSearch: (value: string) => void;
  onCreateDeck: () => void;
  onOpenCards: (deckId: number) => void;
  onStudy: (deck: Deck) => void;
  onEditDeck: (deck: Deck) => void;
  onDeleteDeck: (deckId: number) => void;
}

export function DecksPage({
  decks,
  search,
  importProgress,
  onChangeSearch,
  onCreateDeck,
  onOpenCards,
  onStudy,
  onEditDeck,
  onDeleteDeck,
}: DecksPageProps) {
  const { totals, groups, expanded, toggleGroup, isSingleUngrouped } = useDecksPage(decks);

  return (
    <section className="space-y-6 p-2">
      <div className="py-6">
        <h1 className="text-2xl font-bold text-slate-900">Колоды</h1>
        <p className="text-sm text-slate-500">Управление колодами карточек для изучения языков</p>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <StatCard title="Всего колод" value={decks.length} subtitle="В библиотеке" accent />
        <StatCard title="Карточек" value={totals.cards} subtitle="Общее количество" />
        <StatCard title="Новые" value={totals.newCards} subtitle="Для изучения" />
        <StatCard title="На повтор" value={totals.dueCards} subtitle="Очередь сегодня" />
      </div>

      {importProgress && <ImportProgressBanner progress={importProgress} />}

      <DecksToolbar
        search={search}
        decksCount={decks.length}
        onChangeSearch={onChangeSearch}
        onCreateDeck={onCreateDeck}
      />

      {decks.length === 0 ? (
        <div className="text-center py-20 text-slate-400">
          <Layers size={48} className="mx-auto mb-4" />
          <h3 className="text-lg font-medium text-slate-900">{search ? "Ничего не найдено" : "Нет колод"}</h3>
          <p className="mb-4">{search ? "Попробуйте изменить запрос" : "Создайте первую колоду для начала работы"}</p>
          {!search && (
            <Button onClick={onCreateDeck} variant="primary">
              <Plus size={16} />
              Создать колоду
            </Button>
          )}
        </div>
      ) : isSingleUngrouped ? (
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
      ) : (
        <div className="space-y-3">
          {groups.map(({ lang, decks: groupDecks }) => (
            <LanguageGroup
              key={lang || "__none__"}
              lang={lang}
              decks={groupDecks}
              expanded={expanded.has(lang)}
              onToggle={() => toggleGroup(lang)}
              onOpenCards={onOpenCards}
              onStudy={onStudy}
              onEditDeck={onEditDeck}
              onDeleteDeck={onDeleteDeck}
            />
          ))}
        </div>
      )}
    </section>
  );
}
