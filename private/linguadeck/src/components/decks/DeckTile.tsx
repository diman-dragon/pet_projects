import { BookOpen, CreditCard, Edit3, Play, Trash2 } from "lucide-react";
import type { Deck } from "../../types/domain";
import { Button } from "../ui/Button";
import { getLanguageOption } from "../../app/languageConfig";

interface DeckTileProps {
  deck: Deck;
  onOpenCards: () => void;
  onStudy: () => void;
  onEdit: () => void;
  onDelete: () => void;
}

/** Coloured pill for the deck language */
function LanguageBadge({ language }: { language: string }) {
  if (!language) return null;
  const opt = getLanguageOption(language);
  const bgColor = opt.color || "#6b7280";
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold text-white shadow-sm"
      style={{ backgroundColor: bgColor }}
      title={`Язык: ${opt.label || language}`}
    >
      {opt.flag && <span className="text-xs leading-none">{opt.flag}</span>}
      {opt.label || language}
    </span>
  );
}

export function DeckTile({ deck, onOpenCards, onStudy, onEdit, onDelete }: DeckTileProps) {
  return (
    <article className="group rounded-xl border border-gray-200 bg-white p-5 shadow-sm transition-all hover:shadow-md">
      {/* Top row: status badge + language + actions */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <div
            className="rounded-full px-3 py-1 text-xs font-semibold text-white"
            style={{ backgroundColor: deck.color || "#0078D4" }}
          >
            {deck.due_count > 0
              ? `${deck.due_count} на повтор`
              : deck.new_count > 0
              ? `${deck.new_count} новых`
              : "Готово"}
          </div>
          <LanguageBadge language={deck.language ?? ""} />
        </div>
        <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
          <button
            onClick={onEdit}
            className="rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
            title="Редактировать"
            type="button"
          >
            <Edit3 size={16} />
          </button>
          <button
            onClick={onDelete}
            className="rounded-lg p-1.5 text-gray-400 hover:bg-red-50 hover:text-red-500"
            title="Удалить"
            type="button"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      <div className="mt-4">
        <h3 className="truncate text-lg font-bold text-gray-900">{deck.name}</h3>
        <div className="mt-2 flex items-center gap-3 text-sm text-gray-500">
          <span className="flex items-center gap-1">
            <CreditCard size={14} />
            {deck.cards_count}
          </span>
          <span className="h-4 w-px bg-gray-200" />
          <span className={deck.new_count > 0 ? "font-medium text-green-600" : ""}>
            {deck.new_count} новых
          </span>
          <span className="h-4 w-px bg-gray-200" />
          <span className={deck.due_count > 0 ? "font-medium text-amber-600" : ""}>
            {deck.due_count} повтор
          </span>
        </div>
      </div>

      <p className="mt-3 line-clamp-2 text-sm text-gray-500">
        {deck.description || "Нет описания"}
      </p>

      <div className="mt-5 flex gap-2">
        <Button className="flex-1" onClick={onOpenCards}>
          <BookOpen size={16} />
          Карточки
        </Button>
        <Button className="flex-1" onClick={onStudy} variant="primary">
          <Play size={16} />
          Учить
        </Button>
      </div>
    </article>
  );
}
