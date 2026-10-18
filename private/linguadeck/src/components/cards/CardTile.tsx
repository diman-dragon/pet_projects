import { memo } from "react";
import { Edit3, Hash, Image, Trash2, Video, Volume2 } from "lucide-react";
import type { Card } from "../../types/domain";
import { Checkbox } from "../ui/Checkbox";

interface CardTileProps {
  card: Card;
  selected: boolean;
  onToggle: () => void;
  onEdit: () => void;
  onDelete: () => void;
}

const STATUS_CONFIG: Record<string, { badge: string; accent: string; dot: string }> = {
  new:      { badge: "bg-emerald-100 text-emerald-700", accent: "border-l-emerald-400", dot: "bg-emerald-400" },
  learning: { badge: "bg-amber-100 text-amber-700",     accent: "border-l-amber-400",   dot: "bg-amber-400"   },
  review:   { badge: "bg-blue-100 text-blue-700",       accent: "border-l-blue-400",    dot: "bg-blue-400"    },
};

const STATUS_LABELS: Record<string, string> = {
  new:      "Новая",
  learning: "Изучение",
  review:   "Повтор",
};

// Pure function — no object allocation on every render
function statusText(card: Card) {
  const base = STATUS_LABELS[card.status] ?? card.status;
  return card.status !== "new" ? `${base} · ${card.interval_days}д` : base;
}

export const CardTile = memo(function CardTile({ card, selected, onToggle, onEdit, onDelete }: CardTileProps) {
  const cfg      = STATUS_CONFIG[card.status] ?? STATUS_CONFIG.review;
  const hasMedia = card.audio_path || card.video_path || card.image_path;

  return (
    <article
      className={`group relative rounded-2xl border-l-4 bg-white p-5 shadow-sm transition-all duration-200 hover:shadow-lg hover:scale-[1.01] ${cfg.accent} ${
        selected
          ? "border-t border-r border-b border-[#0078D4]/30 bg-[#E5F1FB]/25"
          : "border-t border-r border-b border-gray-100"
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-4">
        <Checkbox checked={selected} onChange={onToggle} />
        <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
          <button
            onClick={onEdit}
            className="rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-700 transition-colors"
            title="Редактировать" type="button"
          >
            <Edit3 size={16} />
          </button>
          <button
            onClick={onDelete}
            className="rounded-lg p-1.5 text-gray-400 hover:bg-red-50 hover:text-red-500 transition-colors"
            title="Удалить" type="button"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      <div className="space-y-2 mb-3">
        <div className="text-xl font-bold text-gray-900 leading-tight">{card.front}</div>
        <div className="text-base text-gray-600 leading-snug">{card.back}</div>
        {card.transcription && (
          <div className="text-sm text-gray-400 font-medium">{card.transcription}</div>
        )}
      </div>

      {card.example_1 && (
        <div className="mb-4 rounded-xl bg-gray-50 px-3 py-2.5 text-sm italic text-gray-500 leading-relaxed border border-gray-100">
          "{card.example_1}"
        </div>
      )}

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold ${cfg.badge}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${cfg.dot}`} />
            {statusText(card)}
          </span>
          {card.tags && (
            <span className="flex items-center gap-1 text-xs text-gray-400">
              <Hash size={12} />
              {card.tags}
            </span>
          )}
        </div>
        {hasMedia && (
          <div className="flex items-center gap-1.5">
            {card.audio_path && <Volume2 size={14} className="text-blue-400" />}
            {card.image_path && <Image   size={14} className="text-purple-400" />}
            {card.video_path && <Video   size={14} className="text-rose-400" />}
          </div>
        )}
      </div>
    </article>
  );
},
// Custom comparator — skip re-render if nothing meaningful changed
(prev, next) =>
  prev.selected === next.selected &&
  prev.card.id === next.card.id &&
  prev.card.status === next.card.status &&
  prev.card.interval_days === next.card.interval_days &&
  prev.card.front === next.card.front &&
  prev.card.back === next.card.back
);
