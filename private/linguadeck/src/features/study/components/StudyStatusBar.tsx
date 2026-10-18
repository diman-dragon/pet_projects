import type { Card } from "../../../types/domain";
import type { CardStats } from "../../../services/db/CardStatsService";
import { Info, Zap } from "lucide-react";

interface StudyStatusBarProps {
  card: Card;
  stats: CardStats | null;
  currentIndex: number;
  totalCards: number;
  onFinish: () => void;
}

export function StudyStatusBar({ card, stats, currentIndex, totalCards }: Omit<StudyStatusBarProps, "onFinish">) {
  const rating = stats?.recent_reviews?.[0]?.rating;
  const colors = ["#ef4444", "#f97316", "#3b82f6", "#10b981"];
  const bgColor = rating ? colors[rating - 1] : "rgba(30, 41, 59, 0.7)";

  return (
    <div className="flex items-center justify-between w-full mb-4 px-1">
      <div className="flex items-center gap-3">
        <div className="flex items-center bg-white/60 backdrop-blur-md border border-white/40 rounded-2xl px-3 py-1.5 shadow-sm">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider mr-2">Прогресс</span>
          <div className="flex items-center gap-1">
            <span className="text-sm font-bold text-slate-900">{currentIndex + 1}</span>
            <span className="text-xs text-slate-400">/</span>
            <span className="text-xs font-medium text-slate-500">{totalCards}</span>
          </div>
        </div>

        <div 
          className="flex items-center gap-2 rounded-2xl px-3 py-1.5 text-white shadow-sm transition-colors duration-300"
          style={{ backgroundColor: bgColor }}
        >
          <Zap size={12} className="fill-current" />
          <span className="text-[10px] font-bold uppercase tracking-widest">
            {card.status === "new" ? "Новая" : card.status === "learning" ? "Изучается" : "Повторение"}
          </span>
        </div>

        {rating && (
          <div className="flex items-center bg-white/60 backdrop-blur-md border border-white/40 rounded-2xl px-3 py-1.5 shadow-sm">
            <Info size={12} className="text-slate-400 mr-2" />
            <span className="text-[10px] font-bold text-slate-500 uppercase">Последняя оценка: {rating}</span>
          </div>
        )}
      </div>
    </div>
  );
}
