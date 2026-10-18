import { BarChart2, TrendingUp, Zap, RefreshCw, AlertCircle, Eye, Star, RotateCcw, X } from "lucide-react";
import { motion } from "framer-motion";
import type { Card } from "../../../types/domain";
import type { CardStats } from "../../../services/db/CardStatsService";

const STATUS_LABELS: Record<string, string> = { new: "Новая", learning: "Изучается", review: "На повторении" };
const RATING_LABELS: Record<number, { label: string; color: string }> = {
  1: { label: "Снова", color: "text-rose-500" },
  2: { label: "Сложно", color: "text-orange-500" },
  3: { label: "Хорошо", color: "text-blue-500" },
  4: { label: "Легко", color: "text-emerald-500" },
};

function fmt(str: string | null) {
  if (!str) return "—";
  return new Date(str).toLocaleDateString("ru-RU", { day: "2-digit", month: "short", year: "numeric" });
}
// function fmtTime(ms: number): string {
//   if (!ms || ms <= 0) return "—";
//   const s = Math.floor(ms / 1000);
//   if (s < 60) return `${s} сек`;
//   return `${Math.floor(s / 60)} мин ${s % 60} сек`;
// }
function fmtDays(days: number) {
  if (days < 1) return "< 1 дня";
  if (days < 30) return `${Math.round(days)} дн.`;
  if (days < 365) return `${Math.round(days / 30)} мес.`;
  return `${(days / 365).toFixed(1)} г.`;
}

interface StatCellProps { label: string; value: string | number; icon: React.ReactNode; }
function StatCell({ label, value, icon }: StatCellProps) {
  return (
    <div className="bg-white px-4 py-3 flex flex-col gap-0.5">
      <div className="flex items-center gap-1 text-[10px] font-medium uppercase tracking-wide text-gray-400">
        <span className="text-gray-300">{icon}</span>{label}
      </div>
      <div className="text-sm font-bold text-gray-900">{value}</div>
    </div>
  );
}

interface CardStatsPanelProps { card: Card; stats: CardStats | null; loading: boolean; onClose: () => void; }

export function CardStatsPanel({ card, stats, loading, onClose }: CardStatsPanelProps) {
  return (
    <motion.div
      key="stats"
      initial={{ opacity: 0, y: 12, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 12, scale: 0.97 }}
      transition={{ duration: 0.22 }}
      className="w-full h-full overflow-auto"
    >
      <div className="rounded-3xl bg-white shadow-xl shadow-gray-200/60 overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-900">
            <BarChart2 size={16} className="text-[#0078D4]" />Статистика карточки
          </div>
          <button type="button" onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-all">
            <X size={14} />
          </button>
        </div>
        <div className="px-6 pt-4 pb-2">
          <div className="mb-2 text-base font-bold text-gray-900 truncate">{card.front}</div>
          <div className="flex flex-wrap gap-2 text-xs">
            <span className={`rounded-full px-2 py-0.5 font-medium ${card.status === "new" ? "bg-emerald-100 text-emerald-700"
                : card.status === "learning" ? "bg-amber-100 text-amber-700"
                  : "bg-blue-100 text-blue-700"
              }`}>{STATUS_LABELS[card.status] ?? card.status}</span>
            {card.tags && <span className="rounded-full bg-slate-100 px-2 py-0.5 text-slate-600">{card.tags}</span>}
          </div>
        </div>
        <div className="grid grid-cols-3 gap-px bg-gray-100 border-t border-gray-100">
          <StatCell label="Интервал" value={fmtDays(card.interval_days)} icon={<TrendingUp size={13} />} />
          <StatCell label="Ease" value={card.ease_factor.toFixed(2)} icon={<Zap size={13} />} />
          <StatCell label="Повторений" value={card.repetitions} icon={<RefreshCw size={13} />} />
          <StatCell label="Ошибок" value={card.lapses} icon={<AlertCircle size={13} />} />
          <StatCell label="Последний" value={fmt(card.last_reviewed_at)} icon={<Eye size={13} />} />
          <StatCell label="Следующий" value={fmt(card.due_at)} icon={<Star size={13} />} />
        </div>
        {loading ? (
          <div className="px-6 py-4 text-center text-sm text-gray-400">Загрузка...</div>
        ) : stats ? (
          <>
            <div className="grid grid-cols-3 gap-px bg-gray-100 border-t border-gray-100">
              <StatCell label="Просмотрено" value={stats.views_total} icon={<Eye size={13} />} />
              <StatCell label="Перевёрнуто" value={stats.flips_total} icon={<RotateCcw size={13} />} />
              <StatCell label="Оценено" value={stats.reviews_total} icon={<BarChart2 size={13} />} />
            </div>
            {stats.reviews_total > 0 && (
              <div className="px-6 py-4 border-t border-gray-100">
                <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">Последние оценки</div>
                <div className="flex flex-wrap gap-1.5">
                  {stats.recent_reviews.map((r, i) => {
                    const rl = RATING_LABELS[r.rating] ?? { label: String(r.rating), color: "text-gray-500" };
                    return (
                      <div key={i} className="flex flex-col items-center gap-0.5 rounded-xl border border-gray-100 bg-slate-50 px-2.5 py-1.5 min-w-[56px]">
                        <span className={`text-sm font-bold ${rl.color}`}>{rl.label}</span>
                        <span className="text-[10px] text-gray-400">{new Date(r.reviewed_at).toLocaleDateString("ru-RU", { day: "2-digit", month: "short" })}</span>
                        <span className="text-[10px] text-gray-400">→ {fmtDays(r.interval_after)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        ) : null}
      </div>
    </motion.div>
  );
}
