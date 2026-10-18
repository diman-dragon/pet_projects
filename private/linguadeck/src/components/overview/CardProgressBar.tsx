import { Target } from "lucide-react";
import type { DashboardStats } from "../../types/domain";
import { DistLegend } from "./DistLegend";

export function CardProgressBar({ stats }: { stats: DashboardStats }) {
  const t = stats.totalCards || 1;
  const reviewOnly = stats.reviewCards - stats.masteredCards;
  return (
    <div className="rounded-2xl border border-gray-100 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-sm font-bold text-gray-900">
          <Target size={16} className="text-[#0078D4]" />
          Прогресс по базе
        </h3>
        <span className="text-sm font-bold text-[#0078D4]">{stats.completionRate}%</span>
      </div>
      <div className="relative h-3 w-full overflow-hidden rounded-full bg-gray-100">
        <div className="absolute left-0 top-0 h-full rounded-full bg-purple-500 transition-all" style={{ width: `${(stats.masteredCards / t) * 100}%` }} />
        <div className="absolute top-0 h-full rounded-full bg-blue-400 transition-all" style={{ left: `${(stats.masteredCards / t) * 100}%`, width: `${(reviewOnly / t) * 100}%` }} />
        <div className="absolute top-0 h-full rounded-full bg-amber-400 transition-all" style={{ left: `${(stats.reviewCards / t) * 100}%`, width: `${(stats.learningCards / t) * 100}%` }} />
      </div>
      <div className="mt-3 grid grid-cols-4 gap-2">
        <DistLegend color="bg-gray-200" label="Новые"       value={stats.newCards}      pct={stats.newCardsPct} />
        <DistLegend color="bg-amber-400" label="Изучаются"  value={stats.learningCards} pct={stats.learningCardsPct} />
        <DistLegend color="bg-blue-400"  label="Повторение" value={reviewOnly}           pct={stats.reviewOnlyPct} />
        <DistLegend color="bg-purple-500" label="Освоено"   value={stats.masteredCards} pct={stats.masteredCardsPct} />
      </div>
    </div>
  );
}
