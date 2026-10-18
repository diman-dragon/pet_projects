import { useEffect, useState } from "react";
import { Clock, Eye, Layers3, RotateCcw } from "lucide-react";
import { StudyService } from "../../services/db/StudyService";
import type { AppSettings } from "../../types/domain";
import { AlgoSummary } from "./AlgoSummary";
import { formatDuration, formatDateKey } from "./calendarUtils";

interface ViewStatsProps {
  selectedDate?: string;
  settings: AppSettings;
}

interface RawDayStats {
  views: number;
  unique_cards: number;
  flips: number;
  time_ms: number;
}

function isToday(dateStr: string): boolean {
  return dateStr === formatDateKey(new Date());
}

export function ViewStats({ selectedDate, settings }: ViewStatsProps) {
  const [stats, setStats] = useState<RawDayStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const date = selectedDate || formatDateKey(new Date());
    setLoading(true);
    setStats(null);
    StudyService.getDayStats(date)
      .then((s) => setStats(s as unknown as RawDayStats))
      .finally(() => setLoading(false));
  }, [selectedDate]);

  if (loading) {
    return (
      <div className="grid grid-cols-2 gap-3">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-24 animate-pulse rounded-xl bg-gray-100" />
        ))}
      </div>
    );
  }

  const date = selectedDate || formatDateKey(new Date());
  const views = stats?.views ?? 0;
  const uniqueCards = stats?.unique_cards ?? 0;
  const flips = stats?.flips ?? 0;
  const timeMs = stats?.time_ms ?? 0;
  const flipRate = views > 0 ? Math.round((flips / views) * 100) : 0;
  const timeLabel = isToday(date) ? "всего сегодня" : "за этот день";

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <StatCard
          icon={<Eye size={18} />}
          label="Показы"
          value={views}
          sub="карточек"
          color="blue"
        />
        <StatCard
          icon={<Layers3 size={18} />}
          label="Уникальные"
          value={uniqueCards}
          sub="карточек"
          color="teal"
        />
        <StatCard
          icon={<RotateCcw size={18} />}
          label="Перевороты"
          value={flips}
          sub={`${flipRate}% от показов`}
          color="amber"
        />
        <StatCard
          icon={<Clock size={18} />}
          label="Время"
          value={formatDuration(timeMs)}
          sub={timeLabel}
          color="slate"
        />
      </div>

      <AlgoSummary settings={settings} />
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  sub?: string;
  color: "blue" | "teal" | "amber" | "slate";
}) {
  const colors = {
    blue: "from-blue-50 to-white ring-blue-100 text-blue-600",
    teal: "from-teal-50 to-white ring-teal-100 text-teal-600",
    amber: "from-amber-50 to-white ring-amber-100 text-amber-600",
    slate: "from-slate-50 to-white ring-slate-200 text-slate-600",
  };
  return (
    <div className={`flex flex-col gap-2 rounded-xl border border-gray-100 bg-gradient-to-br p-4 shadow-sm ring-1 ${colors[color]}`}>
      <div className="flex items-center gap-2">
        <div className="opacity-80">{icon}</div>
        <div className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">{label}</div>
      </div>
      <div className="mt-1 text-xl font-black text-gray-800">{value}</div>
      {sub && <div className="text-[10px] font-medium text-gray-400">{sub}</div>}
    </div>
  );
}
