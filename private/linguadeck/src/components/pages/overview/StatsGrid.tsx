import { motion } from "framer-motion";
import { Zap, Target, RefreshCw, Clock } from "lucide-react";
import { formatDuration } from "../../overview/calendarUtils";
import type { DashboardStats } from "../../../types/domain";

interface StatsGridProps {
  stats: DashboardStats;
}

export function StatsGrid({ stats }: StatsGridProps) {
  const shownToday = stats.shownToday ?? 0;
  const uniqueToday = stats.uniqueCardsToday ?? 0;
  const flipsToday = stats.shownFlippedToday ?? 0;
  const timeToday = stats.timeTodayMs ?? 0;

  const items = [
    { label: "Показы", value: shownToday, sub: "всего сегодня", icon: Zap, color: "text-blue-600", bg: "bg-blue-600/10" },
    { label: "Уникальные", value: uniqueToday, sub: "карточки", icon: Target, color: "text-slate-600", bg: "bg-slate-600/10" },
    { label: "Перевороты", value: flipsToday, sub: "вторая сторона", icon: RefreshCw, color: "text-slate-600", bg: "bg-slate-600/10" },
    { label: "Время", value: formatDuration(timeToday), sub: "обучения сегодня", icon: Clock, color: "text-blue-600", bg: "bg-blue-600/10" },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {items.map((item, i) => (
        <motion.div
          key={item.label}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05 }}
          className="group relative overflow-hidden rounded-lg border border-white/40 bg-white/60 p-5 shadow-sm backdrop-blur-md transition-all hover:bg-white/80"
        >
          <div className="flex items-start justify-between">
            <div className={`flex h-10 w-10 items-center justify-center rounded-md ${item.bg} ring-1 ring-inset ring-black/5`}>
              <item.icon size={20} className={item.color} />
            </div>
            <div className="text-right">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">{item.label}</div>
              <div className="mt-0.5 text-2xl font-semibold text-slate-900 tabular-nums">
                {item.value}
              </div>
            </div>
          </div>
          <div className="mt-4 flex items-center justify-between">
             <span className="text-[10px] font-medium text-slate-400">{item.sub}</span>
             <div className="h-1 w-12 rounded-full bg-slate-100 overflow-hidden">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  className={`h-full ${item.color.replace('text-', 'bg-')}`} 
                />
             </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

