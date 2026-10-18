import { GraduationCap, BookOpen, Layers, Star, ShieldCheck } from "lucide-react";
import type { DashboardStats } from "../../types/domain";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";

function CountUp({ value, decimals = 0 }: { value: number; decimals?: number }) {
  const [displayValue, setDisplayValue] = useState(0);
  
  useEffect(() => {
    let start = 0;
    const duration = 1000;
    const startTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOutExpo = 1 - Math.pow(2, -10 * progress);
      
      const current = start + (value - start) * easeOutExpo;
      setDisplayValue(current);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value]);

  return <span>{displayValue.toFixed(decimals)}</span>;
}

export function QualityMetrics({ stats }: { stats: DashboardStats }) {
  const easeColor = stats.averageEase >= 2.5 ? "text-blue-600" : stats.averageEase >= 2.0 ? "text-slate-600" : "text-rose-600";
  const easeBg = stats.averageEase >= 2.5 ? "bg-blue-600/10" : stats.averageEase >= 2.0 ? "bg-slate-600/10" : "bg-rose-600/10";
  const easeBar   = Math.min(100, ((stats.averageEase - 1.3) / (3.0 - 1.3)) * 100);

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      {/* Average Ease Metric */}
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-lg border border-white/40 bg-white/60 p-5 shadow-sm backdrop-blur-md"
      >
        <div className="flex items-center justify-between">
          <div className={`rounded-md ${easeBg} p-2.5`}>
            <Star className={easeColor} size={20} />
          </div>
          <div className="text-right">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Ease фактор</div>
            <div className={`text-2xl font-semibold tabular-nums ${easeColor}`}>
              <CountUp value={stats.averageEase} decimals={2} />
            </div>
          </div>
        </div>
        
        <div className="mt-5">
          <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${easeBar}%` }}
              className={`h-full ${easeColor.replace('text-', 'bg-')}`} 
            />
          </div>
          <div className="mt-2 flex justify-between text-[9px] font-bold text-slate-400">
            <span>1.3</span>
            <span>3.0</span>
          </div>
        </div>
      </motion.div>

      {/* Completion Rate Metric */}
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="relative overflow-hidden rounded-lg border border-white/40 bg-white/60 p-5 shadow-sm backdrop-blur-md"
      >
        <div className="flex items-center justify-between">
          <div className="rounded-md bg-blue-600/10 p-2.5 text-blue-600">
            <GraduationCap size={20} />
          </div>
          <div className="text-right">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Прогресс</div>
            <div className="text-2xl font-semibold tabular-nums text-slate-900">
              <CountUp value={stats.completionRate} />%
            </div>
          </div>
        </div>

        <div className="mt-5">
          <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${stats.completionRate}%` }}
              className="h-full bg-blue-600" 
            />
          </div>
          <div className="mt-2 text-[9px] font-bold text-slate-400">
            {stats.totalCards - stats.newCards} из {stats.totalCards} карточек начато
          </div>
        </div>
      </motion.div>

      {/* Stats Summary Metric */}
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="relative overflow-hidden rounded-lg border border-white/40 bg-white/60 p-5 shadow-sm backdrop-blur-md"
      >
        <div className="flex items-center justify-between">
          <div className="rounded-md bg-slate-600/10 p-2.5 text-slate-600">
            <ShieldCheck size={20} />
          </div>
          <div className="text-right">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Общий охват</div>
            <div className="text-2xl font-semibold tabular-nums text-slate-900">
              <CountUp value={stats.viewedAllTime} />
            </div>
          </div>
        </div>

        <div className="mt-5 flex flex-wrap gap-2">
          <div className="flex items-center gap-1 rounded-md bg-slate-50 px-2 py-1 text-[10px] font-semibold text-slate-600 ring-1 ring-inset ring-black/5">
            <Layers size={12} className="text-slate-400" />
            {stats.totalDecks} колод
          </div>
          <div className="flex items-center gap-1 rounded-md bg-slate-50 px-2 py-1 text-[10px] font-semibold text-slate-600 ring-1 ring-inset ring-black/5">
            <BookOpen size={12} className="text-slate-400" />
            {stats.totalCards} карточек
          </div>
        </div>
      </motion.div>
    </div>
  );
}
