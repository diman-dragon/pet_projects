import { CheckCircle2, TrendingUp, Sparkles } from "lucide-react";
import type { DashboardStats } from "../../types/domain";
import { motion } from "framer-motion";

const GRADE = (r: number) =>
  r >= 90 ? { label: "Элита", color: "text-emerald-400", bg: "bg-emerald-500/20", ring: "ring-emerald-500/30", icon: Sparkles } :
  r >= 75 ? { label: "Мастер", color: "text-blue-400",    bg: "bg-blue-500/20",    ring: "ring-blue-500/30", icon: CheckCircle2 } :
  r >= 55 ? { label: "Профи",  color: "text-amber-400",   bg: "bg-amber-500/20",   ring: "ring-amber-500/30", icon: TrendingUp } :
            { label: "Новичок", color: "text-rose-400",    bg: "bg-rose-500/20",    ring: "ring-rose-500/30", icon: TrendingUp };

export function RetentionHero({ stats }: { stats: DashboardStats }) {
  const g = GRADE(stats.retentionRounded);
  const Icon = g.icon;

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="relative col-span-1 overflow-hidden rounded-[2.5rem] bg-slate-950 p-8 text-white shadow-2xl shadow-blue-500/20"
    >
      {/* Dynamic Background Effects */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -left-[10%] -top-[10%] h-[120%] w-[120%] opacity-20 bg-[radial-gradient(circle_at_30%_30%,#3b82f6_0%,transparent_50%),radial-gradient(circle_at_70%_70%,#8b5cf6_0%,transparent_50%)]" />
        <motion.div 
          animate={{ 
            rotate: [0, 360],
            scale: [1, 1.2, 1],
          }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-blue-500/10 blur-[80px]" 
        />
        <motion.div 
          animate={{ 
            x: [-20, 20, -20],
            y: [-20, 20, -20],
          }}
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
          className="absolute -bottom-20 -left-20 h-64 w-64 rounded-full bg-indigo-500/10 blur-[80px]" 
        />
      </div>

      <div className="relative z-10">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2.5 text-blue-300">
            <div className="rounded-lg bg-blue-500/20 p-1.5 ring-1 ring-blue-500/30">
               <TrendingUp size={16} />
            </div>
            <span className="text-[10px] font-black uppercase tracking-[0.25em]">Retention 30d</span>
          </div>
          <motion.div 
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 4, repeat: Infinity }}
            className={`flex items-center gap-1.5 rounded-full px-4 py-1.5 text-[10px] font-black uppercase tracking-widest ring-1 ${g.bg} ${g.color} ${g.ring} backdrop-blur-md`}
          >
            <Icon size={12} /> {g.label}
          </motion.div>
        </div>

        <div className="flex items-baseline gap-2">
          <motion.span 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-7xl font-black tracking-tighter tabular-nums text-transparent bg-clip-text bg-gradient-to-b from-white to-white/50"
          >
            {stats.retentionRounded}
          </motion.span>
          <span className="text-3xl font-black text-blue-400/50">%</span>
        </div>

        <p className="mt-4 max-w-[200px] text-sm font-bold leading-relaxed text-slate-400">
          Вероятность успешного вспоминания через месяц
        </p>

        <div className="mt-8 flex items-center gap-4">
          <div className="h-[2px] flex-1 rounded-full bg-white/5">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${stats.retentionRounded}%` }}
              transition={{ duration: 1.5, ease: "circOut" }}
              className="h-full rounded-full bg-gradient-to-r from-blue-500 to-indigo-400 shadow-[0_0_20px_rgba(59,130,246,0.5)]" 
            />
          </div>
          <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Goal 90%</div>
        </div>
      </div>
    </motion.div>
  );
}
