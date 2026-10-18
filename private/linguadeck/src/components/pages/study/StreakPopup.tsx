import { motion, AnimatePresence } from "framer-motion";
import { Flame, Star, TrendingUp, Clock, X } from "lucide-react";
import type { WeekActivityDay } from "../../../types/domain";

interface StreakPopupProps {
  open: boolean;
  streak: number;
  sessionCount: number;  // cards reviewed in this session
  reviewedToday: number;
  totalTimeMs: number;     // суммарное время за сегодня (view + flip)
  weekActivity: WeekActivityDay[];
  onClose: () => void;
}

function fmtTime(ms: number): string {
  if (ms < 1000) return "< 1 сек";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s} сек`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return rem > 0 ? `${m} мин ${rem} сек` : `${m} мин`;
}

function StreakFlame({ streak }: { streak: number }) {
  const color = streak >= 30 ? "text-red-500" : streak >= 7 ? "text-orange-500" : "text-amber-400";
  const suffix = streak === 1 ? "день подряд" : streak < 5 ? "дня подряд" : "дней подряд";
  return (
    <motion.div
      initial={{ scale: 0.4, rotate: -15 }}
      animate={{ scale: 1, rotate: 0 }}
      transition={{ type: "spring", stiffness: 280, damping: 18 }}
      className="flex flex-col items-center"
    >
      <div className={`relative ${color}`}>
        <Flame size={72} strokeWidth={1.5} className="drop-shadow-lg" />
        <motion.div
          animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
          className="absolute inset-0 flex items-center justify-center"
        >
          <Flame size={72} strokeWidth={1.5} className="opacity-30 blur-[4px]" />
        </motion.div>
      </div>
      <span className="text-5xl font-black text-gray-900 mt-1 tabular-nums">{streak}</span>
      <span className="text-sm font-semibold text-gray-500 mt-0.5">{suffix}</span>
    </motion.div>
  );
}

function WeekChart({ days }: { days: WeekActivityDay[] }) {
  const max = Math.max(...days.map(d => d.count), 1);
  return (
    <div className="w-full">
      <div className="flex items-center gap-1.5 mb-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">
        <TrendingUp size={13} />Активность за неделю
      </div>
      <div className="flex items-end justify-between gap-1.5 h-20">
        {days.map((day, i) => (
          <div key={i} className="flex flex-col items-center gap-1 flex-1">
            {day.count > 0 && (
              <span className="text-[9px] font-bold text-gray-400 tabular-nums">{day.count}</span>
            )}
            <motion.div
              title={`${day.label}: ${day.count} карт, ${fmtTime(day.timeMs)}`}
              className={`w-full rounded-md ${
                day.isToday
                  ? "bg-gradient-to-t from-[#0078D4] to-sky-400 shadow-md shadow-blue-300/50"
                  : day.hasActivity
                    ? "bg-gradient-to-t from-blue-400 to-blue-200"
                    : "bg-gray-100"
              }`}
              initial={{ height: 0 }}
              animate={{ height: `${Math.max((day.count / max) * 100, day.hasActivity ? 20 : 12)}%` }}
              transition={{ delay: i * 0.06, duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
            />
            <span className={`text-[10px] font-bold ${day.isToday ? "text-[#0078D4]" : "text-gray-400"}`}>
              {day.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function StreakPopup({ open, streak, sessionCount, reviewedToday, totalTimeMs, weekActivity, onClose }: StreakPopupProps) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="streak-overlay"
          className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" />
          <motion.div
            className="relative w-full max-w-sm rounded-3xl bg-white shadow-2xl overflow-hidden"
            initial={{ opacity: 0, scale: 0.85, y: 40 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ type: "spring", stiffness: 300, damping: 24 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="h-1.5 bg-gradient-to-r from-amber-400 via-orange-500 to-rose-500" />
            <button type="button" onClick={onClose}
              className="absolute top-4 right-4 flex h-8 w-8 items-center justify-center rounded-xl text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-all z-10">
              <X size={16} />
            </button>

            <div className="px-8 pt-7 pb-8 flex flex-col items-center gap-5">
              <div className="text-center">
                <div className="flex items-center justify-center gap-1.5 mb-1">
                  <Star size={13} className="text-amber-400 fill-amber-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-amber-600">Сессия завершена!</span>
                  <Star size={13} className="text-amber-400 fill-amber-400" />
                </div>
              </div>

              <StreakFlame streak={streak} />

              {/* Stats row */}
              <div className="grid grid-cols-3 gap-3 w-full">
                <div className="rounded-2xl bg-blue-50 border border-blue-100 p-3 text-center">
                  <div className="text-2xl font-black text-[#0078D4] tabular-nums">{sessionCount}</div>
                  <div className="text-[11px] text-blue-600 font-semibold mt-0.5">за сессию</div>
                </div>
                <div className="rounded-2xl bg-indigo-50 border border-indigo-100 p-3 text-center">
                  <div className="text-2xl font-black text-indigo-700 tabular-nums">{reviewedToday}</div>
                  <div className="text-[11px] text-indigo-600 font-semibold mt-0.5">всего сегодня</div>
                </div>
                <div className="rounded-2xl bg-emerald-50 border border-emerald-100 p-3 text-center">
                  <div className="flex items-center justify-center gap-1">
                    <Clock size={14} className="text-emerald-600" />
                    <span className="text-lg font-black text-emerald-700">{fmtTime(totalTimeMs)}</span>
                  </div>
                  <div className="text-[11px] text-emerald-600 font-semibold mt-0.5">время</div>
                </div>
              </div>

              {weekActivity.length > 0 && (
                <div className="w-full rounded-2xl bg-gray-50 border border-gray-100 p-4">
                  <WeekChart days={weekActivity} />
                </div>
              )}

              <motion.button
                type="button"
                onClick={onClose}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                className="w-full rounded-2xl bg-gradient-to-r from-[#0078D4] to-sky-500 py-3.5 text-sm font-bold text-white shadow-lg shadow-blue-400/30 hover:shadow-xl transition-all"
              >
                Продолжить 🚀
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
