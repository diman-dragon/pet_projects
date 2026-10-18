import { motion, AnimatePresence } from "framer-motion";
import { Star, TrendingUp, Clock, X } from "lucide-react";
import type { WeekActivityDay } from "../../../types/domain";

interface StreakPopupProps {
  open: boolean;
  streak: number;
  sessionCount: number;
  reviewedToday: number;
  totalTimeMs: number;
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

function RunningPerson({ streak }: { streak: number }) {
  const color =
    streak >= 30 ? "#ef4444" : streak >= 7 ? "#f97316" : streak >= 1 ? "#3b82f6" : "#94a3b8";
  const light =
    streak >= 30 ? "#fca5a5" : streak >= 7 ? "#fdba74" : streak >= 1 ? "#93c5fd" : "#cbd5e1";
  const suffix =
    streak === 1 ? "день подряд" : streak < 5 ? "дня подряд" : "дней подряд";

  return (
    <motion.div
      initial={{ scale: 0.5, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 260, damping: 18 }}
      className="flex flex-col items-center"
    >
      <svg viewBox="0 0 120 145" width="96" height="116" aria-hidden>
        <style>{`
          @keyframes ld-bob     { 0%,100%{transform:translateY(0)}  50%{transform:translateY(-4px)} }
          @keyframes ld-arm-l   { 0%,100%{transform:rotate(-50deg)} 50%{transform:rotate(35deg)}  }
          @keyframes ld-arm-r   { 0%,100%{transform:rotate(35deg)}  50%{transform:rotate(-50deg)} }
          @keyframes ld-thigh-l { 0%,100%{transform:rotate(-42deg)} 50%{transform:rotate(38deg)}  }
          @keyframes ld-thigh-r { 0%,100%{transform:rotate(38deg)}  50%{transform:rotate(-42deg)} }
          @keyframes ld-shin-l  { 0%,100%{transform:rotate(10deg)}  50%{transform:rotate(50deg)}  }
          @keyframes ld-shin-r  { 0%,100%{transform:rotate(50deg)}  50%{transform:rotate(10deg)}  }
          @keyframes ld-shadow  { 0%,100%{transform:scaleX(0.8) translateX(-2px)} 50%{transform:scaleX(1.1) translateX(2px)} }

          .ld-body   { animation: ld-bob     0.42s ease-in-out infinite; }
          .ld-arm-l  { transform-origin: 60px 50px; animation: ld-arm-l   0.42s ease-in-out infinite; }
          .ld-arm-r  { transform-origin: 60px 50px; animation: ld-arm-r   0.42s ease-in-out infinite; }
          .ld-thi-l  { transform-origin: 60px 83px; animation: ld-thigh-l 0.42s ease-in-out infinite; }
          .ld-thi-r  { transform-origin: 60px 83px; animation: ld-thigh-r 0.42s ease-in-out infinite; }
          .ld-shin-l { transform-origin: 43px 105px; animation: ld-shin-l 0.42s ease-in-out infinite; }
          .ld-shin-r { transform-origin: 77px 105px; animation: ld-shin-r 0.42s ease-in-out infinite; }
          .ld-shadow { transform-origin: 60px 140px; animation: ld-shadow 0.42s ease-in-out infinite; }
        `}</style>

        {/* Shadow */}
        <ellipse className="ld-shadow" cx="60" cy="140" rx="18" ry="4" fill={light} opacity="0.4" />

        <g className="ld-body">
          {/* Head */}
          <circle cx="60" cy="18" r="16" fill={color} />
          {/* Helmet visor shine */}
          <ellipse cx="55" cy="12" rx="5" ry="4" fill="white" opacity="0.18" />
          {/* Eyes */}
          <circle cx="54" cy="17" r="3" fill="white" />
          <circle cx="66" cy="17" r="3" fill="white" />
          <circle cx="55" cy="17.5" r="1.5" fill={color} />
          <circle cx="67" cy="17.5" r="1.5" fill={color} />
          {/* Smile */}
          <path
            d="M 54 23 Q 60 28 66 23"
            stroke="white" strokeWidth="2" fill="none" strokeLinecap="round"
          />

          {/* Torso */}
          <rect x="49" y="34" width="22" height="30" rx="7" fill={color} />
          {/* Chest stripe */}
          <rect x="55" y="37" width="10" height="4" rx="2" fill={light} opacity="0.6" />

          {/* Shorts */}
          <rect x="47" y="60" width="26" height="16" rx="5" fill={color} opacity="0.85" />

          {/* Left arm */}
          <g className="ld-arm-l">
            <rect x="36" y="34" width="11" height="26" rx="5.5" fill={light} />
            {/* Glove */}
            <circle cx="41.5" cy="60" r="6" fill={color} />
          </g>

          {/* Right arm */}
          <g className="ld-arm-r">
            <rect x="73" y="34" width="11" height="26" rx="5.5" fill={light} />
            <circle cx="78.5" cy="60" r="6" fill={color} />
          </g>

          {/* Left thigh */}
          <g className="ld-thi-l">
            <rect x="49" y="74" width="14" height="28" rx="7" fill={light} />

            {/* Left shin — nested, pivots at knee */}
            <g className="ld-shin-l">
              <rect x="43" y="99" width="12" height="24" rx="6" fill={light} opacity="0.85" />
              {/* Shoe */}
              <ellipse cx="48" cy="124" rx="9" ry="5" fill={color} />
              <ellipse cx="53" cy="124" rx="4" ry="4" fill={color} />
            </g>
          </g>

          {/* Right thigh */}
          <g className="ld-thi-r">
            <rect x="57" y="74" width="14" height="28" rx="7" fill={light} />

            {/* Right shin */}
            <g className="ld-shin-r">
              <rect x="65" y="99" width="12" height="24" rx="6" fill={light} opacity="0.85" />
              {/* Shoe */}
              <ellipse cx="72" cy="124" rx="9" ry="5" fill={color} />
              <ellipse cx="77" cy="124" rx="4" ry="4" fill={color} />
            </g>
          </g>
        </g>

        {/* Ground dots — track */}
        {[20, 35, 50, 65, 80, 95, 110].map((x, i) => (
          <circle key={i} cx={x} cy="140" r="2.5" fill={light} opacity="0.5" />
        ))}
      </svg>

      <span className="mt-3 text-5xl font-black tabular-nums" style={{ color }}>
        {streak}
      </span>
      <span className="mt-0.5 text-sm font-semibold text-gray-500">{suffix}</span>
    </motion.div>
  );
}

function WeekChart({ days }: { days: WeekActivityDay[] }) {
  const max = Math.max(...days.map((d) => d.count), 1);
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
              animate={{
                height: `${Math.max((day.count / max) * 100, day.hasActivity ? 20 : 12)}%`,
              }}
              transition={{ delay: i * 0.06, duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
            />
            <span
              className={`text-[10px] font-bold ${
                day.isToday ? "text-[#0078D4]" : "text-gray-400"
              }`}
            >
              {day.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function StreakPopup({
  open,
  streak,
  sessionCount,
  reviewedToday,
  totalTimeMs,
  weekActivity,
  onClose,
}: StreakPopupProps) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="streak-overlay"
          className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
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

            <button
              type="button"
              onClick={onClose}
              className="absolute top-4 right-4 flex h-8 w-8 items-center justify-center rounded-xl text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-all z-10"
            >
              <X size={16} />
            </button>

            <div className="px-8 pt-7 pb-8 flex flex-col items-center gap-5">
              {/* Title */}
              <div className="text-center">
                <div className="flex items-center justify-center gap-1.5 mb-1">
                  <Star size={13} className="text-amber-400 fill-amber-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-amber-600">
                    Сессия завершена!
                  </span>
                  <Star size={13} className="text-amber-400 fill-amber-400" />
                </div>
              </div>

              {/* Running person */}
              <RunningPerson streak={streak} />

              {/* Stats row */}
              <div className="grid grid-cols-3 gap-3 w-full">
                <div className="rounded-2xl bg-blue-50 border border-blue-100 p-3 text-center">
                  <div className="text-2xl font-black text-[#0078D4] tabular-nums">
                    {sessionCount}
                  </div>
                  <div className="text-[11px] text-blue-600 font-semibold mt-0.5">за сессию</div>
                </div>
                <div className="rounded-2xl bg-indigo-50 border border-indigo-100 p-3 text-center">
                  <div className="text-2xl font-black text-indigo-700 tabular-nums">
                    {reviewedToday}
                  </div>
                  <div className="text-[11px] text-indigo-600 font-semibold mt-0.5">
                    всего сегодня
                  </div>
                </div>
                <div className="rounded-2xl bg-emerald-50 border border-emerald-100 p-3 text-center">
                  <div className="flex items-center justify-center gap-1">
                    <Clock size={14} className="text-emerald-600" />
                    <span className="text-lg font-black text-emerald-700">
                      {fmtTime(totalTimeMs)}
                    </span>
                  </div>
                  <div className="text-[11px] text-emerald-600 font-semibold mt-0.5">время</div>
                </div>
              </div>

              {/* Week chart */}
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
