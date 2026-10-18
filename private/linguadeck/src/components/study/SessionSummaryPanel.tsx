import { Trophy, Target, Flame, ArrowRight, BookOpen } from "lucide-react";
import { motion } from "framer-motion";
import type { SessionSummary } from "../../types/domain";

interface SessionSummaryPanelProps {
  summary: SessionSummary;
  deckName: string;
  onContinue: () => void;
}

export function SessionSummaryPanel({ summary, deckName, onContinue }: SessionSummaryPanelProps) {
  const getMessage = () => {
    if (summary.reviewedCount >= 50) return "Невероятная работа! 🔥";
    if (summary.reviewedCount >= 20) return "Отличная сессия! Вы молодец 💪";
    if (summary.reviewedCount >= 10) return "Хорошая работа! Продолжайте 📈";
    if (summary.reviewedCount >= 1) return "Начало положено! Продолжайте 🚀";
    return "Продолжайте заниматься регулярно!";
  };

  return (
    <div className="flex flex-1 flex-col items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 0, scale: 0.92, y: 16 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 28 }}
        className="w-full max-w-md rounded-2xl bg-white p-8 shadow-2xl ring-1 ring-gray-100"
      >
        {/* Icon */}
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.15, type: "spring", stiffness: 400, damping: 20 }}
          className="mb-5 text-center"
        >
          <div className="mx-auto mb-3 flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-amber-400 to-orange-500 shadow-lg shadow-amber-200">
            <Trophy className="h-10 w-10 text-white drop-shadow" />
          </div>
          <h2 className="text-2xl font-black text-gray-900">Сессия завершена!</h2>
          <p className="mt-1 text-sm font-medium text-gray-500">{deckName}</p>
        </motion.div>

        {/* Stats grid */}
        <div className="mb-5 grid grid-cols-2 gap-3">
          <motion.div
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-100 p-4 text-center ring-1 ring-blue-100"
          >
            <div className="mb-1 flex items-center justify-center gap-1 text-blue-400">
              <BookOpen size={14} />
            </div>
            <div className="text-3xl font-black text-blue-600">{summary.reviewedCount}</div>
            <div className="text-xs font-medium text-blue-500/80">карточек изучено</div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.25 }}
            className="rounded-2xl bg-gradient-to-br from-orange-50 to-amber-100 p-4 text-center ring-1 ring-orange-100"
          >
            <div className="mb-1 flex items-center justify-center gap-1 text-orange-400">
              <Flame size={14} />
            </div>
            <div className="text-3xl font-black text-orange-500">{summary.streak}</div>
            <div className="text-xs font-medium text-orange-500/80">дней подряд</div>
          </motion.div>
        </div>

        {/* Message */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="mb-5 flex items-center justify-center gap-2 rounded-xl bg-emerald-50 px-4 py-3 ring-1 ring-emerald-100"
        >
          <Target className="h-4 w-4 text-emerald-500 shrink-0" />
          <span className="text-sm font-medium text-emerald-700">{getMessage()}</span>
        </motion.div>

        {/* Continue button */}
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          onClick={onContinue}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-500 to-blue-600 py-3.5 font-semibold text-white shadow-md shadow-blue-200 transition-all hover:shadow-lg hover:from-blue-600 hover:to-blue-700 active:scale-[0.98]"
        >
          <span>Продолжить</span>
          <ArrowRight className="h-4 w-4" />
        </motion.button>
      </motion.div>
    </div>
  );
}
