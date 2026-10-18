import { BarChart2, Pencil, LogOut } from "lucide-react";
import { motion } from "framer-motion";

interface StudyHeaderProps {
  deckName: string;
  index: number;
  total: number;
  statsOpen: boolean;
  onToggleStats: () => void;
  onEdit?: () => void;
  onFinish?: () => void;
}

export function StudyHeader({
  deckName, index, total, statsOpen, onToggleStats, onEdit, onFinish,
}: StudyHeaderProps) {
  const progress = total > 0 ? ((index + 1) / total) * 100 : 0;

  return (
    <div className="shrink-0 px-5 pt-4 pb-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-[#0078D4] to-[#005A9E] text-white shadow-md">
            <span className="text-sm font-bold">{deckName.charAt(0).toUpperCase()}</span>
          </div>
          <div>
            <p className="text-[11px] text-gray-500 uppercase tracking-wide">{deckName}</p>
            <p className="text-sm font-bold text-gray-900">Изучение</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2">
            {onEdit && (
              <button
                type="button"
                onClick={onEdit}
                title="Редактировать карточку"
                className="flex h-9 w-9 items-center justify-center rounded-xl bg-white text-gray-500 shadow-sm transition-all hover:bg-amber-50 hover:text-amber-600"
              >
                <Pencil size={15} />
              </button>
            )}

            {onFinish && (
              <button
                type="button"
                onClick={onFinish}
                title="Завершить сессию"
                className="flex h-9 items-center gap-1.5 rounded-xl bg-white px-3 text-gray-500 shadow-sm transition-all hover:bg-rose-50 hover:text-rose-600 text-xs font-semibold border border-transparent hover:border-rose-100"
              >
                <LogOut size={14} />
                Закончить
              </button>
            )}

            <button
              type="button"
              onClick={onToggleStats}
              title="Статистика карточки"
              className={`flex h-9 w-9 items-center justify-center rounded-xl transition-all ${
                statsOpen
                  ? "bg-[#0078D4] text-white shadow-md"
                  : "bg-white text-gray-500 shadow-sm hover:bg-gray-50 hover:text-[#0078D4]"
              }`}
            >
              <BarChart2 size={16} />
            </button>

            <div className="flex items-center gap-2 rounded-xl bg-white px-3 py-1.5 shadow-sm border border-slate-50">
              <span className="text-lg font-bold text-[#0078D4]">{index + 1}</span>
              <span className="text-sm text-gray-400">/</span>
              <span className="text-sm text-gray-600">{total}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="h-2 w-full rounded-full bg-gray-200/80">
        <motion.div
          className="h-full rounded-full bg-gradient-to-r from-[#0078D4] via-blue-500 to-sky-400"
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.35, ease: "easeOut" }}
          style={{ boxShadow: "0 0 8px rgba(0,120,212,0.4)" }}
        />
      </div>
    </div>
  );
}
