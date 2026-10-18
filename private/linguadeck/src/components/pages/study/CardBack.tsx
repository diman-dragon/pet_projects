import { motion } from "framer-motion";
import { Zap, BarChart2 } from "lucide-react";
import type { Card } from "../../../types/domain";

interface CardBackProps {
  card: Card;
  rating?: number;
  flipCount: number;
}

const STATUS_LABELS: Record<string, string> = {
  new: "Новая",
  learning: "Изучается",
  review: "На повторении",
};

export function CardBack({ card, rating, flipCount }: CardBackProps) {
  const statusLabel = STATUS_LABELS[card.status] ?? card.status;
  const ratingColors = ["text-rose-600 border-rose-100 bg-rose-50", "text-orange-600 border-orange-100 bg-orange-50", "text-blue-600 border-blue-100 bg-blue-50", "text-emerald-600 border-emerald-100 bg-emerald-50"];
  const ratingColorClass = rating ? ratingColors[rating - 1] : "";

  return (
    <div
      className="absolute inset-0 flex flex-col rounded-3xl overflow-hidden"
      style={{
        backfaceVisibility: "hidden",
        transform: "rotateY(180deg)",
        background: "linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%)",
        boxShadow: "0 32px 80px -16px rgba(15,23,42,0.18), 0 4px 20px -4px rgba(16,185,129,0.18)",
      }}
    >
      <div className="absolute inset-x-0 top-0 h-1.5 bg-gradient-to-r from-emerald-400 via-teal-500 to-green-400" />

      <div className="flex flex-1 flex-col px-12 py-10">
        {/* BUG FIX: key includes flipCount so animation re-triggers on every flip */}
        <motion.div
          key={`back-${card.id}-${flipCount}`}
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.18, duration: 0.38, ease: "easeOut" }}
          className="flex flex-1 flex-col items-center justify-center text-center"
        >
          <h2 className="text-6xl font-black tracking-tight text-gray-900 leading-none break-words">
            {card.back}
          </h2>
          {card.context && (
            <p className="mt-5 max-w-[85%] text-xl leading-relaxed text-gray-500">
              {card.context}
            </p>
          )}
        </motion.div>

        {card.example_1 && (
          <motion.div
            key={`ex-${card.id}-${flipCount}`}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.32, ease: "easeOut" }}
            className="rounded-2xl bg-emerald-50/80 border border-emerald-100 p-5 backdrop-blur-sm mb-4"
          >
            <p className="mb-3 text-[10px] font-bold uppercase tracking-widest text-emerald-600">Примеры</p>
            <div className="space-y-2">
              {[card.example_1, card.example_2, card.example_3]
                .filter(Boolean)
                .slice(0, 3)
                .map((ex, i) => (
                  <p key={i} className="text-sm leading-relaxed text-gray-700">
                    <span className="font-semibold text-emerald-500 mr-1">{i + 1}.</span>
                    {ex}
                  </p>
                ))}
            </div>
          </motion.div>
        )}
      </div>

      {/* Footer Area with Status & Rating */}
      <div className="flex items-center justify-between border-t border-emerald-50 px-8 py-5">
        <div className="flex flex-col gap-1.5">
          <div className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 border ${
            card.status === "new" ? "bg-emerald-50 border-emerald-100 text-emerald-600"
              : card.status === "learning" ? "bg-amber-50 border-amber-100 text-amber-600"
                : "bg-blue-50 border-blue-100 text-[#0078D4]"
          }`}>
            <Zap size={11} className="fill-current" />
            <span className="text-[10px] font-bold uppercase tracking-widest">{statusLabel}</span>
          </div>

          {rating && (
            <div className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 border ${ratingColorClass}`}>
              <BarChart2 size={11} />
              <span className="text-[10px] font-bold uppercase tracking-widest">Оценка: {rating}</span>
            </div>
          )}
        </div>

        <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-600/60">
          Обратная сторона
        </span>
      </div>
    </div>
  );
}
