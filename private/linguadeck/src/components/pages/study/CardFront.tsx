import { Loader2, Video, Volume2, Zap, BarChart2 } from "lucide-react";
import { motion } from "framer-motion";
import type { Card } from "../../../types/domain";

const STATUS_LABELS: Record<string, string> = {
  new: "Новая",
  learning: "Изучается",
  review: "На повторении",
};

interface CardFrontProps {
  card: Card;
  rating?: number;
  audioAvailable: boolean;
  videoAvailable: boolean;
  audioPlaying: boolean;
  onPlayAudio: () => void;
  onPlayVideo: () => void;
}

export function CardFront({ card, rating, audioAvailable, videoAvailable, audioPlaying, onPlayAudio, onPlayVideo }: CardFrontProps) {
  const statusLabel = STATUS_LABELS[card.status] ?? card.status;
  const ratingColors = ["text-rose-600 border-rose-100 bg-rose-50", "text-orange-600 border-orange-100 bg-orange-50", "text-blue-600 border-blue-100 bg-blue-50", "text-emerald-600 border-emerald-100 bg-emerald-50"];
  const ratingColorClass = rating ? ratingColors[rating - 1] : "";
  
  return (
    <div
      className="absolute inset-0 flex flex-col rounded-3xl bg-white overflow-hidden"
      style={{
        backfaceVisibility: "hidden",
        boxShadow: "0 32px 80px -16px rgba(15,23,42,0.18), 0 4px 20px -4px rgba(0,120,212,0.14)",
      }}
    >
      <div className="absolute inset-x-0 top-0 h-1.5 bg-gradient-to-r from-blue-500 via-[#0078D4] to-sky-400" />

      <div className="flex flex-1 flex-col items-center justify-center px-12 py-12 text-center">
        <motion.div
          key={`front-${card.id}`}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08, duration: 0.35, ease: "easeOut" }}
          className="flex flex-col items-center gap-5"
        >
          <h2 className="text-7xl font-black tracking-tight text-gray-900 leading-none break-words">
            {card.front}
          </h2>

          {card.transcription && (
            <p className="text-2xl font-medium text-gray-400 tracking-wide">
              {card.transcription}
            </p>
          )}
        </motion.div>
      </div>

      <div className="flex items-center justify-between border-t border-gray-100 px-8 py-5">
        <div className="flex items-center gap-3">
          {/* Audio button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (audioAvailable && !audioPlaying) onPlayAudio();
            }}
            disabled={!audioAvailable || audioPlaying}
            type="button"
            className={`relative flex h-11 w-11 items-center justify-center rounded-xl transition-all ${
              audioPlaying
                ? "bg-blue-200 text-[#0078D4] cursor-default"
                : audioAvailable
                  ? "bg-blue-100 text-[#0078D4] hover:bg-blue-200 hover:shadow-md active:scale-95 cursor-pointer"
                  : "bg-gray-100 text-gray-300 cursor-not-allowed"
            }`}
          >
            {audioPlaying ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Volume2 size={20} />
            )}
          </button>

          {/* Status & Rating Stack */}
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
        </div>

        <motion.span
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ repeat: Infinity, duration: 2.4, ease: "easeInOut" }}
          className="flex items-center gap-1.5 text-xs font-medium text-gray-400"
        >
          Space или клик — перевернуть
        </motion.span>

        {/* Video button */}
        <button
          onClick={(e) => { e.stopPropagation(); if (videoAvailable) onPlayVideo(); }}
          disabled={!videoAvailable}
          type="button"
          className={`flex h-11 w-11 items-center justify-center rounded-xl transition-all ${
            videoAvailable
              ? "bg-purple-100 text-purple-600 hover:bg-purple-200 hover:shadow-md active:scale-95 cursor-pointer"
              : "bg-gray-100 text-gray-300 cursor-not-allowed"
          }`}
        >
          <Video size={20} />
        </button>
      </div>
    </div>
  );
}
