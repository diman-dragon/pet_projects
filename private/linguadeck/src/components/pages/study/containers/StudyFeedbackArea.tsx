import { motion } from "framer-motion";

interface StudyFeedbackAreaProps {
  isSaving: boolean;
  audioError: string | null;
  onClearAudioError: () => void;
}

export function StudyFeedbackArea({ isSaving, audioError, onClearAudioError }: StudyFeedbackAreaProps) {
  return (
    <>
      {isSaving && (
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute top-16 right-6 z-50 bg-blue-600/90 backdrop-blur-sm text-white text-[10px] font-bold uppercase tracking-widest px-3 py-1.5 rounded-full shadow-lg"
        >
          Сохранение...
        </motion.div>
      )}

      {audioError && (
        <div className="mx-5 mb-2 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
          <span>🔇</span>
          <span className="flex-1">{audioError}</span>
          <button className="text-red-400 hover:text-red-600" onClick={onClearAudioError}>✕</button>
        </div>
      )}
    </>
  );
}
