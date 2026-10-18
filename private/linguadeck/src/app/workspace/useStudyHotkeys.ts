import { useEffect } from "react";

interface StudyHotkeysProps {
  enabled: boolean;
  flipped: boolean;
  total: number;
  onNext: () => void;
  onFlip: () => void;
  onPrev: () => void;
  onManualWrong?: () => void;
}

export function useStudyHotkeys({
  enabled,
  flipped,
  total,
  onNext,
  onFlip,
  onPrev,
  onManualWrong,
}: StudyHotkeysProps) {
  useEffect(() => {
    if (!enabled) return;

    function handleKeyDown(e: KeyboardEvent) {
      if (e.repeat) return;
      
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;

      if (e.code === "Space") {
        e.preventDefault();
        onFlip();
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        onNext();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        onPrev();
      } else if (e.code === "Backspace") {
        e.preventDefault();
        onManualWrong?.();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [enabled, flipped, total, onNext, onFlip, onPrev, onManualWrong]);
}
