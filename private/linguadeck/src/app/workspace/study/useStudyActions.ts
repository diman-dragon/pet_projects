import { useCallback } from "react";
import { StudyService } from "../../../services/db/StudyService";
import type { Card, SessionSummary } from "../../../types/domain";

interface UseStudyActionsOptions {
  studyCards: Card[];
  studyIndex: number;
  studyDeckId: number | null;
  // FIX: принимаем ref, а не значение, чтобы избежать stale closure в finishSession
  sessionReviewedRef: React.MutableRefObject<number>;
  setStudyCards: React.Dispatch<React.SetStateAction<Card[]>>;
  setStudyIndex: React.Dispatch<React.SetStateAction<number>>;
  setStudyFlipped: React.Dispatch<React.SetStateAction<boolean>>;
  setSessionSummary: React.Dispatch<React.SetStateAction<SessionSummary | null>>;
  setStudyDeckId: (id: number | null) => void;
  setView: (view: any) => void;
  // FIX: убран неиспользуемый параметр resetCardTiming
  logView: (card: Card, face: "view" | "view_flipped", ms: number) => void;
  collectMetrics: () => any;
  reloadStats: () => Promise<void>;
  setIsSaving: (isSaving: boolean) => void;
}

export function useStudyActions(options: UseStudyActionsOptions) {
  const {
    studyCards, studyIndex, studyDeckId,
    sessionReviewedRef,
    setStudyCards,
    setSessionSummary, setStudyDeckId, setView,
    collectMetrics, reloadStats, setIsSaving,
  } = options;

  const completeCurrentCard = useCallback(async (behavior?: any) => {
    const card = studyCards[studyIndex];
    if (!card || !studyDeckId) return false;

    const metrics = {
      ...collectMetrics(),
      ...(behavior || {}),
      manualWrong: behavior?.manualWrong ?? false,
    };
    setIsSaving(true);

    try {
      await StudyService.autoReviewCard(card.id, metrics);
      const updatedCard = await StudyService.getCard(card.id);
      if (updatedCard) {
        setStudyCards((prev) => {
          const next = [...prev];
          next[studyIndex] = updatedCard;
          return next;
        });
      }

      // Предзагрузка следующей карточки для ускорения перехода
      if (studyIndex + 1 < studyCards.length) {
        StudyService.getCard(studyCards[studyIndex + 1].id).catch(() => {});
      }
    } catch (err) {
      console.error("[study] autoReviewCard failed", err);
    } finally {
      setIsSaving(false);
    }

    reloadStats().catch(() => undefined);
    return true;
  // FIX: убраны лишние зависимости (logView, setStudyIndex, setStudyFlipped, setSessionSummary),
  // которые не используются внутри — они вызывали лишние ре-рендеры
  }, [studyCards, studyIndex, studyDeckId, collectMetrics, reloadStats, setStudyCards, setIsSaving]);

  const finishSession = useCallback(async () => {
    setIsSaving(true);
    try {
      const streakData = await StudyService.getStreak().catch(() => ({ current: 0 }));

      const summary: SessionSummary = {
        // FIX: ранее использовался studyIndex, который сбрасывается в 0 до формирования summary.
        // Теперь читаем актуальное значение из ref.
        reviewedCount: sessionReviewedRef.current,
        streak: (streakData as any).current ?? 0,
      };

      setSessionSummary(summary);
      setStudyCards([]);
      setStudyDeckId(null);
    } catch (err) {
      console.error("[study] finishSession failed", err);
      setView("decks");
    } finally {
      setIsSaving(false);
      reloadStats().catch(() => undefined);
    }
  }, [sessionReviewedRef, reloadStats, setStudyCards, setStudyDeckId, setView, setSessionSummary, setIsSaving]);

  const clearSession = useCallback(() => {
    setSessionSummary(null);
    setView("decks");
  }, [setSessionSummary, setView]);

  return { completeCurrentCard, finishSession, clearSession };
}
