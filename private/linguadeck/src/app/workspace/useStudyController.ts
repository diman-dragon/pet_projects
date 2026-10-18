import { useRef, useState, useCallback } from "react";
import type { ViewId } from "../../components/layout/SidebarNav";
import { StudyService } from "../../services/db/StudyService";
import type { Card, ReviewRating, SessionSummary } from "../../types/domain";
import type { SessionConfig } from "../../components/study/SessionSetupDialog";

import { shuffleNewTail } from "./study/studyUtils";
import { useStudyTiming } from "./study/useStudyTiming";
import { useStudyActions } from "./study/useStudyActions";

export { INFERRED_RATING_META } from "./study/studyUtils";

interface UseStudyControllerOptions {
  loadWorkspace: () => Promise<void>;
  reloadStats: () => Promise<void>;
  setSelectedDeckId: (deckId: number | null) => void;
  setView: (view: ViewId) => void;
  withFeedback: <T>(scope: string, action: () => Promise<T>, msg?: string) => Promise<T>;
}

export function useStudyController({
  reloadStats,
  setSelectedDeckId,
  setView,
  withFeedback,
}: UseStudyControllerOptions) {
  const [studyDeckId, setStudyDeckId] = useState<number | null>(null);
  const [studyCards, setStudyCards] = useState<Card[]>([]);
  const [studyIndex, setStudyIndex] = useState(0);
  const [studyFlipped, setStudyFlipped] = useState(false);
  const [sessionSummary, setSessionSummary] = useState<SessionSummary | null>(null);
  const [inferredRating, setInferredRating] = useState<ReviewRating | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const sessionReviewedRef = useRef<number>(0);

  const timing = useStudyTiming();
  const { resetCardTiming, collectMetrics, recordFrontTime } = timing;

  const logView = (card: Card, face: "view" | "view_flipped", ms?: number) =>
    timing.logView(card, face, ms);

  const actions = useStudyActions({
    studyCards,
    studyIndex,
    studyDeckId,
    // FIX: передаём ref вместо значения — finishSession читает актуальный счётчик
    sessionReviewedRef,
    setStudyCards,
    setStudyIndex,
    setStudyFlipped,
    setSessionSummary,
    setStudyDeckId,
    setView,
    // FIX: убран resetCardTiming из параметров — useStudyActions его не использовал
    logView: logView as any,
    collectMetrics,
    reloadStats,
    setIsSaving,
  });

  const currentStudyCard = studyCards[studyIndex] ?? null;

  async function startStudy(deckId: number, config?: SessionConfig) {
    await withFeedback("study", async () => {
      setSelectedDeckId(deckId);
      setStudyDeckId(deckId);
      let raw = await StudyService.getSessionCards(deckId);

      if (config) {
        raw = raw.filter((card) => {
          if (card.status === "new") return config.includeNew;
          if (card.status === "review") return config.includeReview;
          if (card.status === "learning") return config.includeLearning;
          return true;
        });
        if (config.limit && config.limit < raw.length) {
          raw = raw.slice(0, config.limit);
        }
      }

      const cards = config?.shuffle !== false ? shuffleNewTail(raw) : raw;
      sessionReviewedRef.current = 0;
      setStudyCards(cards);
      setStudyIndex(0);
      setStudyFlipped(false);
      setSessionSummary(null);
      setInferredRating(null);
      resetCardTiming();
      if (cards[0]) logView(cards[0], "view");
      setView("study");
    });
  }

  const goToNextStudyCard = useCallback(async (behavior?: any) => {
    const card = studyCards[studyIndex];

    // Фиксируем время на текущей грани перед переходом
    const ms = timing.elapsedMs();
    const face = timing.hasFlippedRef.current ? "view_flipped" : "view";
    if (card) logView(card, face, ms);

    // Сохраняем результат карточки
    const success = await actions.completeCurrentCard(behavior);
    sessionReviewedRef.current += 1;

    if (studyIndex < studyCards.length - 1) {
      setStudyIndex((i) => i + 1);
      setStudyFlipped(false);
      resetCardTiming();
    } else {
      // Последняя карточка — завершаем сессию
      await actions.finishSession();
    }
    return success;
  }, [actions, studyIndex, studyCards, timing, resetCardTiming, logView]);

  const goToPreviousStudyCard = useCallback(() => {
    if (studyIndex > 0) {
      const card = studyCards[studyIndex];
      const ms = timing.elapsedMs();
      if (card) logView(card, timing.hasFlippedRef.current ? "view_flipped" : "view", ms);

      setStudyIndex((i) => i - 1);
      setStudyFlipped(false);
      resetCardTiming();
    }
  }, [studyIndex, studyCards, timing, resetCardTiming, logView]);

  const flipStudyCard = useCallback(() => {
    const card = studyCards[studyIndex];
    if (!card) return;

    if (!studyFlipped) {
      // Лицо → оборот:
      // FIX: вызываем recordFrontTime() — она атомарно фиксирует frontTimeRef
      // и сбрасывает faceStartRef. Ранее frontTimeRef никогда не обновлялся,
      // поэтому collectMetrics() всегда возвращал frontTimeMs: 0.
      recordFrontTime();
      logView(card, "view_flipped", timing.frontTimeRef.current);
    } else {
      // Оборот → лицо: логируем время на обороте
      const ms = timing.elapsedMs();
      logView(card, "view", ms);
      timing.faceStartRef.current = Date.now();
    }

    setStudyFlipped((prev) => !prev);
    setInferredRating(null);
  }, [studyIndex, studyCards, studyFlipped, timing, logView, recordFrontTime]);

  return {
    studyCards,
    studyIndex,
    studyFlipped,
    currentStudyCard,
    studyDeckId,
    sessionSummary,
    inferredRating,
    isSaving,
    sessionCount: sessionReviewedRef.current,
    startStudy,
    completeCurrentCard: actions.completeCurrentCard,
    finishSession: actions.finishSession,
    clearSession: actions.clearSession,
    flipStudyCard,
    goToPreviousStudyCard,
    goToNextStudyCard,
  } as const;
}
