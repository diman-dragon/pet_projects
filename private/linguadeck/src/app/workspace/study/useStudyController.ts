import { useRef, useState } from "react";
import type { ViewId } from "../../../components/layout/SidebarNav";
import { StudyService } from "../../../services/db/StudyService";
import type { Card, ReviewRating, SessionSummary } from "../../../types/domain";
import type { SessionConfig } from "../../../components/study/SessionSetupDialog";
import { shuffleNewTail } from "./studyUtils";
import { useStudyTiming } from "./useStudyTiming";
import { useStudyNavigation } from "./useStudyNavigation";

export { INFERRED_RATING_META } from "./studyUtils";

interface UseStudyControllerOptions {
  loadWorkspace: () => Promise<void>;
  reloadStats: () => Promise<void>;
  setSelectedDeckId: (deckId: number | null) => void;
  setView: (view: ViewId) => void;
  withFeedback: <T>(scope: string, action: () => Promise<T>, msg?: string) => Promise<T>;
}

export interface UseStudyControllerReturn {
  studyCards: Card[];
  studyIndex: number;
  studyFlipped: boolean;
  currentStudyCard: Card | null;
  studyDeckId: number | null;
  sessionSummary: SessionSummary | null;
  inferredRating: ReviewRating | null;
  sessionCount: number;
  weekActivity: any[];
  startStudy: (deckId: number, config?: SessionConfig) => Promise<void>;
  completeCurrentCard: () => Promise<boolean>;
  finishSession: () => void;
  flipStudyCard: () => void;
  goToPreviousStudyCard: () => void;
  goToNextStudyCard: () => void;
}

export function useStudyController({
  reloadStats,
  setSelectedDeckId,
  setView,
  withFeedback,
}: UseStudyControllerOptions): UseStudyControllerReturn {
  const [studyDeckId, setStudyDeckId] = useState<number | null>(null);
  const [studyCards, setStudyCards] = useState<Card[]>([]);
  const [studyIndex, setStudyIndex] = useState(0);
  const [studyFlipped, setStudyFlipped] = useState(false);
  const [sessionSummary, setSessionSummary] = useState<SessionSummary | null>(null);
  const [inferredRating, setInferredRating] = useState<ReviewRating | null>(null);

  const sessionReviewedRef = useRef<number>(0);
  const studyCardsRef = useRef<Card[]>([]);
  const studyIndexRef = useRef<number>(0);
  studyCardsRef.current = studyCards;
  studyIndexRef.current = studyIndex;

  const currentStudyCard = studyCards[studyIndex] ?? null;

  const timing = useStudyTiming();
  const { resetCardTiming, elapsedMs, logView, collectMetrics } = timing;

  const nav = useStudyNavigation({
    studyCardsRef,
    studyIndexRef,
    timing,
    setStudyIndex,
    setStudyFlipped,
    setInferredRating,
    onComplete: completeCurrentCard,
  });

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
async function completeCurrentCard() {
  const card = studyCards[studyIndex];
  if (!card || !studyDeckId) return false;

  const ms = elapsedMs();
  const face = studyFlipped ? "view_flipped" : "view";
  logView(card, face, ms);

  const metrics = collectMetrics();

  setInferredRating(null);
  resetCardTiming();
  sessionReviewedRef.current += 1;

  try {
    await StudyService.autoReviewCard(card.id, metrics);
    const updatedCard = await StudyService.getCard(card.id);
    if (updatedCard) {
      setStudyCards(prev => {
        const next = [...prev];
        next[studyIndex] = updatedCard;
        return next;
      });
    }
  } catch (err) {
    console.error("[study] autoReviewCard failed", err);
  }

  // Переходим к следующей, если это не конец
  if (studyIndex < studyCards.length - 1) {
     setStudyIndex(prev => prev + 1);
     setStudyFlipped(false);
  } else {
     setSessionSummary({ reviewedCount: sessionReviewedRef.current, streak: 1 });
     setView("study");
  }

  reloadStats().catch(() => undefined);
  return true;
}


  function finishSession() {
    const card = studyCardsRef.current[studyIndexRef.current];
    if (card) {
      logView(card, studyFlipped ? "view_flipped" : "view", elapsedMs());
    }
    setStudyCards([]);
    setStudyIndex(0);
    setStudyFlipped(false);
    setInferredRating(null);
    resetCardTiming();
    if (sessionReviewedRef.current > 0) {
      setSessionSummary({ reviewedCount: sessionReviewedRef.current, streak: 1 });
      setView("study");
    } else {
      setStudyDeckId(null);
      setView("decks");
    }
    reloadStats().catch(() => undefined);
  }

  return {
    studyCards,
    studyIndex,
    studyFlipped,
    currentStudyCard,
    studyDeckId,
    sessionSummary,
    inferredRating,
    sessionCount: sessionReviewedRef.current,
    weekActivity: [],
    startStudy,
    completeCurrentCard,
    finishSession,
    flipStudyCard: nav.flipStudyCard,
    goToPreviousStudyCard: nav.goToPreviousStudyCard,
    goToNextStudyCard: nav.goToNextStudyCard,
  } as const;
}
