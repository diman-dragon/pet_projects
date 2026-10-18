import { useCallback } from "react";
import type { Card, ReviewRating } from "../../../types/domain";
import { inferRating } from "./studyUtils";
import { useStudyTiming } from "./useStudyTiming";

interface UseStudyNavigationOptions {
  studyCardsRef: React.MutableRefObject<Card[]>;
  studyIndexRef: React.MutableRefObject<number>;
  timing: ReturnType<typeof useStudyTiming>;
  setStudyIndex: (i: number) => void;
  setStudyFlipped: React.Dispatch<React.SetStateAction<boolean>>;
  setInferredRating: (r: ReviewRating | null) => void;
  onComplete: () => Promise<boolean>;
}

export function useStudyNavigation({
  studyCardsRef,
  studyIndexRef,
  timing,
  setStudyIndex,
  setStudyFlipped,
  setInferredRating,
  onComplete,
}: UseStudyNavigationOptions) {
  const { logView, resetCardTiming, hasFlippedRef } = timing;
  const elapsedMs = timing.elapsedMs;

  const goToCard = useCallback(
    async (nextIndex: number, isBacktrack: boolean = false) => {
      const cards = studyCardsRef.current;
      if (nextIndex < 0 || nextIndex >= cards.length) return;

      if (!isBacktrack) {
        // onComplete() already calls logView for the current card with correct timing
        await onComplete();
      }

      setStudyIndex(nextIndex);
      setStudyFlipped(false);
      setInferredRating(null);
      resetCardTiming();
      // Note: we do NOT log a "view" here for the next card.
      // The view for each card is logged in completeCurrentCard() when the user moves away from it,
      // and the very first card is logged in startStudy(). This avoids double-logging and zero-time entries.
    },
    [onComplete, resetCardTiming, setStudyIndex, setStudyFlipped, setInferredRating]
  );


  function goToPreviousStudyCard() {
    goToCard(studyIndexRef.current - 1, true);
  }

  function goToNextStudyCard() {
    goToCard(studyIndexRef.current + 1, false);
  }

  function flipStudyCard() {
    const card = studyCardsRef.current[studyIndexRef.current];
    if (!card) return;

    setStudyFlipped((wasFlipped: boolean) => {
      const ms = elapsedMs();
      if (!wasFlipped) {
        hasFlippedRef.current = true;
        logView(card, "view_flipped", ms);
        setInferredRating(inferRating(ms, true));
      } else {
        logView(card, "view", ms);
        setInferredRating(null);
      }
      return !wasFlipped;
    });
  }

  return { goToCard, goToPreviousStudyCard, goToNextStudyCard, flipStudyCard };
}
