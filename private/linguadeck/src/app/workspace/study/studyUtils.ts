import type { Card, ReviewRating } from "../../../types/domain";

export const INFERRED_RATING_META: (t: any) => Record<ReviewRating, { color: string; label: string }> = (t) => ({
  1: { color: "#ef4444", label: t("study.rating.again") },
  2: { color: "#f97316", label: t("study.rating.hard") },
  3: { color: "#3b82f6", label: t("study.rating.good") },
  4: { color: "#10b981", label: t("study.rating.easy") },
});

export function inferRating(frontTimeMs: number, hasFlipped: boolean): ReviewRating {
  if (!hasFlipped) return 2; // Минимальный рейтинг - "сложно"
  // Если смотрел меньше 5 секунд - легко (4), до 15 - хорошо (3), дольше - сложно (2)
  if (frontTimeMs < 5_000) return 4;
  if (frontTimeMs < 15_000) return 3;
  return 2;
}

export function shuffleNewTail(cards: Card[]): Card[] {
  const firstNewIdx = cards.findIndex((c) => c.status === "new");
  if (firstNewIdx < 0) {
    const shuffled = [...cards];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }
  const due = cards.slice(0, firstNewIdx);
  const newCards = cards.slice(firstNewIdx);
  for (let i = due.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [due[i], due[j]] = [due[j], due[i]];
  }
  for (let i = newCards.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [newCards[i], newCards[j]] = [newCards[j], newCards[i]];
  }
  return [...due, ...newCards];
}
