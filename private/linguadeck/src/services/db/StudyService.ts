import type { ForecastDay } from "../../types/domain";
import type { Card } from "../../types/domain";
import { invokeDb } from "./invokeDb";

export interface DayCardEntry {
  card_id: number;
  front: string;
  back: string;
  status: string;
  deck_name: string;
  deck_color: string;
  rating: number;
  rating_source: "auto" | null;
  interval_after: number;
  reviewed_at: string;
}

export interface ViewCardEntry {
  card_id: number;
  front: string;
  back: string;
  status: string;
  deck_name: string;
  deck_color: string;
  views: number;
}

export interface DayViewSummary {
  uniqueCards: number;
  totalShows: number;
  totalFlips: number;
  avgFrontMs: number;
  avgBackMs: number;
  totalTimeMs: number;
}

export interface DayReviewSummary {
  totalReviews: number;
  reviewedCards: number;
  rating1: number;
  rating2: number;
  rating3: number;
  rating4: number;
  avgFrontMs: number;
  avgTotalMs: number;
  totalFlips: number;
  avgFlips: number;
}

export interface DayCardStat {
  cardId: number;
  front: string;
  back: string;
  status: string;
  deckName: string;
  deckColor: string;
  shows: number;
  flips: number;
  totalViewMs: number;
  lastRating: number | null;
  intervalAfter: number | null;
  reviewedAt: string | null;
}

export interface DayStats {
  date: string;
  summary: DayViewSummary;
  reviews: DayReviewSummary;
  cards: DayCardStat[];
}

export interface StudyBehaviorMetrics {
  frontTimeMs: number;
  totalTimeMs: number;
  flipsCount: number;
  backtracksCount: number;
}

export const StudyService = {
  getSessionCards(deckId: number) {
    return invokeDb<Card[]>("study_session_cards", { deckId });
  },

  autoReviewCard(cardId: number, metrics: StudyBehaviorMetrics) {
    return invokeDb("study_auto_review_card", { cardId, metrics });
  },

  getCard(cardId: number) {
    return invokeDb<Card>("card_get", { cardId });
  },

  getDueForecast(days = 30) {
    return invokeDb<Array<ForecastDay>>("study_forecast", { days });
  },

  getCardsForDay(date: string) {
    return invokeDb<DayCardEntry[]>("study_cards_for_day", { date });
  },

  getViewsForDay(date: string) {
    return invokeDb<ViewCardEntry[]>("views_for_day", { date });
  },

  getActivityHistory(days: number) {
    return invokeDb<Array<{ date: string; count: number }>>("activity_history", { days });
  },

  getDayStats(date: string) {
    return invokeDb<DayStats>("day_stats", { date });
  },

  getStreak() {
    return invokeDb<{ current: number; best: number; is_active: boolean; message: string }>("streak", {});
  }
};
