import { invokeDb } from "./invokeDb";

export interface CardReviewEntry {
  rating: number;
  rating_source?: "auto";
  front_time_ms?: number;
  total_time_ms?: number;
  flips_count?: number;
  backtracks_count?: number;
  interval_before: number;
  interval_after: number;
  ease_before: number;
  ease_after: number;
  reviewed_at: string;
}

export interface CardStats {
  // Просмотры
  views_total: number;        // сколько раз показана лицом
  flips_total: number;        // сколько раз перевёрнута
  flip_rate: number;          // % сессий, где карточку перевернули (вычисляется на клиенте)
  avg_view_ms: number;        // среднее время чтения лицевой стороны, мс
  total_time_ms: number;      // суммарное время на карточке, мс
  // Повторения
  reviews_total: number;
  avg_rating: number;
  best_rating: number;
  worst_rating: number;
  recent_reviews: CardReviewEntry[];
}

export const CardStatsService = {
  get(cardId: number) {
    return invokeDb<CardStats>("card_stats", { cardId }).then((raw) => {
      const views = raw.views_total ?? 0;
      const flips = raw.flips_total ?? 0;
      return {
        ...raw,
        flip_rate: views > 0 ? Math.round((flips / views) * 100) : 0,
      } as CardStats;
    });
  },
};
