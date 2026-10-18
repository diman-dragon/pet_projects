export type ReviewRating = 1 | 2 | 3 | 4;
export type AutoRating = ReviewRating;

export type CardLearningStatus = "new" | "learning" | "review";

export interface AppSettings {
  learningLanguage: string;
  nativeLanguage: string;
  mediaRoot: string | null;
  studyAlgorithm: "sm2" | "cram" | "fsrs";
  learningSteps: string;
  graduatingInterval: number;
  easyInterval: number;
  maxReviewsPerSession: number;
}

export interface Deck {
  id: number;
  course_id: number;
  name: string;
  description: string;
  icon: string;
  color: string;
  /** ISO 639-1 language code or full name, e.g. "en", "English", "de" */
  language: string;
  cards_count: number;
  due_count: number;
  new_count: number;
  created_at: string;
  updated_at: string;
}

export interface Card {
  id: number;
  deck_id: number;
  front: string;
  back: string;
  context: string | null;
  transcription: string | null;
  etymology: string | null;
  mnemonic: string | null;
  notes: string | null;
  example_1: string | null;
  example_2: string | null;
  example_3: string | null;
  audio_path: string | null;
  image_path: string | null;
  video_path: string | null;
  source: string | null;
  tags: string | null;
  status: CardLearningStatus;
  due_at: string;
  last_reviewed_at: string | null;
  interval_days: number;
  ease_factor: number;
  repetitions: number;
  lapses: number;
  created_at: string;
  updated_at: string;
}

export interface CardDraft {
  front: string;
  back: string;
  context?: string | null;
  transcription?: string | null;
  etymology?: string | null;
  mnemonic?: string | null;
  notes?: string | null;
  example_1?: string | null;
  example_2?: string | null;
  example_3?: string | null;
  audio_path?: string | null;
  image_path?: string | null;
  video_path?: string | null;
  source?: string | null;
  tags?: string[] | string | null;
}

export interface DashboardStats {
  totalDecks: number;
  totalCards: number;
  newCards: number;
  learningCards: number;
  reviewCards: number;
  masteredCards: number;
  dueToday: number;
  reviewedToday: number;
  newLearnedToday: number;
  viewedToday: number;
  viewedFlippedToday: number;
  studiedAllTime: number;
  viewedAllTime: number;
  viewedFlippedAllTime: number;
  streakDays: number;
  shownToday: number;
  shownFlippedToday: number;
  /** Total unique cards viewed today (from view_logs, localtime) */
  uniqueCardsToday: number;
  /** Total time spent studying today in ms */
  timeTodayMs: number;
  weekActivity: WeekActivityDay[];
  averageEase: number;
  retentionRate: number;
  retentionRounded: number;
  completionRate: number;
  newCardsPct: number;
  learningCardsPct: number;
  reviewOnlyPct: number;
  masteredCardsPct: number;
}

export interface ForecastDay {
  date: string;
  count: number;
  reviewed?: number;
  kind: "past" | "today" | "future";
  weekday: number;
}

export type ViewType = "view" | "view_flipped";

export interface WeekActivityDay {
  date: string;
  label: string;
  weekday: number;
  count: number;
  uniqueCount?: number;
  flipsCount?: number;
  timeMs: number;
  isToday: boolean;
  hasActivity: boolean;
}

export interface ViewLog {
  id: number;
  card_id: number;
  deck_id: number;
  view_type: ViewType;
  viewed_at: string;
}

export interface StudyCard extends Card {
  position: number;
  total: number;
}

export interface SessionSummary {
  reviewedCount: number;
  streak: number;
}

export interface ReviewLog {
  id: number;
  card_id: number;
  deck_id: number;
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

export interface AppLogEntry {
  id: number;
  level: "info" | "warn" | "error";
  scope: string;
  message: string;
  details: string | null;
  created_at: string;
}

export type GrammarStatus = "not_started" | "in_progress" | "mastered";
export type GrammarExerciseType = "choice" | "fill_blank" | "reorder" | "translate";

export interface GrammarExercise {
  id: number;
  unit_id: number;
  type: GrammarExerciseType;
  prompt: string;
  payload: string;
  answer: string;
  explanation: string;
  created_at: string;
}

export interface GrammarUnit {
  id: number;
  title: string;
  language: string;
  level: string;
  section: string;
  tags: string | null;
  theory: string;
  examples: string;
  status: GrammarStatus;
  mastery_streak: number;
  last_score: number | null;
  last_practiced_at: string | null;
  next_review_at: string | null;
  exercise_count: number;
  linked_cards_count: number;
  created_at: string;
  updated_at: string;
}

export interface GrammarSessionResult {
  unit_id: number;
  score: number;
  correct_count: number;
  total_count: number;
  status: GrammarStatus;
  mastery_streak: number;
  next_review_at: string | null;
}

export interface GrammarDashboard {
  total: number;
  notStarted: number;
  inProgress: number;
  mastered: number;
  dueReviews: number;
}

export interface GrammarPracticeItem extends GrammarExercise {
  userAnswer?: string;
  isCorrect?: boolean;
}

/** Progress state for batch card import */
export interface ImportProgress {
  total: number;
  inserted: number;
  status: "idle" | "running" | "done" | "error";
  error?: string;
}
