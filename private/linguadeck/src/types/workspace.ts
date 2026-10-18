import type { ViewId } from "../components/layout/SidebarNav";
import type {
  AppLogEntry,
  AppSettings,
  ReviewRating,
  Card,
  Deck,
  DashboardStats,
  ImportProgress,
  SessionSummary,
  WeekActivityDay,
} from "./domain";

export interface WorkspacePagesProps {
  view: ViewId;
  settings: AppSettings;
  stats: DashboardStats;
  decks: Deck[];
  cards: Card[];
  logs: AppLogEntry[];
  selectedDeck: Deck | null;
  selectedDeckId: number | null;
  selectedCardIds: number[];
  deckSearch: string;
  cardSearch: string;
  studyCards: Card[];
  studyIndex: number;
  studyFlipped: boolean;
  currentStudyCard: Card | null;
  inferredRating: ReviewRating | null;
  sessionCount: number;
  sessionSummary: SessionSummary | null;
  streak: number;
  reviewedToday: number;
  totalTimeMs: number;
  weekActivity: WeekActivityDay[];
  grammarFocusId?: number | null;
  importProgress?: ImportProgress;
  onChangeDeckSearch: (value: string) => void;
  onCreateDeck: () => void;
  onOpenCards: (deckId: number) => void;
  onStudyDeck: (deck: Deck) => void;
  onEditDeck: (deck: Deck) => void;
  onDeleteDeck: (deckId: number) => void;
  onChangeCardSearch: (value: string) => void;
  onCreateCard: () => void;
  onDeleteCard: (cardId: number) => void;
  onDeleteSelected: () => void;
  onEditCard: (card: Card) => void;
  onFinishSession: () => void;
  onImport: () => void;
  onSelectDeck: (deckId: number | null) => void;
  onToggleCard: (cardId: number) => void;
  onFlip: () => void;
  onNext: () => void;
  onPrev: () => void;
  onCompleteCard: (behavior?: any) => Promise<boolean>;
  onPlayAudio?: (audioPath: string) => void;
  onPlayVideo?: (videoPath: string) => void;
  onCardEdited?: (cardId: number) => void;
  onOpenGrammarTopic?: (unitId: number) => void;
  onChangeSettings: (patch: Partial<AppSettings>) => void;
  onClearLogs: () => void;
}
