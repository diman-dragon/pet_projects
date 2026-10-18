import { createContext, useContext } from "react";
import type { Card, SessionSummary } from "../../../types/domain";

interface StudyContextValue {
  studyCards: Card[];
  studyIndex: number;
  studyFlipped: boolean;
  currentStudyCard: Card | null;
  sessionSummary: SessionSummary | null;
  isSaving: boolean;
  completeCurrentCard: (behavior?: any) => Promise<boolean>;
  goToNextStudyCard: (behavior?: any) => Promise<boolean>;
  goToPreviousStudyCard: () => void;
  flipStudyCard: () => void;
  finishSession: () => void;
  clearSession: () => void;
}

export const StudyContext = createContext<StudyContextValue | null>(null);

export function useStudy() {
  const context = useContext(StudyContext);
  if (!context) throw new Error("useStudy must be used within StudyProvider");
  return context;
}
