import { useCallback, useState } from "react";
import { useCardsController } from "./useCardsController";
import { useConfirmAction } from "./useConfirmAction";
import { useDecksController } from "./useDecksController";
import { useSettingsController } from "./useSettingsController";
import { useStudyController } from "./useStudyController";
import { useStudyHotkeys } from "./useStudyHotkeys";
import { useWorkspaceConfirm } from "./useWorkspaceConfirm";
import { useWorkspaceData } from "./useWorkspaceData";
import { useWorkspaceDialogs } from "./useWorkspaceDialogs";
import { useWorkspaceFeedback } from "./useWorkspaceFeedback";
import { useWorkspaceFilters } from "./useWorkspaceFilters";
import { useWorkspaceSelection } from "./useWorkspaceSelection";
import { useAudioPlayer } from "./useAudioPlayer";
import { useWorkspaceView } from "./useWorkspaceView";

interface OrchestratorProps {
  libraryPath: string;
}

export function useWorkspaceOrchestrator({ libraryPath }: OrchestratorProps) {
  const [grammarFocusId, setGrammarFocusId] = useState<number | null>(null);
  const { view, setView } = useWorkspaceView();
  const audio = useAudioPlayer({ libraryPath });
  const selection = useWorkspaceSelection();
  const dialogState = useWorkspaceDialogs();
  const { message, isError, withFeedback } = useWorkspaceFeedback();
  const confirm = useWorkspaceConfirm();
  const data = useWorkspaceData({ selectedDeckId: selection.selectedDeckId });
  const filters = useWorkspaceFilters(data.decks, data.cards);

  const study = useStudyController({
    loadWorkspace: data.loadWorkspace,
    reloadStats: data.reloadStats,
    setSelectedDeckId: selection.setSelectedDeckId,
    setView,
    withFeedback,
  });

  const decks = useDecksController({
    decks: data.decks,
    editingDeck: dialogState.editingDeck,
    deckDraft: dialogState.deckDraft,
    setEditingDeck: dialogState.setEditingDeck,
    setDeckDraft: dialogState.setDeckDraft,
    setDeckEditorOpen: dialogState.setDeckEditorOpen,
    loadWorkspace: data.loadWorkspace,
    withFeedback,
  });

  const cards = useCardsController({
    selectedDeckId: selection.selectedDeckId,
    selectedCardIds: selection.selectedCardIds,
    editingCardId: dialogState.editingCardId,
    cardDraft: dialogState.cardDraft,
    setEditingCardId: dialogState.setEditingCardId,
    setCardDraft: dialogState.setCardDraft,
    setCardEditorOpen: dialogState.setCardEditorOpen,
    setSelectedCardIds: selection.setSelectedCardIds,
    loadWorkspace: data.loadWorkspace,
    withFeedback,
  });

  const settings = useSettingsController({
    settings: data.settings,
    loadWorkspace: data.loadWorkspace,
    withFeedback,
  });

  const confirmAction = useConfirmAction({
    confirmState: confirm.confirmState,
    closeConfirm: confirm.closeConfirm,
    deleteDeck: decks.deleteDeck,
    deleteCard: cards.deleteCard,
    deleteSelectedCards: cards.deleteSelectedCards,
  });

  // FIX: добавлен guard study.isSaving — без него быстрые нажатия Space/ArrowRight
  // могли запустить несколько переходов подряд, пока предыдущий ещё сохранялся
  useStudyHotkeys({
    enabled: view === "study" && !!study.currentStudyCard && !study.isSaving,
    flipped: study.studyFlipped,
    total: study.studyCards.length,
    onNext: study.goToNextStudyCard,
    onFlip: study.flipStudyCard,
    onPrev: study.goToPreviousStudyCard,
  });

  const settingsHandler = useCallback(
    (patch: Partial<NonNullable<typeof data.settings>>) => {
      data.setSettings((current) =>
        // FIX: убрано принудительное nativeLanguage: "Russian" —
        // ранее любой patch сбрасывал родной язык пользователя обратно на Russian
        current ? { ...current, ...patch } : current
      );
      settings.debouncedSave();
    },
    [data.setSettings, settings.debouncedSave]
  );

  return {
    view,
    setView,
    grammarFocusId,
    setGrammarFocusId,
    audio,
    selection,
    dialogState,
    message,
    isError,
    confirm,
    data,
    filters,
    study,
    decks,
    cards,
    settings,
    confirmAction,
    settingsHandler,
  };
}
