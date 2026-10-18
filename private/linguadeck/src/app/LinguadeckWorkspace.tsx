import { SidebarNav } from "../components/layout/SidebarNav";
import { WorkspaceDialogs } from "./WorkspaceDialogs";
import { WorkspacePages } from "./WorkspacePages";
import { StudyContext } from "./workspace/study/StudyContext";
import { useWorkspaceOrchestrator } from "./workspace/useWorkspaceOrchestrator";

interface LinguadeckWorkspaceProps {
  libraryPath: string;
}

export function LinguadeckWorkspace({ libraryPath }: LinguadeckWorkspaceProps) {
  const {
    view,
    setView,
    grammarFocusId,
    setGrammarFocusId,
    audio,
    selection,
    dialogState,
    data,
    filters,
    study,
    decks,
    cards,
    settings,
    confirm,
    confirmAction,
    settingsHandler,
  } = useWorkspaceOrchestrator({ libraryPath });

  if (!data.settings || !data.stats) {
    return <div className="min-h-screen bg-slate-100" />;
  }

  return (
    <div className="min-h-screen bg-[linear-gradient(180deg,#eff4fa_0%,#f8fafc_26%,#edf2f7_100%)] text-slate-900">
      <div className="mx-auto flex min-h-screen max-w-[1700px] gap-6 px-5 py-5">
        <SidebarNav activeView={view} onChangeView={setView} />
        <main className="glass-panel ml-60 flex min-h-[calc(100vh-40px)] flex-1 flex-col overflow-hidden p-6">
          <StudyContext.Provider value={study}>
            <WorkspacePages
              key={`workspace-${study.studyIndex}-${study.currentStudyCard?.status}`}
              view={view}
            settings={data.settings}
            stats={data.stats}
            decks={filters.visibleDecks}
            cards={filters.visibleCards}
            logs={data.logs}
            selectedDeck={data.selectedDeck}
            selectedDeckId={selection.selectedDeckId}
            selectedCardIds={selection.selectedCardIds}
            deckSearch={filters.deckSearch}
            cardSearch={filters.cardSearch}
            studyCards={study.studyCards}
            studyIndex={study.studyIndex}
            studyFlipped={study.studyFlipped}
            currentStudyCard={study.currentStudyCard}
            sessionCount={study.sessionCount}
            sessionSummary={study.sessionSummary}
            weekActivity={data.stats.weekActivity}
            streak={data.stats.streakDays}
            reviewedToday={data.stats.reviewedToday}
            totalTimeMs={data.stats.weekActivity.find((day) => day.isToday)?.timeMs ?? 0}
            grammarFocusId={grammarFocusId}
            onChangeDeckSearch={filters.setDeckSearch}
            onCreateDeck={decks.openCreateDeck}
            onOpenCards={(deckId) => selection.openDeckCards(deckId, () => setView("cards"))}
            onStudyDeck={dialogState.openSessionSetup}
            onEditDeck={decks.openEditDeck}
            onDeleteDeck={(deckId) => confirm.setConfirmState({ kind: "deck", id: deckId })}
            onChangeCardSearch={filters.setCardSearch}
            onCreateCard={cards.openCreateCard}
            onDeleteCard={(cardId) => confirm.setConfirmState({ kind: "card", id: cardId })}
            onDeleteSelected={() => confirm.setConfirmState({ kind: "cards" })}
            onEditCard={cards.openEditCard}
            onFinishSession={study.finishSession}
            onImport={cards.importCards}
            onSelectDeck={selection.selectDeck}
            onToggleCard={cards.toggleCardSelection}
            onFlip={study.flipStudyCard}
            onNext={study.goToNextStudyCard}
            onPrev={study.goToPreviousStudyCard}
            onCompleteCard={study.completeCurrentCard}
            inferredRating={study.inferredRating}
            onPlayAudio={audio.playAudio}
            onOpenGrammarTopic={(unitId) => {
              setGrammarFocusId(unitId);
              setView("grammar");
            }}
            onChangeSettings={settingsHandler}
            onClearLogs={settings.clearLogs}
            />
            </StudyContext.Provider>
            </main>
            </div>
      <WorkspaceDialogs
        cardEditorOpen={dialogState.cardEditorOpen}
        editingCardId={dialogState.editingCardId}
        cardDraft={dialogState.cardDraft}
        selectedDeckName={data.selectedDeck?.name ?? null}
        onChangeCardDraft={(field, value) =>
          dialogState.setCardDraft((current) => ({ ...current, [field]: value }))
        }
        onCloseCardEditor={() => dialogState.setCardEditorOpen(false)}
        onPickAudio={() => cards.pickMedia("audio_path")}
        onPickVideo={() => cards.pickMedia("video_path")}
        onPickImage={() => cards.pickMedia("image_path")}
        onSaveCard={cards.saveCard}
        deckEditorOpen={dialogState.deckEditorOpen}
        editingDeck={dialogState.editingDeck}
        deckDraft={dialogState.deckDraft}
        onChangeDeckDraft={(patch) =>
          dialogState.setDeckDraft((current) => ({ ...current, ...patch }))
        }
        onCloseDeckEditor={() => dialogState.setDeckEditorOpen(false)}
        onSaveDeck={decks.saveDeck}
        confirmOpen={confirm.confirmState !== null}
        confirmTitle={confirmAction.confirmTitle()}
        confirmMessage={confirmAction.confirmMessage()}
        onCloseConfirm={confirm.closeConfirm}
        onConfirm={confirmAction.onConfirm}
        sessionSetupOpen={dialogState.sessionSetupOpen}
        pendingSessionDeck={dialogState.pendingSessionDeck}
        onCloseSessionSetup={dialogState.closeSessionSetup}
        onConfirmSessionSetup={(config) => {
          const deck = dialogState.pendingSessionDeck;

          dialogState.confirmSessionSetup(config);
          if (deck?.id) {

            study.startStudy(deck.id);
          } else {

          }
        }}
        sessionDeckStats={
          dialogState.pendingSessionDeck
            ? (() => {
              const deck = dialogState.pendingSessionDeck;
              const newCount = deck.new_count || 0;
              const dueCount = deck.due_count || 0;
              // FIX: ранее review = due_count (завышено, включало learning).
              // Теперь: due = learning + review, поэтому review = due - new - learning.
              // learning = due - new (карточки не новые, но ещё в обучении)
              const learningCount = Math.max(0, dueCount - newCount);
              const reviewCount = Math.max(0, dueCount - newCount - learningCount);
              return {
                new: newCount,
                learning: learningCount,
                review: reviewCount,
                total: deck.cards_count || 0,
              };
            })()
            : { new: 0, learning: 0, review: 0, total: 0 }
        }
      />
    </div>
  );
}
