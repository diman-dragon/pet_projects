import { CardsPage } from "../components/pages/CardsPage";
import { DecksPage } from "../components/pages/DecksPage";
import { DonatePage } from "../components/pages/DonatePage";
import { GrammarPage } from "../components/pages/GrammarPage";
import { OverviewPage } from "../components/pages/OverviewPage";
import { SettingsPage } from "../components/pages/SettingsPage";
import { StudyPage } from "../components/pages/StudyPage";
import type { WorkspacePagesProps } from "../types/workspace";

function ScrollablePage({ children }: { children: React.ReactNode }) {
  return <div className="custom-scrollbar flex-1 overflow-auto pr-1">{children}</div>;
}

function FixedPage({ children }: { children: React.ReactNode }) {
  return <div className="flex-1 min-h-0 overflow-hidden">{children}</div>;
}

export function WorkspacePages(props: WorkspacePagesProps) {
  switch (props.view) {
    case "decks":
      return (
        <ScrollablePage>
          <DecksPage
            decks={props.decks}
            search={props.deckSearch}
            importProgress={props.importProgress}
            onChangeSearch={props.onChangeDeckSearch}
            onCreateDeck={props.onCreateDeck}
            onOpenCards={props.onOpenCards}
            onStudy={props.onStudyDeck}
            onEditDeck={props.onEditDeck}
            onDeleteDeck={props.onDeleteDeck}
          />
        </ScrollablePage>
      );

    case "cards":
      return (
        <ScrollablePage>
          <CardsPage
            cards={props.cards}
            decks={props.decks}
            selectedDeck={props.selectedDeck}
            search={props.cardSearch}
            selectedCardIds={props.selectedCardIds}
            selectedDeckId={props.selectedDeckId}
            onChangeSearch={props.onChangeCardSearch}
            onCreateCard={props.onCreateCard}
            onDeleteCard={props.onDeleteCard}
            onDeleteSelected={props.onDeleteSelected}
            onEditCard={props.onEditCard}
            onImport={props.onImport}
            onSelectDeck={props.onSelectDeck}
            onToggleCard={props.onToggleCard}
          />
        </ScrollablePage>
      );

    case "study":
      console.log("WorkspacePages: study card ID =", props.currentStudyCard?.id, "status =", props.currentStudyCard?.status);
      return (
        <FixedPage>
          <StudyPage
            card={props.currentStudyCard}
            deckName={props.selectedDeck?.name ?? "Сессия"}
            total={props.studyCards.length}
            sessionCount={props.sessionCount}
            streak={props.streak}
            reviewedToday={props.reviewedToday}
            totalTimeMs={props.totalTimeMs}
            weekActivity={props.weekActivity}
            onPlayAudio={props.onPlayAudio}
            onPlayVideo={props.onPlayVideo}
            onEditCard={props.onEditCard}
            onOpenGrammarTopic={props.onOpenGrammarTopic}
          />
        </FixedPage>
      );

    case "grammar":
      return (
        <FixedPage>
          <GrammarPage focusUnitId={props.grammarFocusId} />
        </FixedPage>
      );

    case "settings":
      return (
        <ScrollablePage>
          <SettingsPage
            logs={props.logs}
            settings={props.settings}
            onChangeSettings={props.onChangeSettings}
            onClearLogs={props.onClearLogs}
          />
        </ScrollablePage>
      );

    case "donate":
      return (
        <ScrollablePage>
          <DonatePage />
        </ScrollablePage>
      );

    default:
      return (
        <ScrollablePage>
          <OverviewPage settings={props.settings} stats={props.stats} />
        </ScrollablePage>
      );
  }
}
