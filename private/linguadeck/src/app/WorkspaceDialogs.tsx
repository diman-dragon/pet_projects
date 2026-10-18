import { CardEditorModal } from "../components/cards/CardEditorModal";
import { DeckEditorModal } from "../components/decks/DeckEditorModal";
import { ConfirmModal } from "../components/ui/ConfirmModal";
import { SessionSetupDialog, type SessionConfig } from "../components/study/SessionSetupDialog";
import type { CardDraft, Deck } from "../types/domain";
import type { DeckDraft } from "./workspace/useDecksController";

interface WorkspaceDialogsProps {
  cardEditorOpen: boolean;
  editingCardId: number | null;
  cardDraft: CardDraft;
  selectedDeckName: string | null;
  onChangeCardDraft: (field: keyof CardDraft, value: string) => void;
  onCloseCardEditor: () => void;
  onPickAudio: () => void;
  onPickVideo: () => void;
  onPickImage: () => void;
  onSaveCard: () => void;
  deckEditorOpen: boolean;
  editingDeck: Deck | null;
  deckDraft: DeckDraft;
  onChangeDeckDraft: (patch: Partial<DeckDraft>) => void;
  onCloseDeckEditor: () => void;
  onSaveDeck: () => void;
  confirmOpen: boolean;
  confirmTitle: string;
  confirmMessage: string;
  onCloseConfirm: () => void;
  onConfirm: () => void;
  sessionSetupOpen: boolean;
  pendingSessionDeck: Deck | null;
  onCloseSessionSetup: () => void;
  onConfirmSessionSetup: (config: SessionConfig) => void;
  sessionDeckStats: {
    new: number;
    learning: number;
    review: number;
    total: number;
  };
}

export function WorkspaceDialogs(props: WorkspaceDialogsProps) {
  return (
    <>
      <CardEditorModal
        open={props.cardEditorOpen}
        editing={Boolean(props.editingCardId)}
        draft={props.cardDraft}
        deckName={props.selectedDeckName}
        onChange={props.onChangeCardDraft}
        onClose={props.onCloseCardEditor}
        onPickAudio={props.onPickAudio}
        onPickVideo={props.onPickVideo}
        onPickImage={props.onPickImage}
        onSave={props.onSaveCard}
      />
      <DeckEditorModal
        open={props.deckEditorOpen}
        editingDeck={props.editingDeck}
        value={props.deckDraft}
        onChange={props.onChangeDeckDraft}
        onClose={props.onCloseDeckEditor}
        onSave={props.onSaveDeck}
      />
      <ConfirmModal
        open={props.confirmOpen}
        title={props.confirmTitle}
        message={props.confirmMessage}
        confirmLabel="Удалить"
        onClose={props.onCloseConfirm}
        onConfirm={props.onConfirm}
      />
      {props.sessionSetupOpen && props.pendingSessionDeck && (
        <SessionSetupDialog
          deckName={props.pendingSessionDeck.name}
          stats={props.sessionDeckStats}
          onStart={props.onConfirmSessionSetup}
          onCancel={props.onCloseSessionSetup}
        />
      )}
    </>
  );
}
