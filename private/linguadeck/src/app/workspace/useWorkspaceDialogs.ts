import { useState } from "react";
import type { CreateCardInput } from "../../services/db/CardService";
import type { Deck } from "../../types/domain";
import { EMPTY_CARD } from "./defaults";
import type { DeckDraft } from "./useDecksController";
import type { SessionConfig } from "../../components/study/SessionSetupDialog";

export function useWorkspaceDialogs() {
  const [cardEditorOpen, setCardEditorOpen] = useState(false);
  const [editingCardId, setEditingCardId] = useState<number | null>(null);
  const [cardDraft, setCardDraft] = useState<CreateCardInput>({ ...EMPTY_CARD });

  const [deckEditorOpen, setDeckEditorOpen] = useState(false);
  const [editingDeck, setEditingDeck] = useState<Deck | null>(null);
  const [deckDraft, setDeckDraft] = useState<DeckDraft>({
    name: "",
    description: "",
    language: "",
  });

  const [sessionSetupOpen, setSessionSetupOpen] = useState(false);
  const [pendingSessionDeck, setPendingSessionDeck] = useState<Deck | null>(null);
  const [sessionConfig, setSessionConfig] = useState<SessionConfig | null>(null);

  const openSessionSetup = (deck: Deck) => {
    setPendingSessionDeck(deck);
    setSessionSetupOpen(true);
  };

  const closeSessionSetup = () => {
    setSessionSetupOpen(false);
    setPendingSessionDeck(null);
  };

  const confirmSessionSetup = (config: SessionConfig) => {
    setSessionConfig(config);
    setSessionSetupOpen(false);
  };

  return {
    cardEditorOpen,
    setCardEditorOpen,
    editingCardId,
    setEditingCardId,
    cardDraft,
    setCardDraft,
    deckEditorOpen,
    setDeckEditorOpen,
    editingDeck,
    setEditingDeck,
    deckDraft,
    setDeckDraft,
    sessionSetupOpen,
    pendingSessionDeck,
    sessionConfig,
    setSessionConfig,
    openSessionSetup,
    closeSessionSetup,
    confirmSessionSetup,
  };
}
