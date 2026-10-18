import type { Deck } from "../../types/domain";
import { DeckService } from "../../services/db/DeckService";
import { deckPalette } from "./defaults";

export interface DeckDraft {
  name: string;
  description: string;
  language: string;
}

interface UseDecksControllerOptions {
  decks: Deck[];
  editingDeck: Deck | null;
  deckDraft: DeckDraft;
  setEditingDeck: (deck: Deck | null) => void;
  setDeckDraft: (
    value: DeckDraft | ((current: DeckDraft) => DeckDraft)
  ) => void;
  setDeckEditorOpen: (open: boolean) => void;
  loadWorkspace: () => Promise<void>;
  withFeedback: <T>(
    scope: string,
    action: () => Promise<T>,
    successMessage?: string
  ) => Promise<T>;
}

export function useDecksController({
  decks,
  editingDeck,
  deckDraft,
  setEditingDeck,
  setDeckDraft,
  setDeckEditorOpen,
  loadWorkspace,
  withFeedback,
}: UseDecksControllerOptions) {
  function openCreateDeck() {
    setEditingDeck(null);
    setDeckDraft({ name: "", description: "", language: "" });
    setDeckEditorOpen(true);
  }

  function openEditDeck(deck: Deck) {
    setEditingDeck(deck);
    setDeckDraft({
      name: deck.name,
      description: deck.description,
      language: deck.language ?? "",
    });
    setDeckEditorOpen(true);
  }

  async function saveDeck() {
    const name = deckDraft.name.trim();
    if (!name) return;

    await withFeedback(
      "deck",
      async () => {
        if (editingDeck) {
          await DeckService.update(editingDeck.id, {
            name,
            description: deckDraft.description.trim(),
            icon: editingDeck.icon,
            color: editingDeck.color,
            language: deckDraft.language.trim(),
          });
        } else {
          await DeckService.create(
            1,
            name,
            deckDraft.description.trim(),
            "Layers3",
            deckPalette(decks.length),
            deckDraft.language.trim()
          );
        }

        setDeckEditorOpen(false);
        setEditingDeck(null);
        await loadWorkspace();
      },
      editingDeck ? "Колода обновлена." : "Колода создана."
    );
  }

  async function deleteDeck(deckId: number) {
    await withFeedback(
      "deck",
      async () => {
        await DeckService.delete(deckId);
        await loadWorkspace();
      },
      "Колода удалена."
    );
  }

  return {
    openCreateDeck,
    openEditDeck,
    saveDeck,
    deleteDeck,
  };
}
