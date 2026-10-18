import type { WorkspaceConfirmState } from "./useWorkspaceConfirm";

interface UseConfirmActionOptions {
  confirmState: WorkspaceConfirmState;
  closeConfirm: () => void;
  deleteDeck: (deckId: number) => Promise<void>;
  deleteCard: (cardId: number) => Promise<void>;
  deleteSelectedCards: () => Promise<void>;
}

export function useConfirmAction({
  confirmState,
  closeConfirm,
  deleteDeck,
  deleteCard,
  deleteSelectedCards
}: UseConfirmActionOptions) {
  function confirmTitle() {
    if (confirmState?.kind === "deck") return "Удалить колоду";
    if (confirmState?.kind === "cards") return "Удалить карточки";
    return "Удалить карточку";
  }

  function confirmMessage() {
    if (confirmState?.kind === "deck") {
      return "Колода, все её карточки и история повторений будут удалены.";
    }
    if (confirmState?.kind === "cards") {
      return "Все выбранные карточки будут удалены.";
    }
    return "Карточка будет удалена без возможности восстановления.";
  }

  function onConfirm() {
    if (confirmState?.kind === "deck") {
      deleteDeck(confirmState.id).then(closeConfirm).catch(() => undefined);
    }
    if (confirmState?.kind === "card") {
      deleteCard(confirmState.id).then(closeConfirm).catch(() => undefined);
    }
    if (confirmState?.kind === "cards") {
      deleteSelectedCards().then(closeConfirm).catch(() => undefined);
    }
  }

  return {
    confirmTitle,
    confirmMessage,
    onConfirm
  };
}
