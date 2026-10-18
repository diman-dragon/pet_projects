import { open } from "@tauri-apps/plugin-dialog";
import { readTextFile } from "@tauri-apps/plugin-fs";
import { useState } from "react";
import { CardService, type CreateCardInput } from "../../services/db/CardService";
import type { Card, ImportProgress } from "../../types/domain";
import { EMPTY_CARD } from "./defaults";
import { parseImport } from "./importParser";

interface UseCardsControllerOptions {
  selectedDeckId: number | null;
  selectedCardIds: number[];
  editingCardId: number | null;
  cardDraft: CreateCardInput;
  setEditingCardId: (value: number | null) => void;
  setCardDraft: (
    value:
      | CreateCardInput
      | ((current: CreateCardInput) => CreateCardInput)
  ) => void;
  setCardEditorOpen: (open: boolean) => void;
  setSelectedCardIds: (
    value: number[] | ((current: number[]) => number[])
  ) => void;
  loadWorkspace: () => Promise<void>;
  withFeedback: <T>(
    scope: string,
    action: () => Promise<T>,
    successMessage?: string
  ) => Promise<T>;
}

export function useCardsController({
  selectedDeckId,
  selectedCardIds,
  editingCardId,
  cardDraft,
  setEditingCardId,
  setCardDraft,
  setCardEditorOpen,
  setSelectedCardIds,
  loadWorkspace,
  withFeedback,
}: UseCardsControllerOptions) {
  const [importProgress, setImportProgress] = useState<ImportProgress>({
    total: 0,
    inserted: 0,
    status: "idle",
  });

  function openCreateCard() {
    setEditingCardId(null);
    setCardDraft({ ...EMPTY_CARD });
    setCardEditorOpen(true);
  }

  function openEditCard(card: Card) {
    setEditingCardId(card.id);
    setCardDraft({
      front: card.front,
      back: card.back,
      context: card.context ?? "",
      transcription: card.transcription ?? "",
      etymology: card.etymology ?? "",
      mnemonic: card.mnemonic ?? "",
      notes: card.notes ?? "",
      example_1: card.example_1 ?? "",
      example_2: card.example_2 ?? "",
      example_3: card.example_3 ?? "",
      audio_path: card.audio_path ?? "",
      image_path: card.image_path ?? "",
      video_path: card.video_path ?? "",
      source: card.source ?? "user",
      tags: card.tags ?? "",
    });
    setCardEditorOpen(true);
  }

  async function saveCard() {
    if (!selectedDeckId || !cardDraft.front.trim() || !cardDraft.back.trim()) {
      return;
    }

    await withFeedback(
      "card",
      async () => {
        if (editingCardId) {
          await CardService.update(editingCardId, cardDraft);
        } else {
          await CardService.createExtended(selectedDeckId, cardDraft);
        }

        setCardEditorOpen(false);
        setEditingCardId(null);
        setCardDraft({ ...EMPTY_CARD });
        await loadWorkspace();
      },
      editingCardId ? "Карточка обновлена." : "Карточка создана."
    );
  }

  async function deleteCard(cardId: number) {
    if (!selectedDeckId) return;

    await withFeedback(
      "card",
      async () => {
        await CardService.delete(cardId, selectedDeckId);
        await loadWorkspace();
      },
      "Карточка удалена."
    );
  }

  async function deleteSelectedCards() {
    if (!selectedDeckId || selectedCardIds.length === 0) return;

    await withFeedback(
      "card",
      async () => {
        await CardService.deleteMany(selectedCardIds, selectedDeckId);
        setSelectedCardIds([]);
        await loadWorkspace();
      },
      "Выбранные карточки удалены."
    );
  }

  function toggleCardSelection(cardId: number) {
    setSelectedCardIds((current) =>
      current.includes(cardId)
        ? current.filter((item) => item !== cardId)
        : [...current, cardId]
    );
  }

  async function importCards() {
    if (!selectedDeckId) return;

    const filePath = await open({
      directory: false,
      multiple: false,
      filters: [{ name: "Text Data", extensions: ["txt", "csv", "tsv"] }],
      title: "Импорт карточек",
    });

    if (typeof filePath !== "string") return;

    let rows: CreateCardInput[] = [];
    try {
      const raw = await readTextFile(filePath);
      rows = parseImport(raw);
    } catch (err) {
      setImportProgress({
        total: 0,
        inserted: 0,
        status: "error",
        error: err instanceof Error ? err.message : "Ошибка чтения файла",
      });
      return;
    }

    if (rows.length === 0) {
      setImportProgress({ total: 0, inserted: 0, status: "error", error: "Файл не содержит карточек" });
      return;
    }

    setImportProgress({ total: rows.length, inserted: 0, status: "running" });

    try {
      // Single Rust batch transaction — O(1) IPC instead of O(n)
      const inserted = await CardService.createBatch(selectedDeckId, rows);
      setImportProgress({ total: rows.length, inserted, status: "done" });
      await loadWorkspace();
      // Reset after 4 s
      setTimeout(() => setImportProgress({ total: 0, inserted: 0, status: "idle" }), 4000);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Ошибка импорта";
      setImportProgress({ total: rows.length, inserted: 0, status: "error", error: msg });
    }
  }

  async function pickMedia(field: "audio_path" | "image_path" | "video_path") {
    const titleMap = {
      audio_path: "Выберите аудио",
      image_path: "Выберите изображение",
      video_path: "Выберите видео",
    };

    const value = await open({
      directory: false,
      multiple: false,
      title: titleMap[field],
    });

    if (typeof value === "string") {
      setCardDraft((current) => ({ ...current, [field]: value }));
    }
  }

  return {
    importProgress,
    openCreateCard,
    openEditCard,
    saveCard,
    deleteCard,
    deleteSelectedCards,
    toggleCardSelection,
    importCards,
    pickMedia,
  };
}
