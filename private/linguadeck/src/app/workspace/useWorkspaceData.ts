import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AppSettingsService } from "../../services/db/AppSettingsService";
import { CardService } from "../../services/db/CardService";
import { DeckService } from "../../services/db/DeckService";
import { LogService } from "../../services/db/LogService";
import type {
  AppLogEntry,
  AppSettings,
  Card,
  DashboardStats,
  Deck,
} from "../../types/domain";

interface UseWorkspaceDataOptions {
  selectedDeckId: number | null;
}

export function useWorkspaceData({ selectedDeckId }: UseWorkspaceDataOptions) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [stats, setStats]       = useState<DashboardStats | null>(null);
  const [decks, setDecks]       = useState<Deck[]>([]);
  const [cards, setCards]       = useState<Card[]>([]);
  const [logs, setLogs]         = useState<AppLogEntry[]>([]);

  const loadingRef = useRef(false);
  const selectedDeckIdRef = useRef(selectedDeckId);
  selectedDeckIdRef.current = selectedDeckId;

  // FIX: убрано принудительное nativeLanguage: "Russian" — ранее оно перезаписывало
  // любое сохранённое значение языка при каждой загрузке воркспейса.
  // Теперь используем данные как есть из БД, с фоллбэком только если поле пустое.
  function normalizeSettings(loaded: AppSettings): AppSettings {
    return {
      ...loaded,
      nativeLanguage: loaded.nativeLanguage || "Russian",
    };
  }

  const selectedDeck = useMemo(
    () => decks.find((d) => d.id === selectedDeckId) ?? null,
    [decks, selectedDeckId]
  );

  const reloadStats = useCallback(async () => {
    const [newStats, newDecks] = await Promise.all([
      DeckService.getDashboardStats(),
      DeckService.getAll(),
    ]);
    setStats(newStats);
    setDecks(newDecks);
  }, []);

  const reloadCards = useCallback(async (deckId: number | null) => {
    if (!deckId) { setCards([]); return; }
    const loaded = await CardService.getByDeckId(deckId);
    setCards(loaded);
  }, []);

  async function loadWorkspace() {
    if (loadingRef.current) return;
    loadingRef.current = true;
    try {
      const [loadedSettings, loadedStats, loadedDecks, loadedLogs] =
        await Promise.all([
          AppSettingsService.get(),
          DeckService.getDashboardStats(),
          DeckService.getAll(),
          LogService.getRecent(120),
        ]);

      setSettings(normalizeSettings(loadedSettings));
      setStats(loadedStats);
      setDecks(loadedDecks);
      setLogs(loadedLogs);

      const deckId = selectedDeckIdRef.current;
      if (deckId) {
        setCards(await CardService.getByDeckId(deckId));
      } else {
        setCards([]);
      }
    } finally {
      loadingRef.current = false;
    }
  }

  useEffect(() => {
    loadWorkspace().catch(() => undefined);
  }, []);

  useEffect(() => {
    reloadCards(selectedDeckId).catch(() => undefined);
  }, [selectedDeckId]);

  return {
    settings,
    setSettings,
    stats,
    decks,
    selectedDeck,
    setDecks,
    cards,
    setCards,
    logs,
    loadWorkspace,
    reloadStats,
    reloadCards,
  };
}
