import { useCallback, useEffect, useRef } from "react";
import type { AppSettings } from "../../types/domain";
import { AppSettingsService } from "../../services/db/AppSettingsService";
import { LogService } from "../../services/db/LogService";

interface UseSettingsControllerOptions {
  settings: AppSettings | null;
  loadWorkspace: () => Promise<void>;
  withFeedback: <T>(
    scope: string,
    action: () => Promise<T>,
    successMessage?: string
  ) => Promise<T>;
}

export function useSettingsController({
  settings,
  loadWorkspace,
  withFeedback,
}: UseSettingsControllerOptions) {
  const settingsRef = useRef(settings);
  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, []);

  const debouncedSave = useCallback(() => {
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    saveTimeoutRef.current = setTimeout(async () => {
      const currentSettings = settingsRef.current;
      if (!currentSettings) return;

      await withFeedback(
        "settings",
        async () => {
          // FIX: убрано принудительное nativeLanguage: "Russian" —
          // ранее любое изменение языка пользователем сбрасывалось при сохранении
          await AppSettingsService.save(currentSettings);
          await loadWorkspace();
        },
        "Настройки сохранены."
      );
    }, 500);
  }, [loadWorkspace, withFeedback]);

  async function clearLogs() {
    await withFeedback(
      "logs",
      async () => {
        await LogService.clear();
        await loadWorkspace();
      },
      "Логи очищены."
    );
  }

  return {
    debouncedSave,
    clearLogs,
  };
}
