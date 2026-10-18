// hooks/useAIStatus.ts
// [3.3] Оптимизация: интервал опроса зависит от профиля машины (ModelContextProfile)

import { useEffect, useCallback, useRef } from 'react';
import { useOllamaStore } from '../../../store/ollamaStore';
import { getAIPollingIntervalMs } from '../../../services/ai/ModelContextProfile';

/**
 * Проверяет доступность Ollama при монтировании и периодически опрашивает статус.
 * Интервал определяется профилем машины через getAIPollingIntervalMs():
 *   minimal → 30 сек, light → 20 сек, balanced → 15 сек, full → 10 сек.
 */
export function useAIStatus() {
  const { sidecarStatus, refreshStatus } = useOllamaStore();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const check = useCallback(async () => {
    await refreshStatus();
  }, [refreshStatus]);

  useEffect(() => {
    // Немедленная проверка при монтировании
    check();

    // [3.3] Адаптивный интервал из ModelContextProfile
    const ms = getAIPollingIntervalMs();
    intervalRef.current = setInterval(() => {
      check();
    }, ms);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [check]);

  return { aiReady: sidecarStatus.running, recheckAI: check };
}
