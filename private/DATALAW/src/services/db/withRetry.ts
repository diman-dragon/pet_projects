// src/services/db/withRetry.ts
// Утилита для повторных попыток операций с БД при SQLITE_BUSY (database is locked)

import { createLogger } from '../logger/Logger';

const log = createLogger('withRetry');

export interface RetryOptions {
  maxAttempts?: number;      // максимум попыток (default: 3)
  baseDelayMs?: number;      // начальная задержка (default: 100)
  maxDelayMs?: number;       // максимальная задержка (default: 2000)
  onRetry?: (error: Error, attempt: number, nextDelay: number) => void;
}

/**
 * Выполняет асинхронную функцию с повторными попытками при ошибке SQLITE_BUSY.
 * Использует exponential backoff: delay = baseDelay * 2^attempt
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {},
): Promise<T> {
  const { maxAttempts = 3, baseDelayMs = 100, maxDelayMs = 2000, onRetry } = options;

  let lastError: Error | undefined;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Проверяем, что это ошибка блокировки БД
      const isBusyError =
        lastError.message.includes('database is locked') ||
        lastError.message.includes('busy') ||
        lastError.message.includes('SQLITE_BUSY');

      if (!isBusyError) {
        throw lastError; // Не retry-ошибка — пробрасываем сразу
      }

      if (attempt === maxAttempts - 1) {
        log.error('[❌ DB] Max retry attempts reached:', { error: lastError.message, attempts: maxAttempts });
        throw lastError;
      }

      // Экспоненциальный backoff с jitter
      const delay = Math.min(baseDelayMs * Math.pow(2, attempt), maxDelayMs);
      const jitteredDelay = delay + Math.random() * 100;

      log.warn('[⚠️ DB] Database busy, retrying...', { attempt: attempt + 1, nextDelayMs: Math.round(jitteredDelay) });

      if (onRetry) {
        onRetry(lastError, attempt + 1, Math.round(jitteredDelay));
      }

      await sleep(jitteredDelay);
    }
  }

  throw lastError || new Error('Retry failed');
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
