// src/services/utils/hash.ts
//
// Utility functions for hashing.
// Non-cryptographic, for internal use only.

/**
 * Быстрый короткий hash строки (не криптографический).
 * Используется для анонимного отслеживания запросов без сохранения текста.
 *
 * @param str - Строка для хеширования
 * @returns 7-символьный hash в base36 (0-9, a-z)
 */
export function shortHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(36).padStart(7, '0');
}
