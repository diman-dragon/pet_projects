// src/services/search/FolderCategorizerService.ts
//
// Определяет категорию документа на основе его пути к файлу.
// Правила загружаются из таблицы category_rules (kb_main) и кешируются
// в памяти с event-driven инвалидацией.
//
// Приоритет правил:
//   1. path_pattern — LIKE-совпадение по нормализованному пути (приоритет выше)
//   2. extension    — по расширению файла
//   3. default      — если ничего не подошло
//
// Если category_overridden = true в metadata — категория не меняется.

import { ConnectionManager } from '../db/ConnectionManager';
import { createLogger } from '../logger/Logger';

const log = createLogger('FolderCategorizerService');

interface CategoryRule {
  rule_type: 'path_pattern' | 'extension' | 'default';
  pattern: string;
  category: string;
  priority: number;
}

// Простой in-memory кеш правил
let rulesCache: CategoryRule[] | null = null;
let cacheLoadedAt = 0;
const CACHE_TTL_MS = 60_000; // 1 минута — достаточно для статичных правил

/**
 * Нормализует путь для сравнения: слэши → /, нижний регистр.
 */
function normalizePath(p: string): string {
  return p.replace(/\\/g, '/').toLowerCase();
}

/**
 * Простая glob-like проверка LIKE-паттерна SQLite (%...%).
 * Не поддерживает _ (одиночный символ) — в наших правилах он не используется.
 */
function likeMatch(pattern: string, value: string): boolean {
  const normalized = pattern.toLowerCase();
  const parts = normalized.split('%').filter(Boolean);
  let pos = 0;
  for (const part of parts) {
    const idx = value.indexOf(part, pos);
    if (idx === -1) return false;
    pos = idx + part.length;
  }
  return true;
}

async function loadRules(): Promise<CategoryRule[]> {
  const now = Date.now();
  if (rulesCache && now - cacheLoadedAt < CACHE_TTL_MS) {
    return rulesCache;
  }

  if (!ConnectionManager.exists('kb_main')) {
    log.warn('kb_main not open, cannot load category_rules');
    return [];
  }

  try {
    const db = ConnectionManager.get('kb_main');
    const rows = await db.select<CategoryRule[]>(
      `SELECT rule_type, pattern, category, priority
       FROM category_rules
       ORDER BY priority DESC`,
    );
    rulesCache = rows;
    cacheLoadedAt = now;
    log.debug(`Loaded ${rows.length} category rules`);
    return rows;
  } catch (err) {
    log.error('Failed to load category_rules', err);
    return [];
  }
}

/**
 * Инвалидирует кеш правил (вызывать после изменения category_rules).
 */
export function invalidateCategoryRulesCache(): void {
  rulesCache = null;
  cacheLoadedAt = 0;
}

/**
 * Определяет категорию файла по его пути.
 *
 * @param filePath        — абсолютный или относительный путь к файлу
 * @param metadataJson    — JSON строка metadata из library (проверяет category_overridden)
 * @param currentCategory — текущая категория (возвращается если overridden)
 */
export async function categorizeByPath(
  filePath: string,
  metadataJson?: string | null,
  currentCategory?: string | null,
): Promise<string> {
  // Если пользователь вручную переопределил категорию — не трогаем
  if (metadataJson && currentCategory) {
    try {
      const meta = JSON.parse(metadataJson);
      if (meta?.category_overridden === true) {
        return currentCategory;
      }
    } catch {
      // невалидный JSON — игнорируем
    }
  }

  const rules = await loadRules();
  if (!rules.length) return currentCategory ?? 'General';

  const normalizedPath = normalizePath(filePath);
  const ext = normalizedPath.split('.').pop() ?? '';

  // path_pattern — наивысший приоритет
  for (const rule of rules) {
    if (rule.rule_type === 'path_pattern' && likeMatch(rule.pattern, normalizedPath)) {
      return rule.category;
    }
  }

  // extension
  for (const rule of rules) {
    if (rule.rule_type === 'extension' && rule.pattern.toLowerCase() === ext) {
      return rule.category;
    }
  }

  // default
  for (const rule of rules) {
    if (rule.rule_type === 'default') {
      return rule.category;
    }
  }

  return currentCategory ?? 'General';
}

/**
 * Извлекает нормализованный folder_path из полного пути файла.
 * Пример: "/data/clients/ООО Ромашка/договоры/акт.pdf" → "/data/clients/ООО Ромашка/договоры"
 */
export function extractFolderPath(filePath: string): string {
  const normalized = filePath.replace(/\\/g, '/');
  const lastSlash = normalized.lastIndexOf('/');
  return lastSlash > 0 ? normalized.slice(0, lastSlash) : normalized;
}

export const FolderCategorizerService = {
  categorizeByPath,
  extractFolderPath,
  invalidateCache: invalidateCategoryRulesCache,
};
