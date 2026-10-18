// src/services/search/SearchSuggestionsService.ts
//
// ИТЕРАЦИЯ 4: Подсказки поиска без AI.
//
// Источники (в порядке приоритета):
//   1. search_history  — недавние запросы пользователя
//   2. search_selections — ранее выбранные результаты
//   3. prefix lookup по clients_fts — имена клиентов
//   4. популярные titles документов из library_fts
//
// API:
//   getSuggestions(prefix)  — debounce снаружи (120ms), возвращает ≤6 вариантов
//   getRecentActivity()     — пустой ввод: недавние запросы + открытые сущности

import { ConnectionManager } from '../db/ConnectionManager';
import { createLogger } from '../logger/Logger';

const log = createLogger('SearchSuggestionsService');

export interface SearchSuggestion {
  id: string;
  text: string;
  type: 'history' | 'client' | 'document' | 'recent_entity';
  subtitle?: string;
}

// Простой LRU-кеш для быстрых prefix-запросов
const prefixCache = new Map<string, { suggestions: SearchSuggestion[]; ts: number }>();
const PREFIX_CACHE_TTL_MS = 5_000;
const MAX_SUGGESTIONS = 6;

function normPrefix(s: string): string {
  return s.trim().toLowerCase();
}

function pruneCache(): void {
  const now = Date.now();
  for (const [k, v] of prefixCache) {
    if (now - v.ts > PREFIX_CACHE_TTL_MS) prefixCache.delete(k);
  }
}

/**
 * Сбрасывает кеш подсказок (вызывать при мутациях clients/library).
 */
export function invalidateSuggestionsCache(): void {
  prefixCache.clear();
}

async function getHistorySuggestions(prefix: string): Promise<SearchSuggestion[]> {
  if (!ConnectionManager.exists('kb_main')) return [];
  try {
    const db = ConnectionManager.get('kb_main');
    const term = `${prefix}%`;
    const rows = await db.select<Array<{ query: string; cnt: number }>>(
      `SELECT query, COUNT(*) as cnt
       FROM search_history
       WHERE query LIKE $1 ESCAPE '\\'
         AND LENGTH(query) > 1
       GROUP BY query
       ORDER BY MAX(created_at) DESC
       LIMIT 4`,
      [term],
    );
    return rows.map((r) => ({
      id: `hist:${r.query}`,
      text: r.query,
      type: 'history' as const,
      subtitle: 'Недавний запрос',
    }));
  } catch {
    return [];
  }
}

async function getClientSuggestions(prefix: string): Promise<SearchSuggestion[]> {
  if (!ConnectionManager.exists('clients_main')) return [];
  try {
    const db = ConnectionManager.get('clients_main');
    // FTS prefix search
    const ftsExpr = `"${prefix.replace(/"/g, '""')}"*`;
    const rows = await db.select<Array<{ id: number; full_name: string }>>(
      `SELECT c.id, c.full_name
       FROM clients_fts
       JOIN clients c ON c.id = clients_fts.rowid
       WHERE clients_fts MATCH $1
         AND c.deleted_at IS NULL
       ORDER BY rank
       LIMIT 4`,
      [ftsExpr],
    );
    return rows.map((r) => ({
      id: `client:${r.id}`,
      text: r.full_name,
      type: 'client' as const,
      subtitle: 'Клиент',
    }));
  } catch {
    return [];
  }
}

async function getDocumentSuggestions(prefix: string): Promise<SearchSuggestion[]> {
  if (!ConnectionManager.exists('kb_main')) return [];
  try {
    const db = ConnectionManager.get('kb_main');
    const ftsExpr = `"${prefix.replace(/"/g, '""')}"*`;
    const rows = await db.select<Array<{ id: number; title: string; category: string | null }>>(
      `SELECT l.id, l.title, l.category
       FROM library_fts
       JOIN library l ON l.id = library_fts.rowid
       WHERE library_fts MATCH $1
         AND l.file_path NOT LIKE 'virtual://%'
       ORDER BY rank
       LIMIT 4`,
      [ftsExpr],
    );
    return rows.map((r) => ({
      id: `doc:${r.id}`,
      text: r.title,
      type: 'document' as const,
      subtitle: r.category ?? 'База знаний',
    }));
  } catch {
    return [];
  }
}

async function getRecentHistoryItems(): Promise<SearchSuggestion[]> {
  if (!ConnectionManager.exists('kb_main')) return [];
  try {
    const db = ConnectionManager.get('kb_main');
    const rows = await db.select<Array<{ query: string }>>(
      `SELECT DISTINCT query
       FROM search_history
       WHERE LENGTH(query) > 1
       ORDER BY created_at DESC
       LIMIT 6`,
    );
    return rows.map((r) => ({
      id: `hist:${r.query}`,
      text: r.query,
      type: 'history' as const,
      subtitle: 'Недавний запрос',
    }));
  } catch {
    return [];
  }
}

async function getRecentSelectionItems(): Promise<SearchSuggestion[]> {
  if (!ConnectionManager.exists('kb_main')) return [];
  try {
    const db = ConnectionManager.get('kb_main');
    const rows = await db.select<Array<{ result_id: string }>>(
      `SELECT DISTINCT result_id
       FROM search_selections
       ORDER BY selected_at DESC
       LIMIT 4`,
    );
    // Разрешаем result_id → читаемые названия
    const results: SearchSuggestion[] = [];
    for (const r of rows) {
      const m = r.result_id.match(/^(client|case|appeal|lib)-(\d+)$/);
      if (!m) continue;
      const [, type, rawId] = m;
      const id = parseInt(rawId, 10);
      try {
        if ((type === 'client') && ConnectionManager.exists('clients_main')) {
          const db2 = ConnectionManager.get('clients_main');
          const [row] = await db2.select<Array<{ full_name: string }>>(
            `SELECT full_name FROM clients WHERE id = $1 AND deleted_at IS NULL`, [id],
          );
          if (row) results.push({ id: `sel:${r.result_id}`, text: row.full_name, type: 'recent_entity', subtitle: 'Клиент' });
        } else if ((type === 'lib') && ConnectionManager.exists('kb_main')) {
          const db2 = ConnectionManager.get('kb_main');
          const [row] = await db2.select<Array<{ title: string }>>(
            `SELECT title FROM library WHERE id = $1`, [id],
          );
          if (row) results.push({ id: `sel:${r.result_id}`, text: row.title, type: 'recent_entity', subtitle: 'Документ' });
        }
      } catch { /* skip */ }
    }
    return results;
  } catch {
    return [];
  }
}

export const SearchSuggestionsService = {
  /**
   * Возвращает подсказки для непустого prefix (≥2 символа).
   * Debounce 120ms должен применяться снаружи.
   */
  async getSuggestions(prefix: string): Promise<SearchSuggestion[]> {
    const norm = normPrefix(prefix);
    if (norm.length < 2) return [];

    pruneCache();
    const cached = prefixCache.get(norm);
    if (cached) return cached.suggestions;

    try {
      const [history, clients, docs] = await Promise.allSettled([
        getHistorySuggestions(norm),
        getClientSuggestions(norm),
        getDocumentSuggestions(norm),
      ]);

      const merged: SearchSuggestion[] = [];
      const seen = new Set<string>();

      const add = (items: SearchSuggestion[]) => {
        for (const s of items) {
          const key = s.text.toLowerCase();
          if (!seen.has(key)) {
            seen.add(key);
            merged.push(s);
          }
        }
      };

      if (history.status === 'fulfilled') add(history.value);
      if (clients.status === 'fulfilled') add(clients.value);
      if (docs.status === 'fulfilled') add(docs.value);

      const result = merged.slice(0, MAX_SUGGESTIONS);
      prefixCache.set(norm, { suggestions: result, ts: Date.now() });
      return result;
    } catch (err) {
      log.warn('getSuggestions failed', err);
      return [];
    }
  },

  /**
   * Для пустого ввода: недавние запросы + недавно открытые сущности.
   */
  async getRecentActivity(): Promise<SearchSuggestion[]> {
    try {
      const [history, selections] = await Promise.allSettled([
        getRecentHistoryItems(),
        getRecentSelectionItems(),
      ]);

      const merged: SearchSuggestion[] = [];
      const seen = new Set<string>();

      if (history.status === 'fulfilled') {
        for (const s of history.value) {
          if (!seen.has(s.text)) { seen.add(s.text); merged.push(s); }
        }
      }
      if (selections.status === 'fulfilled') {
        for (const s of selections.value) {
          if (!seen.has(s.text)) { seen.add(s.text); merged.push(s); }
        }
      }

      return merged.slice(0, MAX_SUGGESTIONS);
    } catch {
      return [];
    }
  },
};
