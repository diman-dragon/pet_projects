// src/services/search/SearchCache.ts
//
// ИТЕРАЦИЯ 4: Двухуровневый кеш результатов глобального поиска.
//
// L1 — точный запрос + нормализованные параметры (exact match)
// L2 — нормализованный запрос (normalizedQuery из QueryProcessor)
//
// Инвалидация:
//   - По TTL (30 секунд для L1, 60 секунд для L2)
//   - По событию мутации (invalidateOnMutation) — вызывать при:
//       * добавлении/удалении документов из library
//       * изменении clients/cases/appeals
//
// Размер: не более 50 записей в L1, 50 в L2 (LRU eviction).

import type { GlobalSearchResult } from './GlobalSearchService';

const L1_TTL_MS = 30_000;
const L2_TTL_MS = 60_000;
const MAX_ENTRIES = 50;

interface CacheEntry<T> {
  value: T;
  ts: number;
}

class LRUCache<T> {
  private map = new Map<string, CacheEntry<T>>();
  private readonly ttl: number;
  private readonly max: number;

  constructor(ttl: number, max: number) {
    this.ttl = ttl;
    this.max = max;
  }

  get(key: string): T | null {
    const entry = this.map.get(key);
    if (!entry) return null;
    if (Date.now() - entry.ts > this.ttl) {
      this.map.delete(key);
      return null;
    }
    // LRU: move to end
    this.map.delete(key);
    this.map.set(key, entry);
    return entry.value;
  }

  set(key: string, value: T): void {
    if (this.map.size >= this.max) {
      // Evict oldest
      const firstKey = this.map.keys().next().value;
      if (firstKey !== undefined) this.map.delete(firstKey);
    }
    this.map.set(key, { value, ts: Date.now() });
  }

  invalidate(): void {
    this.map.clear();
  }

  size(): number {
    return this.map.size;
  }
}

const l1 = new LRUCache<GlobalSearchResult[]>(L1_TTL_MS, MAX_ENTRIES);
const l2 = new LRUCache<GlobalSearchResult[]>(L2_TTL_MS, MAX_ENTRIES);

function l1Key(query: string): string {
  return query.trim().toLowerCase();
}

function l2Key(normalizedQuery: string): string {
  return `norm:${normalizedQuery}`;
}

export const SearchCache = {
  /**
   * Ищет в L1 (точный запрос), затем в L2 (нормализованный).
   */
  get(query: string, normalizedQuery?: string): GlobalSearchResult[] | null {
    const exact = l1.get(l1Key(query));
    if (exact) return exact;
    if (normalizedQuery) {
      return l2.get(l2Key(normalizedQuery));
    }
    return null;
  },

  /**
   * Сохраняет в оба уровня.
   */
  set(query: string, normalizedQuery: string, results: GlobalSearchResult[]): void {
    l1.set(l1Key(query), results);
    l2.set(l2Key(normalizedQuery), results);
  },

  /**
   * Вызывать при мутациях данных (добавление/удаление документов, изменение клиентов).
   * Сбрасывает оба уровня кеша.
   */
  invalidateOnMutation(): void {
    l1.invalidate();
    l2.invalidate();
  },

  stats(): { l1Size: number; l2Size: number } {
    return { l1Size: l1.size(), l2Size: l2.size() };
  },
};
