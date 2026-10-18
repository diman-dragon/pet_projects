// src/hooks/useSearchHistory.ts
// Хук для истории поиска из search_history (kb_main).
// Поддерживает удаление по одному запросу и очистку всей истории.

import { useCallback, useEffect, useState } from 'react';
import { ConnectionManager } from '../services/db/ConnectionManager';

export interface SearchHistoryItem {
  query: string;
  ts: string;
}

const MAX_ITEMS = 8;

async function fetchHistory(): Promise<SearchHistoryItem[]> {
  if (!ConnectionManager.exists('kb_main')) return [];
  try {
    const db = ConnectionManager.get('kb_main');
    const rows = await db.select<Array<{ query: string; created_at: string }>>(
      `SELECT query, MAX(created_at) as created_at
       FROM search_history
       WHERE LENGTH(query) > 1
       GROUP BY query
       ORDER BY created_at DESC
       LIMIT $1`,
      [MAX_ITEMS],
    );
    return rows.map((r) => ({ query: r.query, ts: r.created_at }));
  } catch {
    return [];
  }
}

export function useSearchHistory(active: boolean) {
  const [history, setHistory] = useState<SearchHistoryItem[]>([]);

  useEffect(() => {
    if (!active) return;
    fetchHistory().then(setHistory).catch(() => setHistory([]));
  }, [active]);

  const refresh = useCallback(() => {
    fetchHistory().then(setHistory).catch(() => setHistory([]));
  }, []);

  const deleteOne = useCallback(async (query: string) => {
    if (!ConnectionManager.exists('kb_main')) return;
    try {
      const db = ConnectionManager.get('kb_main');
      await db.execute(`DELETE FROM search_history WHERE query = $1`, [query]);
      setHistory((prev) => prev.filter((h) => h.query !== query));
    } catch { /* ignore */ }
  }, []);

  const clearAll = useCallback(async () => {
    if (!ConnectionManager.exists('kb_main')) return;
    try {
      const db = ConnectionManager.get('kb_main');
      await db.execute(`DELETE FROM search_history`);
      setHistory([]);
    } catch { /* ignore */ }
  }, []);

  return { history, refresh, deleteOne, clearAll };
}
