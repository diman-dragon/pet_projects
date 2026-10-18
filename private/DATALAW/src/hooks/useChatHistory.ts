// src/hooks/useChatHistory.ts
// Хук для истории сообщений пользователя из chat_messages (user_system).
// Поддерживает удаление по одному и очистку всей истории.

import { useCallback, useEffect, useState } from 'react';
import { ConnectionManager } from '../services/db/ConnectionManager';
import { MEMORY_DB_KEY } from '../services/ai/memory/constants';

export interface ChatHistoryItem {
  content: string;
  ts: string;
}

const MAX_ITEMS = 8;

async function fetchChatHistory(): Promise<ChatHistoryItem[]> {
  if (!ConnectionManager.exists(MEMORY_DB_KEY)) return [];
  try {
    const db = ConnectionManager.get(MEMORY_DB_KEY);
    const rows = await db.select<Array<{ content: string; created_at: string }>>(
      `SELECT content, MAX(created_at) as created_at
       FROM chat_messages
       WHERE role = 'user'
         AND LENGTH(content) > 1
         AND LENGTH(content) < 200
       GROUP BY content
       ORDER BY created_at DESC
       LIMIT $1`,
      [MAX_ITEMS],
    );
    return rows.map((r) => ({ content: r.content, ts: r.created_at }));
  } catch {
    return [];
  }
}

export function useChatHistory(active: boolean) {
  const [history, setHistory] = useState<ChatHistoryItem[]>([]);

  useEffect(() => {
    if (!active) return;
    fetchChatHistory().then(setHistory).catch(() => setHistory([]));
  }, [active]);

  const refresh = useCallback(() => {
    fetchChatHistory().then(setHistory).catch(() => setHistory([]));
  }, []);

  const deleteOne = useCallback(async (content: string) => {
    if (!ConnectionManager.exists(MEMORY_DB_KEY)) return;
    try {
      const db = ConnectionManager.get(MEMORY_DB_KEY);
      await db.execute(
        `DELETE FROM chat_messages WHERE role = 'user' AND content = $1`,
        [content],
      );
      setHistory((prev) => prev.filter((h) => h.content !== content));
    } catch { /* ignore */ }
  }, []);

  const clearAll = useCallback(async () => {
    if (!ConnectionManager.exists(MEMORY_DB_KEY)) return;
    try {
      const db = ConnectionManager.get(MEMORY_DB_KEY);
      await db.execute(`DELETE FROM chat_messages WHERE role = 'user'`);
      setHistory([]);
    } catch { /* ignore */ }
  }, []);

  return { history, refresh, deleteOne, clearAll };
}
