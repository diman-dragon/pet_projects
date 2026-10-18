// src/hooks/useChatSessions.ts
// Хук для списка сохранённых сессий чата Лайлы из chat_sessions (user_system).
// Позволяет загружать сессию по ID, удалять одну или все.

import { useCallback, useEffect, useState } from 'react';
import { ConnectionManager } from '../services/db/ConnectionManager';
import { MEMORY_DB_KEY } from '../services/ai/memory/constants';
import type { ChatMessageRecord } from '../services/ai/memory/types';

export interface ChatSessionSummary {
  id: string;
  title: string | null;
  origin: string;
  lastMessageAt: string | null;
  createdAt: string;
  preview: string; // первое user-сообщение или пусто
  messageCount: number;
}

const MAX_SESSIONS = 30;

async function fetchSessions(originFilter?: 'chat' | 'widget'): Promise<ChatSessionSummary[]> {
  if (!ConnectionManager.exists(MEMORY_DB_KEY)) return [];
  try {
    const db = ConnectionManager.get(MEMORY_DB_KEY);
    const whereClause = originFilter ? `WHERE s.origin = '${originFilter}'` : '';
    const rows = await db.select<Array<{
      id: string;
      title: string | null;
      origin: string;
      last_message_at: string | null;
      created_at: string;
      preview: string | null;
      message_count: number;
    }>>(
      `SELECT s.id, s.title, s.origin, s.last_message_at, s.created_at,
              (SELECT content FROM chat_messages
               WHERE session_id = s.id AND role = 'user'
               ORDER BY message_index ASC LIMIT 1) as preview,
              (SELECT COUNT(*) FROM chat_messages WHERE session_id = s.id) as message_count
       FROM chat_sessions s
       ${whereClause}
       ORDER BY COALESCE(s.last_message_at, s.created_at) DESC
       LIMIT $1`,
      [MAX_SESSIONS],
    );
    return rows
      .filter((r) => r.message_count > 0)
      .map((r) => ({
        id: r.id,
        title: r.title,
        origin: r.origin,
        lastMessageAt: r.last_message_at,
        createdAt: r.created_at,
        preview: r.preview ?? '',
        messageCount: r.message_count,
      }));
  } catch {
    return [];
  }
}

async function loadSessionMessages(sessionId: string): Promise<ChatMessageRecord[]> {
  if (!ConnectionManager.exists(MEMORY_DB_KEY)) return [];
  try {
    const db = ConnectionManager.get(MEMORY_DB_KEY);
    return await db.select<ChatMessageRecord[]>(
      `SELECT id, session_id, agent_id, message_index, role, content, token_estimate, created_at
       FROM chat_messages
       WHERE session_id = $1
       ORDER BY message_index ASC`,
      [String(sessionId)],
    );
  } catch {
    return [];
  }
}

async function deleteSession(sessionId: string): Promise<void> {
  if (!ConnectionManager.exists(MEMORY_DB_KEY)) return;
  const db = ConnectionManager.get(MEMORY_DB_KEY);
  await db.execute(`DELETE FROM chat_messages WHERE session_id = $1`, [String(sessionId)]);
  await db.execute(`DELETE FROM chat_summaries WHERE session_id = $1`, [String(sessionId)]);
  await db.execute(`DELETE FROM chat_sessions WHERE id = $1`, [String(sessionId)]);
}

async function deleteAllSessions(originFilter?: 'chat' | 'widget'): Promise<void> {
  if (!ConnectionManager.exists(MEMORY_DB_KEY)) return;
  const db = ConnectionManager.get(MEMORY_DB_KEY);
  const whereClause = originFilter ? `WHERE origin = '${originFilter}'` : '';
  const sessions = await db.select<Array<{ id: string }>>(
    `SELECT id FROM chat_sessions ${whereClause}`,
  );
  for (const s of sessions) {
    await db.execute(`DELETE FROM chat_messages WHERE session_id = $1`, [s.id]);
    await db.execute(`DELETE FROM chat_summaries WHERE session_id = $1`, [s.id]);
  }
  await db.execute(`DELETE FROM chat_sessions ${whereClause}`);
}

export function useChatSessions(active: boolean, originFilter?: 'chat' | 'widget') {
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchSessions(originFilter)
      .then(setSessions)
      .catch(() => setSessions([]))
      .finally(() => setLoading(false));
  }, [originFilter]);

  useEffect(() => {
    if (!active) return;
    refresh();
  }, [active, refresh]);

  const loadMessages = useCallback(
    (sessionId: string) => loadSessionMessages(sessionId),
    [],
  );

  const deleteOne = useCallback(
    async (sessionId: string) => {
      await deleteSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    },
    [],
  );

  const clearAll = useCallback(async () => {
    await deleteAllSessions(originFilter);
    setSessions([]);
  }, [originFilter]);

  return { sessions, loading, refresh, loadMessages, deleteOne, clearAll };
}
