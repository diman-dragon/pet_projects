// src\services\ai\memory\ChatMemoryService.ts

import type { DatabaseLike } from '../../db/ConnectionManager';
import { ConnectionManager } from '../../db/ConnectionManager';
import { createLogger } from '../../logger/Logger';
import { MEMORY_DB_KEY } from './constants';
import type {
  ChatMessageRecord,
  ChatSessionRecord,
  ChatSummaryRecord,
  MemoryAssistantId,
  MemoryMessageRole,
  MemoryOrigin,
} from './types';

const log = createLogger('ChatMemoryService');

function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.trim().length / 4));
}

function getDb(): DatabaseLike | null {
  if (!ConnectionManager.exists(MEMORY_DB_KEY)) {
    return null;
  }

  return ConnectionManager.get(MEMORY_DB_KEY);
}

async function initVecMemory(): Promise<void> {
  const db = getDb();
  if (!db) return;

  try {
    // Create virtual table for semantic memory search using sqlite-vec
    await db.execute(`
      CREATE VIRTUAL TABLE IF NOT EXISTS vec_chat_memory USING vec0(
        record_id INTEGER,
        embedding FLOAT[1024]
      )
    `);
    log.debug('[vec_chat_memory] Virtual table created or already exists');
  } catch (error) {
    log.warn('[vec_chat_memory] Failed to create virtual table:', error);
  }
}

export const ChatMemoryService = {
  estimateTokens,

  async init(): Promise<void> {
    await initVecMemory();
  },

  async ensureSession(
    sessionId: string,
    agentId: MemoryAssistantId,
    origin: MemoryOrigin,
    contextScope: string,
    title?: string,
  ): Promise<void> {
    const db = getDb();
    if (!db) return;

    await db.execute(
      `INSERT INTO chat_sessions
       (id, agent_id, origin, context_scope, title, last_summarized_index, last_message_at, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, 0, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
       ON CONFLICT(id) DO UPDATE SET
         agent_id = excluded.agent_id,
         origin = excluded.origin,
         context_scope = excluded.context_scope,
         title = COALESCE(chat_sessions.title, excluded.title),
         updated_at = CURRENT_TIMESTAMP`,
      [String(sessionId), agentId, origin, contextScope, title ?? null],
    );
  },

  async listSessionMessages(sessionId: string): Promise<ChatMessageRecord[]> {
    const db = getDb();
    if (!db) return [];

    return db.select<ChatMessageRecord[]>(
      `SELECT id, session_id, agent_id, message_index, role, content, token_estimate, created_at
       FROM chat_messages
       WHERE session_id = $1
       ORDER BY message_index ASC`,
      [String(sessionId)],
    );
  },

  async listRecentMessages(sessionId: string, limit: number): Promise<ChatMessageRecord[]> {
    const db = getDb();
    if (!db) return [];

    const rows = await db.select<ChatMessageRecord[]>(
      `SELECT id, session_id, agent_id, message_index, role, content, token_estimate, created_at
       FROM chat_messages
       WHERE session_id = $1
       ORDER BY message_index DESC
       LIMIT $2`,
      [String(sessionId), limit],
    );

    return [...rows].reverse();
  },

  async appendMessage(params: {
    id: string;
    sessionId: string;
    agentId: MemoryAssistantId;
    role: MemoryMessageRole;
    content: string;
  }): Promise<ChatMessageRecord | null> {
    const db = getDb();
    if (!db) return null;

    const [{ next_index: nextIndex } = { next_index: 1 }] = await db.select<
      Array<{ next_index: number }>
    >(
      `SELECT COALESCE(MAX(message_index), 0) + 1 AS next_index
       FROM chat_messages
       WHERE session_id = $1`,
      [String(params.sessionId)],
    );

    const tokenEstimate = estimateTokens(params.content);

    await db.execute(
      `INSERT OR REPLACE INTO chat_messages
       (id, session_id, agent_id, message_index, role, content, token_estimate, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)`,
      [
        params.id,
        String(params.sessionId),
        params.agentId,
        nextIndex,
        params.role,
        params.content,
        tokenEstimate,
      ],
    );

    await db.execute(
      `UPDATE chat_sessions
       SET last_message_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
       WHERE id = $1`,
      [String(params.sessionId)],
    );

    const rows = await db.select<ChatMessageRecord[]>(
      `SELECT id, session_id, agent_id, message_index, role, content, token_estimate, created_at
       FROM chat_messages
       WHERE id = $1
       LIMIT 1`,
      [params.id],
    );

    return rows[0] ?? null;
  },

  async getSession(sessionId: string): Promise<ChatSessionRecord | null> {
    const db = getDb();
    if (!db) return null;

    const rows = await db.select<ChatSessionRecord[]>(
      `SELECT id, agent_id, origin, context_scope, title, last_message_at, last_summarized_index, created_at, updated_at
       FROM chat_sessions
       WHERE id = $1
       LIMIT 1`,
      [String(sessionId)],
    );

    return rows[0] ?? null;
  },

  async listRecentSummaries(sessionId: string, limit: number): Promise<ChatSummaryRecord[]> {
    const db = getDb();
    if (!db) return [];

    const rows = await db.select<ChatSummaryRecord[]>(
      `SELECT id, session_id, agent_id, origin, context_scope, start_message_index, end_message_index, summary_text, token_estimate, created_at, updated_at
       FROM chat_summaries
       WHERE session_id = $1
       ORDER BY end_message_index DESC
       LIMIT $2`,
      [String(sessionId), limit],
    );

    return [...rows].reverse();
  },

  async listRelevantSummaries(params: {
    agentId: MemoryAssistantId;
    sessionId: string;
    origin: MemoryOrigin;
    contextScope: string;
    limit: number;
  }): Promise<ChatSummaryRecord[]> {
    const db = getDb();
    if (!db) return [];

    return db.select<ChatSummaryRecord[]>(
      `SELECT id, session_id, agent_id, origin, context_scope, start_message_index, end_message_index, summary_text, token_estimate, created_at, updated_at
       FROM chat_summaries
       WHERE agent_id = $1
         AND session_id != $2
       ORDER BY
         CASE
           WHEN origin = $3 THEN 0
           WHEN context_scope = $4 THEN 1
           ELSE 2
         END,
         created_at DESC
       LIMIT $5`,
      [params.agentId, String(params.sessionId), params.origin, params.contextScope, params.limit],
    );
  },

  async listMessagesForSummary(sessionId: string): Promise<ChatMessageRecord[]> {
    const db = getDb();
    if (!db) return [];

    const session = await this.getSession(sessionId);
    if (!session) return [];

    return db.select<ChatMessageRecord[]>(
      `SELECT id, session_id, agent_id, message_index, role, content, token_estimate, created_at
       FROM chat_messages
       WHERE session_id = $1
         AND message_index > $2
       ORDER BY message_index ASC`,
      [String(sessionId), session.last_summarized_index],
    );
  },

  async saveSummary(params: {
    sessionId: string;
    agentId: MemoryAssistantId;
    origin: MemoryOrigin;
    contextScope: string;
    startMessageIndex: number;
    endMessageIndex: number;
    summaryText: string;
  }): Promise<number | null> {
    const db = getDb();
    if (!db) return null;

    const tokenEstimate = estimateTokens(params.summaryText);

    await db.execute(
      `INSERT INTO chat_summaries
       (session_id, agent_id, origin, context_scope, start_message_index, end_message_index, summary_text, token_estimate, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)`,
      [
        String(params.sessionId),
        params.agentId,
        params.origin,
        params.contextScope,
        params.startMessageIndex,
        params.endMessageIndex,
        params.summaryText,
        tokenEstimate,
      ],
    );

    const rows = await db.select<Array<{ id: number }>>(
      `SELECT id
       FROM chat_summaries
       WHERE session_id = $1
       ORDER BY id DESC
       LIMIT 1`,
      [String(params.sessionId)],
    );

    await db.execute(
      `UPDATE chat_sessions
       SET last_summarized_index = $2, updated_at = CURRENT_TIMESTAMP
       WHERE id = $1`,
      [String(params.sessionId), params.endMessageIndex],
    );

    return rows[0]?.id ?? null;
  },

  async saveMemoryRecord(params: {
    sessionId: string;
    agentId: MemoryAssistantId;
    origin: MemoryOrigin;
    contextScope: string;
    sourceType: 'message' | 'summary';
    sourceRef: string;
    role: MemoryMessageRole | null;
    text: string;
    tokenEstimate?: number;
    embeddingModel?: string;
  }): Promise<number | null> {
    const db = getDb();
    if (!db) return null;

    await db.execute(
      `INSERT OR REPLACE INTO chat_memory_records
       (session_id, agent_id, origin, context_scope, source_type, source_ref, role, text, token_estimate, embedding_model, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)`,
      [
        String(params.sessionId),
        params.agentId,
        params.origin,
        params.contextScope,
        params.sourceType,
        params.sourceRef,
        params.role,
        params.text,
        params.tokenEstimate ?? estimateTokens(params.text),
        params.embeddingModel ?? null,
      ],
    );

    const rows = await db.select<Array<{ id: number }>>(
      `SELECT id
       FROM chat_memory_records
       WHERE session_id = $1 AND source_type = $2 AND source_ref = $3
       LIMIT 1`,
      [String(params.sessionId), params.sourceType, params.sourceRef],
    );

    return rows[0]?.id ?? null;
  },

  async saveEmbedding(params: {
    recordId: number;
    sessionId: string;
    agentId: MemoryAssistantId;
    model: string;
    dim: number;
    vectorJson: string;
  }): Promise<void> {
    const db = getDb();
    if (!db) return;

    await db.execute(
      `INSERT OR REPLACE INTO chat_memory_embeddings
       (record_id, session_id, agent_id, model, dim, vector_blob, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)`,
      [
        params.recordId,
        String(params.sessionId),
        params.agentId,
        params.model,
        params.dim,
        params.vectorJson,
      ],
    );

    try {
      // vec_chat_memory (sqlite-vec) поддерживает только INTEGER колонки —
      // session_id, agent_id, model хранятся в chat_memory_records и берутся через JOIN
      const idAsNumber = Number(params.recordId);
      if (isNaN(idAsNumber)) {
        log.warn('vec_chat_memory skipped: recordId is not a number', params.recordId);
        return;
      }
      await db.execute(
        `INSERT OR REPLACE INTO vec_chat_memory (record_id, embedding)
         VALUES ($1, $2)`,
        [idAsNumber, params.vectorJson],
      );
    } catch (error) {
      log.warn('vec_chat_memory unavailable, semantic memory will use scalar fallback', error);
    }
  },
};
