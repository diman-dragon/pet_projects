import type { DatabaseLike } from '../../db/ConnectionManager';
import { EmbeddingService } from '../../rag/EmbeddingService';
import { createLogger } from '../../logger/Logger';
import { ConnectionManager } from '../../db/ConnectionManager';
import {
  getMemoryConstants,
  getMemoryEmbedModel,
  MEMORY_DB_KEY,
} from './constants';
import { ChatMemoryService } from './ChatMemoryService';
import { useOllamaStore } from '../../../store/ollamaStore';
import type {
  MemoryAssistantId,
  MemoryContextBundle,
  MemoryOrigin,
  MemorySearchResult,
} from './types';

const log = createLogger('MemoryContextService');

function getDb(): DatabaseLike | null {
  if (!ConnectionManager.exists(MEMORY_DB_KEY)) {
    return null;
  }

  return ConnectionManager.get(MEMORY_DB_KEY);
}

function clipText(value: string, max = 700): string {
  if (value.length <= max) return value;
  return `${value.slice(0, max - 1)}...`;
}

function computeRecencyBoost(createdAt: string): number {
  const ageMs = Date.now() - new Date(createdAt).getTime();
  const ageDays = ageMs / (1000 * 60 * 60 * 24);
  return Math.max(0, 1 - ageDays / 30);
}

function computeContextWeight(
  origin: MemoryOrigin,
  currentOrigin: MemoryOrigin,
  contextScope: string,
  currentScope: string,
  memoryConstants: ReturnType<typeof getMemoryConstants>,
): number {
  if (origin === currentOrigin) return memoryConstants.MEMORY_CONTEXT_PRIMARY_WEIGHT;
  if (contextScope === currentScope) return 0.46;
  return memoryConstants.MEMORY_CONTEXT_SECONDARY_WEIGHT;
}

async function searchWithVec(
  queryText: string,
  agentId: MemoryAssistantId,
  sessionId: string,
  memoryConstants: ReturnType<typeof getMemoryConstants>,
): Promise<MemorySearchResult[]> {
  const db = getDb();
  if (!db) return [];

  const embedModel = await getMemoryEmbedModel();
  const embedding = await EmbeddingService.embed(queryText, embedModel, true);
  const vectorJson = JSON.stringify(Array.from(embedding.vector));

  const rows = await db.select<
    Array<{
      record_id: number;
      session_id: string;
      agent_id: MemoryAssistantId;
      origin: MemoryOrigin;
      context_scope: string;
      source_type: 'message' | 'summary';
      role: 'user' | 'assistant' | null;
      text: string;
      created_at: string;
      score: number;
    }>
  >(
    `SELECT
       r.id         AS record_id,
       r.session_id AS session_id,
       r.agent_id   AS agent_id,
       r.origin     AS origin,
       r.context_scope,
       r.source_type,
       r.role,
       r.text,
       r.created_at,
       (1.0 - distance) AS score
     FROM vec_chat_memory v
     JOIN chat_memory_records r ON r.id = v.record_id
     WHERE v.embedding MATCH $1
       AND k = $2
       AND r.embedding_model = $3
       AND r.agent_id = $4
       AND (r.session_id != $5 OR r.source_type = 'summary')
     ORDER BY distance ASC`,
    [vectorJson, memoryConstants.LTM_MAX_RESULTS * 3, embedModel, agentId, sessionId],
  );

  return (rows ?? []).map((row) => ({
    recordId: row.record_id,
    sessionId: row.session_id,
    agentId: row.agent_id,
    origin: row.origin,
    contextScope: row.context_scope,
    sourceType: row.source_type,
    role: row.role,
    text: row.text,
    score: Math.min(Math.max(Number(row.score), 0), 1),
    createdAt: row.created_at,
  }));
}

async function searchWithScalar(
  queryText: string,
  agentId: MemoryAssistantId,
  sessionId: string,
  _memoryConstants: ReturnType<typeof getMemoryConstants>,
): Promise<MemorySearchResult[]> {
  const db = getDb();
  if (!db) return [];

  const embedModel = await getMemoryEmbedModel();

  const countRows = await db.select<[{ c: number }]>(
    `SELECT COUNT(*) AS c FROM chat_memory_embeddings WHERE model = $1 AND agent_id = $2`,
    [embedModel, agentId],
  );
  if ((countRows[0]?.c ?? 0) === 0) return [];

  const embedding = await EmbeddingService.embed(queryText, embedModel, true);

  const rows = await db.select<
    Array<{
      record_id: number;
      session_id: string;
      agent_id: MemoryAssistantId;
      origin: MemoryOrigin;
      context_scope: string;
      source_type: 'message' | 'summary';
      role: 'user' | 'assistant' | null;
      text: string;
      created_at: string;
      vector_blob: string;
    }>
  >(
    `SELECT
       e.record_id,
       r.session_id,
       r.agent_id,
       r.origin,
       r.context_scope,
       r.source_type,
       r.role,
       r.text,
       r.created_at,
       e.vector_blob
     FROM chat_memory_embeddings e
     JOIN chat_memory_records r ON r.id = e.record_id
     WHERE e.model = $1
       AND r.agent_id = $2
       AND (r.session_id != $3 OR r.source_type = 'summary')
     ORDER BY r.created_at DESC
     LIMIT 96`,
    [embedModel, agentId, sessionId],
  );

  const scored: MemorySearchResult[] = [];

  for (const row of rows ?? []) {
    try {
      const vector = EmbeddingService.deserialize(row.vector_blob);
      const cosine = await EmbeddingService.cosineSimilarity(embedding.vector, vector);
      scored.push({
        recordId: row.record_id,
        sessionId: row.session_id,
        agentId: row.agent_id,
        origin: row.origin,
        contextScope: row.context_scope,
        sourceType: row.source_type,
        role: row.role,
        text: row.text,
        score: cosine,
        createdAt: row.created_at,
      });
    } catch (error) {
      log.warn('Scalar semantic memory skipped corrupted vector', error);
    }
  }

  return scored;
}

async function searchLongTermMemory(
  queryText: string,
  agentId: MemoryAssistantId,
  sessionId: string,
  memoryConstants: ReturnType<typeof getMemoryConstants>,
): Promise<MemorySearchResult[]> {
  try {
    const rows = await searchWithVec(queryText, agentId, sessionId, memoryConstants);
    if (rows.length > 0) return rows;
  } catch (error) {
    log.warn('vec search for chat memory failed, switching to scalar fallback', error);
  }

  return searchWithScalar(queryText, agentId, sessionId, memoryConstants);
}

export const MemoryContextService = {
  async buildContext(params: {
    sessionId: string;
    agentId: MemoryAssistantId;
    origin: MemoryOrigin;
    contextScope: string;
    userMessage: string;
  }): Promise<MemoryContextBundle> {
    // Получаем константы с учетом профиля машины
    const machineProfile = useOllamaStore.getState().machineProfile;
    const memoryConstants = getMemoryConstants(machineProfile);

    const stm = await ChatMemoryService.listRecentMessages(params.sessionId, memoryConstants.STM_MAX_MESSAGES);

    let stmTokens = 0;
    const history = [...stm]
      .reverse()
      .filter((message) => {
        stmTokens += message.token_estimate;
        return stmTokens <= memoryConstants.STM_MAX_TOKENS;
      })
      .reverse()
      .map((message) => ({
        role: message.role === 'user' ? 'user' : 'assistant',
        content: message.content,
      })) as MemoryContextBundle['history'];

    // Получаем индекс самого старого сообщения в STM
    const oldestStmIndex = stm.length > 0 ? stm[0].message_index : 0;

    const localSummaries = await ChatMemoryService.listRecentSummaries(
      params.sessionId,
      memoryConstants.MTM_MAX_SUMMARIES,
    );

    // Фильтруем суммаризации, чтобы не было дублирования с STM
    // и чтобы они покрывали историю до STM
    const compressedBlocks = localSummaries.filter(
      (s) => s.end_message_index < oldestStmIndex
    );
    const relatedSummaries = await ChatMemoryService.listRelevantSummaries({
      agentId: params.agentId,
      sessionId: params.sessionId,
      origin: params.origin,
      contextScope: params.contextScope,
      limit: memoryConstants.MTM_MAX_SUMMARIES,
    });

    const semanticMatches = await searchLongTermMemory(
      params.userMessage,
      params.agentId,
      params.sessionId,
      memoryConstants,
    ).catch((err) => {
      log.warn('LTM search failed, continuing without semantic context', err);
      return [] as MemorySearchResult[];
    });

    const rankedMatches = semanticMatches
      .map((item) => {
        const semanticScore = item.score * 0.58;
        const contextScore =
          computeContextWeight(item.origin, params.origin, item.contextScope, params.contextScope, memoryConstants) *
          0.26;
        const recencyScore = computeRecencyBoost(item.createdAt) * 0.16;
        return {
          ...item,
          blendedScore: semanticScore + contextScore + recencyScore,
        };
      })
      .sort((left, right) => right.blendedScore - left.blendedScore)
      .slice(0, memoryConstants.LTM_MAX_RESULTS);

    const parts: string[] = [];

    if (compressedBlocks.length > 0) {
      parts.push(
        'Compressed session history (MTM):\n' +
        compressedBlocks
          .map(
            (summary, index) =>
              `${index + 1}. [Messages ${summary.start_message_index}-${summary.end_message_index}] Summary: ${clipText(summary.summary_text, 420)}`,
          )
          .join('\n'),
      );
    }

    if (relatedSummaries.length > 0) {
      parts.push(
        'Relevant cross-window summaries:\n' +
        relatedSummaries
          .map(
            (summary, index) =>
              `${index + 1}. [${summary.origin}] ${clipText(summary.summary_text, 360)}`,
          )
          .join('\n'),
      );
    }

    if (rankedMatches.length > 0) {
      parts.push(
        'Long-term semantic memory:\n' +
        rankedMatches
          .map((match, index) => {
            const label =
              match.sourceType === 'summary'
                ? `Summary from ${match.origin} / ${match.sessionId}`
                : `${match.role === 'user' ? 'User' : 'Assistant'} message from ${match.origin} / ${match.sessionId}`;
            return `${index + 1}. [score ${(match.blendedScore * 100).toFixed(0)}] ${label}: ${clipText(match.text, 280)}`;
          })
          .join('\n'),
      );
    }

    return {
      history,
      systemContext: parts.join('\n\n'),
    };
  },
};
